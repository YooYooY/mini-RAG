from typing import Literal, TypedDict
import uuid
import json
import os

import dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from Checkpoint import has_checkpoint, resume_from_checkpoint, save_checkpoint

dotenv.load_dotenv()

# ============== TypedDict 定义 ==============


class IntentContext(TypedDict):
    topic: str
    intent: str
    task_plan: list[str]


class RetrievalContext(TypedDict):
    doc_scope: list[str]
    retriever_hits: list[dict]


class ExecutionTrace(TypedDict):
    step: str  # retriever / executor / critic / fail_answer
    tool: str  # doc_retriever / answer_generator / llm_critic / system_fallback
    input: dict
    output: dict | None
    status: str  # success / warning / error
    error: str | None
    critic_round: int  # 当前是第几轮 critic（第几次 revise）


class CriticResult(TypedDict):
    status: Literal["pass", "revise", "fail"]
    reason: str
    critic_count: int  # critic 调用次数（防止死循环）


class TaskState(TypedDict):
    task_id: str
    intent_context: IntentContext
    retrieval_context: RetrievalContext
    answer: str
    execution_trace: list[ExecutionTrace]
    critic_result: CriticResult


# ============== 全局 Memory & LLM ==============

memory_store: dict[str, dict] = {}

# 你可以按需换模型，比如 gpt-4.1 / gpt-4.1-mini
llm_critic = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


# ============== Trace 记录工具函数 ==============


def append_trace(
    task_id: str,
    step: str,
    tool: str,
    input_data: dict,
    output: dict | None = None,
    status: str = "success",
    error: str | None = None,
):
    critic_round = memory_store[task_id]["critic_result"]["critic_count"]

    trace_item: ExecutionTrace = {
        "step": step,
        "tool": tool,
        "input": input_data,
        "output": output,
        "status": status,
        "error": error,
        "critic_round": critic_round,
    }

    memory_store[task_id]["execution_trace"].append(trace_item)


def init_task_memory() -> str:
    task_id = str(uuid.uuid4())
    memory_store[task_id] = {
        "task_meta": {"task_id": task_id},
        "intent_context": {},
        "retrieval_context": {},
        "execution_trace": [],
        "critic_result": {"critic_count": 0, "status": "pass", "reason": ""},
    }
    return task_id


# ============== 各 Node 实现 ==============


def planner_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    topic = "API / orders"
    intent = "查询接口说明并生成示例代码"
    task_plan = ["检索文档", "抽取参数", "生成代码示例"]

    intent_context: IntentContext = {
        "topic": topic,
        "intent": intent,
        "task_plan": task_plan,
    }

    task_memory["intent_context"] = intent_context
    state["intent_context"] = intent_context

    return state


def retriever_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    query = "订单查询 API"

    # 这里为了 demo，让 hits 至少有一个“伪文档”，方便看到 pass 情况
    hits = [
        {
            "doc_id": "orders-api-001",
            "title": "订单查询接口说明",
            "score": 0.92,
            "chunk": "GET /api/orders/{order_id} 支持根据订单 ID 查询详细信息。",
        }
    ]

    retrieval_context: RetrievalContext = {
        "doc_scope": ["orders", "api"],
        "retriever_hits": hits,
    }

    task_memory["retrieval_context"] = retrieval_context

    append_trace(
        task_id=task_id,
        step="retriever",
        tool="doc_retriever",
        input_data={"query": query},
        output={"hits": hits},
    )

    state["retrieval_context"] = retrieval_context

    save_checkpoint(task_id, task_memory, state, "retriever_node")

    return state


def executor_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    intent = task_memory["intent_context"]["intent"]
    hits = task_memory["retrieval_context"]["retriever_hits"]

    answer = f"根据意图「{intent}」，基于 {len(hits)} 个文档生成的示例回答。"

    append_trace(
        task_id=task_id,
        step="executor",
        tool="answer_generator",
        input_data={"intent": intent, "hits_count": len(hits)},
        output={"answer": answer},
    )

    state["answer"] = answer

    save_checkpoint(task_id, task_memory, state, "executor_node")

    return state


# ============== LLM 语义 Critic ==============


def _run_llm_critic(payload: dict) -> CriticResult:
    """
    真正调用 LLM 的部分：输入 payload，要求 LLM 输出 JSON：
    {
      "status": "pass" | "revise" | "fail",
      "reason": "string"
    }
    """
    prompt = f"""
你是一个 API 文档问答系统的「语义审查员」（Critic）。

现在给你一轮问答流水线的信息，请你根据下面的规则进行评估，并严格输出 JSON：

----------------
【任务意图】
{payload["intent"]}

【任务主题】
{payload["topic"]}

【检索到的文档（只给前几条摘要）】
{payload["hits_preview"]}

【当前回答】
{payload["answer"]}

【静态规则检测发现的问题（可能为空）】
{payload["base_problems"]}
----------------

请根据以下标准判断：

1. status 字段（必须三选一）：
   - "pass": 检索的文档和回答在语义上基本吻合，信息完整，没有明显逻辑问题。
   - "revise": 回答有一定参考价值，但存在明显缺陷（例如文档不足、回答过于空泛），适合再走一轮「重新检索 / 优化回答」。
   - "fail": 明确无法基于当前文档给出可靠回答（文档严重不足、完全答非所问、或者静态规则强烈不通过）。

2. reason 字段（简要中文说明原因），可以参考静态规则发现的问题，但重点是你基于语义的判断。

⚠️ 输出要求：
- 只输出 JSON
- 不要任何额外说明或自然语言
- JSON 形如：
  {{"status": "revise", "reason": "xxx"}}
"""

    resp = llm_critic.invoke(prompt)
    content = resp.content.strip()

    print("=====> prompt", prompt)
    print("-----> resp", resp)

    # 尝试解析 JSON，失败就 fallback
    try:
        data = json.loads(content)
        status = data.get("status", "fail")
        reason = data.get("reason", "LLM critic 未返回 reason。")
    except Exception as e:
        status = "fail"
        reason = f"LLM critic 解析失败：{e}；原始内容：{content[:200]}"

    # 返回一个基础 CriticResult（不包含 critic_count，由外层补）
    return {
        "status": status,  # type: ignore
        "reason": reason,
        "critic_count": 0,  # 这里先占位，外层会覆盖
    }


def critic_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    intent_ctx: IntentContext = task_memory["intent_context"]
    retrieval_ctx: RetrievalContext = task_memory["retrieval_context"]
    hits = retrieval_ctx.get("retriever_hits", [])
    trace = task_memory["execution_trace"]
    critic_count = task_memory["critic_result"]["critic_count"]

    base_problems: list[str] = []

    # 一些静态规则（不需要 LLM）
    if len(hits) == 0:
        base_problems.append("retriever returned no documents")

    executed_steps = [t["step"] for t in trace]
    if "executor" not in executed_steps:
        base_problems.append("executor was never called")

    if not state.get("answer"):
        base_problems.append("no answer was generated")

    # 1. 强制上限：防止死循环
    if critic_count >= 2:
        reason = "critic count exceeded; " + "; ".join(base_problems)
        critic: CriticResult = {
            "status": "fail",
            "reason": reason,
            "critic_count": critic_count,
        }

        append_trace(
            task_id=task_id,
            step="critic",
            tool="llm_critic",
            input_data={"base_problems": base_problems, "force_fail": True},
            output={"critic_result": critic},
            status="warning",
            error=None,
        )

        task_memory["critic_result"] = critic
        state["critic_result"] = critic
        return state

    # 2. 正常情况：调用 LLM 做语义审查
    #    只把 docs 做一个 preview，避免 prompt 太长
    hits_preview = [
        {
            "doc_id": h.get("doc_id"),
            "title": h.get("title"),
            "score": h.get("score"),
        }
        for h in hits[:3]
    ]

    critic_payload = {
        "topic": intent_ctx.get("topic", ""),
        "intent": intent_ctx.get("intent", ""),
        "hits_preview": hits_preview,
        "answer": state.get("answer", ""),
        "base_problems": base_problems,
    }

    llm_result = _run_llm_critic(critic_payload)

    status = llm_result["status"]
    reason = llm_result["reason"]

    # 更新 critic_count 逻辑：
    if status == "pass":
        new_critic_count = 0
    else:
        new_critic_count = critic_count + 1

    critic: CriticResult = {
        "status": status,
        "reason": reason,
        "critic_count": new_critic_count,
    }

    append_trace(
        task_id=task_id,
        step="critic",
        tool="llm_critic",
        input_data=critic_payload,
        output={"critic_result": critic},
        status="success",
        error=None,
    )

    task_memory["critic_result"] = critic
    state["critic_result"] = critic

    save_checkpoint(task_id, task_memory, state, "critic_node")

    return state


def fail_answer_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    critic = task_memory["critic_result"]
    reason = critic.get("reason", "unknown error")

    answer = "⚠️ 当前查询未能成功处理（已终止）。\n" "原因（供内部排查）：" f"{reason}"

    append_trace(
        task_id=task_id,
        step="fail_answer",
        tool="system_fallback",
        input_data={"critic": critic},
        output={"answer": answer},
        status="warning",
    )

    state["answer"] = answer

    save_checkpoint(task_id, task_memory, state, "fail_answer_node")
    return state


# ============== 路由逻辑 ==============


def route_after_critic(state: TaskState) -> str:
    status = state["critic_result"]["status"]

    if status == "pass":
        return "end"

    if status == "revise":
        return "retriever"

    if status == "fail":
        return "fail_answer"

    # 理论上不会走到这
    return "fail_answer"


# ============== 构建 Graph ==============


def build_graph():
    graph = StateGraph(TaskState)

    graph.add_node("planner_node", planner_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("executor_node", executor_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("fail_answer_node", fail_answer_node)

    graph.set_entry_point("planner_node")

    graph.add_edge("planner_node", "retriever_node")
    graph.add_edge("retriever_node", "executor_node")
    graph.add_edge("executor_node", "critic_node")

    graph.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {
            "retriever": "retriever_node",
            "fail_answer": "fail_answer_node",
            "end": END,
        },
    )

    # 显式声明 fail 节点结束
    graph.add_edge("fail_answer_node", END)

    return graph.compile()


# ============== Demo 入口 ==============


def create_init_state(task_id: str):
    return {
        "task_id": task_id,
        "intent_context": {},
        "retrieval_context": {},
        "answer": "",
        "execution_trace": [],
        "critic_result": {"critic_count": 0, "status": "pass", "reason": ""},
    }


if __name__ == "__main__":
    app = build_graph()

    test_task_id = "3e6d0b6c-221a-4b66-a293-31e9f92e00a1"

    if has_checkpoint(test_task_id):
        result = resume_from_checkpoint(app, test_task_id, memory_store)
        task_id = test_task_id
    else:
        task_id = init_task_memory()
        init_state = create_init_state(task_id)
        result = app.invoke(init_state)

    print("\n=== 最终答案 ===")
    print(result["answer"])

    print("\n=== 任务级 Memory ===")
    print(memory_store[task_id])

    print("\n=== Execution Trace ===")
    for step in memory_store[task_id]["execution_trace"]:
        print(f"\n=== STEP:{step["step"]} ===")
        print(step)
