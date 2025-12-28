from typing import TypedDict, Literal, List, Dict, Optional
import uuid
import json
import os
from langgraph.graph import StateGraph, END


# ============ 任务状态类型定义 ============


class IntentContext(TypedDict, total=False):
    topic: str
    intent: str
    task_plan: List[str]


class RetrievalContext(TypedDict, total=False):
    query: str
    doc_scope: List[str]
    retriever_hits: List[Dict]


class ExecutionTrace(TypedDict, total=False):
    step: str
    tool: str
    input: Dict
    output: Optional[Dict]
    status: str  # success / warning / error
    error: Optional[str]
    critic_round: int
    next_step: Optional[str]  # ⭐︎ 关键：记录“下一跳”节点


class CriticResult(TypedDict, total=False):
    status: Literal["pass", "revise", "fail"]
    reason: str
    critic_count: int


class TaskState(TypedDict, total=False):
    task_id: str
    intent_context: IntentContext
    retrieval_context: RetrievalContext
    answer: str
    execution_trace: List[ExecutionTrace]
    critic_result: CriticResult
    # 用于入口路由（可选）
    resume_next_step: Optional[str]


# ============ 全局内存（任务级 Memory） ============

memory_store: Dict[str, Dict] = {}

# ============ Checkpoint 存储 ============

CHECKPOINT_DIR = "./checkpoints"


def ensure_checkpoint_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def checkpoint_path(task_id: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{task_id}.json")


def save_checkpoint(task_id: str, state: TaskState, last_step: str, next_step: str):
    """保存当前任务的 checkpoint（带上下一跳信息）"""
    ensure_checkpoint_dir()
    payload = {
        "task_id": task_id,
        "last_step": last_step,
        "next_step": next_step,
        "state": state,
        "memory": memory_store[task_id],
    }
    with open(checkpoint_path(task_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_checkpoint(task_id: str):
    path = checkpoint_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_checkpoint(task_id: str) -> bool:
    return os.path.exists(checkpoint_path(task_id))


# ============ Trace 工具函数 ============


def append_trace(
    task_id: str,
    step: str,
    tool: str,
    input_data: Dict,
    output: Optional[Dict] = None,
    status: str = "success",
    error: Optional[str] = None,
    next_step: Optional[str] = None,
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
        "next_step": next_step,
    }

    memory_store[task_id]["execution_trace"].append(trace_item)


# ============ 任务初始化 ============


def init_task_memory(task_id: Optional[str] = None) -> str:
    if task_id is None:
        task_id = str(uuid.uuid4())
    memory_store[task_id] = {
        "task_meta": {"task_id": task_id},
        "intent_context": {},
        "retrieval_context": {},
        "execution_trace": [],
        "critic_result": {"critic_count": 0, "status": "pass", "reason": ""},
    }
    return task_id


def create_init_state(task_id: str) -> TaskState:
    return TaskState(
        task_id=task_id,
        intent_context={},
        retrieval_context={},
        answer="",
        execution_trace=[],
        critic_result={"critic_count": 0, "status": "pass", "reason": ""},
    )


# ============ 各 Node 实现 ============


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
    # planner 执行完后，下一跳是 retriever_node
    next_step = "retriever_node"

    append_trace(
        task_id=task_id,
        step="planner_node",
        tool="intent_planner",
        input_data={"topic": topic},
        output={"intent_context": intent_context},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="planner_node", next_step=next_step)

    return state


def retriever_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    # 这里用 mock 检索，你可以换成 Chroma 版本
    query = "订单查询 API"
    hits: List[Dict] = [
        {
            "doc_id": "orders-api-001",
            "title": "订单查询接口",
            "chunk": "GET /api/orders/{order_id} ...",
            "score": 0.9,
        }
    ]

    retrieval_context: RetrievalContext = {
        "query": query,
        "doc_scope": ["orders", "api"],
        "retriever_hits": hits,
    }

    task_memory["retrieval_context"] = retrieval_context
    state["retrieval_context"] = retrieval_context

    next_step = "executor_node"

    append_trace(
        task_id=task_id,
        step="retriever_node",
        tool="mock_vector_retriever",
        input_data={"query": query},
        output={"hit_count": len(hits)},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="retriever_node", next_step=next_step)

    return state


def executor_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    intent = task_memory["intent_context"].get("intent", "")
    hits = task_memory["retrieval_context"].get("retriever_hits", [])

    answer = f"根据意图「{intent}」，基于 {len(hits)} 个文档生成的示例回答（这里省略真正的 LLM 调用）。"

    state["answer"] = answer

    next_step = "critic_node"

    append_trace(
        task_id=task_id,
        step="executor_node",
        tool="answer_generator",
        input_data={"intent": intent, "hit_count": len(hits)},
        output={"answer": answer},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="executor_node", next_step=next_step)

    return state


def critic_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    hits = task_memory["retrieval_context"].get("retriever_hits", [])
    critic_count = task_memory["critic_result"]["critic_count"]
    problems = []

    if len(hits) == 0:
        problems.append("retriever returned no documents")

    if not state.get("answer"):
        problems.append("no answer was generated")

    # 简单规则版 critic：首次通过，第二次起 fail
    if critic_count >= 2:
        status: Literal["fail"] = "fail"
        reason = "critic count exceeded; " + "; ".join(problems)
    elif problems:
        status = "revise"
        reason = "; ".join(problems)
    else:
        status = "pass"
        reason = "pipeline executed correctly"

    # critic_count 更新
    if status == "pass":
        new_critic_count = 0
    else:
        new_critic_count = critic_count + 1

    critic_result: CriticResult = {
        "status": status,
        "reason": reason,
        "critic_count": new_critic_count,
    }

    task_memory["critic_result"] = critic_result
    state["critic_result"] = critic_result

    # ⭐ 根据 critic_result 决定下一跳（写进 trace & checkpoint）
    if status == "pass":
        next_step = "end"
    elif status == "revise":
        next_step = "retriever_node"
    else:  # fail
        next_step = "fail_answer_node"

    append_trace(
        task_id=task_id,
        step="critic_node",
        tool="rule_based_critic",
        input_data={"hit_count": len(hits)},
        output={"critic_result": critic_result},
        status="success",
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="critic_node", next_step=next_step)

    return state


def fail_answer_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    critic = task_memory["critic_result"]
    reason = critic.get("reason", "unknown error")

    answer = "⚠️ 当前查询未能成功处理（已终止）。\n" f"原因：{reason}"

    state["answer"] = answer

    next_step = "end"

    append_trace(
        task_id=task_id,
        step="fail_answer_node",
        tool="system_fallback",
        input_data={"critic": critic},
        output={"answer": answer},
        status="warning",
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="fail_answer_node", next_step=next_step)

    return state


# ============ Trace-Driven Resume 入口节点 ============


def entry_node(state: TaskState) -> TaskState:
    """
    统一入口：
    - 如果是新任务：没有 checkpoint，走 planner_node
    - 如果是恢复任务：外层会把 resume_next_step 填好，我们只保留即可
    """
    return state


def route_from_entry(state: TaskState) -> str:
    """
    根据 state.resume_next_step 决定真正的起始节点。
    - 新任务：resume_next_step 不存在 → planner_node
    - 恢复任务：resume_next_step 由 checkpoint 决定
    """
    resume_next = state.get("resume_next_step")
    if not resume_next:
        return "planner_node"
    return resume_next


# ============ 构建 Graph ============


def build_graph():
    graph = StateGraph(TaskState)

    graph.add_node("entry_node", entry_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("executor_node", executor_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("fail_answer_node", fail_answer_node)

    graph.set_entry_point("entry_node")

    # entry → 动态路由
    graph.add_conditional_edges(
        "entry_node",
        route_from_entry,
        {
            "planner_node": "planner_node",
            "retriever_node": "retriever_node",
            "executor_node": "executor_node",
            "critic_node": "critic_node",
            "fail_answer_node": "fail_answer_node",
            "end": END,
        },
    )

    # 其余节点按正常拓扑连接
    graph.add_edge("planner_node", "retriever_node")
    graph.add_edge("retriever_node", "executor_node")
    graph.add_edge("executor_node", "critic_node")

    graph.add_conditional_edges(
        "critic_node",
        lambda s: {
            "pass": "end",
            "revise": "retriever_node",
            "fail": "fail_answer_node",
        }[s["critic_result"]["status"]],
        {
            "retriever_node": "retriever_node",
            "fail_answer_node": "fail_answer_node",
            "end": END,
        },
    )

    graph.add_edge("fail_answer_node", END)

    return graph.compile()


# ============ Trace-Driven Resume 封装 ============


def resume_from_checkpoint(app, task_id: str) -> Optional[TaskState]:
    ckpt = load_checkpoint(task_id)
    if not ckpt:
        print(f"❌ 没有找到任务 {task_id} 的 checkpoint")
        return None

    print(f"🔁 从 checkpoint 恢复任务: {task_id}")
    print(f"   last_step: {ckpt['last_step']}")
    print(f"   next_step: {ckpt['next_step']}")

    # 恢复 memory
    memory_store[task_id] = ckpt["memory"]

    # 用 checkpoint 的 state 作为输入，但额外加上 resume_next_step，让 entry_node 做路由
    state: TaskState = ckpt["state"]
    state["resume_next_step"] = ckpt["next_step"]

    result: TaskState = app.invoke(state)
    return result


# ============ Demo 入口 ============

if __name__ == "__main__":
    app = build_graph()

    # 你可以用已有的 task_id 试，还可以先跑一次拿到新的 task_id
    # 这里为了演示，我们先新建一个任务 → 跑一遍 → 再用同一个 task_id 恢复

    print("=== 第一次执行（新任务） ===")
    task_id = init_task_memory()
    init_state = create_init_state(task_id)

    # 新任务执行（从 planner 开始）
    result1 = app.invoke(init_state)

    print("\n[Run1] 最终答案：")
    print(result1["answer"])

    print("\n[Run1] Execution Trace：")
    for step in memory_store[task_id]["execution_trace"]:
        print(f"- {step['step']} -> next: {step.get('next_step')}")

    # # 模拟“中断后恢复”——使用同一个 task_id
    # print("\n\n=== 从 checkpoint 恢复执行（Trace-Driven Resume） ===")
    # task_id = "3644bfe5-e685-442e-a461-df37e92e6769"
    # result2 = resume_from_checkpoint(app, task_id)

    # if result2:
    #     print("\n[Run2] 最终答案：")
    #     print(result2["answer"])

    #     print("\n[Run2] Execution Trace（追加后的）：")
    #     for step in memory_store[task_id]["execution_trace"]:
    #         print(f"- {step['step']} -> next: {step.get('next_step')}")
