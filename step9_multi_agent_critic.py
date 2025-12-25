from __future__ import annotations

from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph

# ========= 常量 =========
Plan = Literal["retrieve", "rewrite", "answer", "end"]
Verdict = Literal["approve", "revise", "end"]

MAX_TURNS = 10
MAX_REWRITES = 2
MIN_DOCS_TO_ANSWER = 1

# quality valve（mock）
MIN_DOC_TOTAL_CHARS = 80


# ========= State =========
class AgentState(TypedDict):
    # user question, unchangeable
    question: str

    # retrieve query, (can be rewrite)
    query: str

    # cycle control
    turn_count: int
    rewrite_count: int

    # retreve content
    retrieved_docs: list[str]
    answer_draft: str
    final_answer: str | None

    # muti Agent collobrate
    plan: Plan
    verdict: Verdict
    critique: str | None

    # work trace
    messages: list[str]


# ========= Mock KB =========
KB = {
    "langgraph": (
        "LangGraph builds stateful, multi-step agent workflows as graphs. "
        "It supports conditional routing, loops, memory, and checkpoints."
    ),
    "react agent": (
        "ReAct combines reasoning and acting by iterating Thought -> Action -> Observation."
    ),
    "rag": (
        "RAG (Retrieval-Augmented Generation) retrieves relevant context first, then generates an answer."
    ),
    # "query rewrite": (
    #     "Query rewrite improves retrieval by transforming user questions into search-friendly queries."
    # ),
    # for critic stepback revise
    "weak": "Too short.",
}


def rag_retrieve(query: str) -> list[str]:
    q = query.lower()
    hits: list[str] = []
    for k, v in KB.items():
        if k in q:
            hits.append(v)
    return hits


def doc_quality_score(docs: list[str]) -> int:
    # mock quality：len total calcute
    return sum(len(d) for d in docs)


def generate_answer(question: str, docs: list[str]) -> str:
    # mock, will use llm at the ultimate
    if not docs:
        return f"I couldn't find relevant information in the knowledge base for: {question}"
    return f"Answer for: {question}\n\n" + "\n".join(f"- {d}" for d in docs)


# ========= Nodes =========
def planner_node(state: AgentState) -> dict:
    """
    Planner: decide plan（retrieve / rewrite / answer / end）
    """
    turn = state["turn_count"] + 1
    docs = state["retrieved_docs"]
    critique = state["critique"]

    if turn >= MAX_TURNS:
        plan: Plan = "end"
        msg = "[planner] max turns -> end"

    else:
        if critique == "need_rewrite" and state["rewrite_count"] < MAX_REWRITES:
            plan = "rewrite"
            msg = "[planner] critic asked rewrite -> rewrite"

        elif not docs:
            plan = "retrieve"
            msg = "[planner] no docs -> retrieve"

        else:
            plan = "answer"
            msg = "[planner] have docs -> answer"

    return {
        "plan": plan,
        "turn_count": turn,
        "messages": state["messages"] + [msg],
    }


def executor_node(state: AgentState) -> dict:
    """
    Executor：work by the plan
    """
    plan = state["plan"]
    msgs = state["messages"]

    if plan == "retrieve":
        docs = rag_retrieve(state["query"])
        msgs = msgs + [f"[executor] retrieve query='{state['query']}' hits={len(docs)}"]
        return {"retrieved_docs": docs, "messages": msgs}

    if plan == "rewrite":
        new_query = f"{state['query']} query rewrite {state['question']}".lower()
        msgs = msgs + [f"[executor] rewrite -> query='{new_query}'"]
        return {
            "query": new_query,
            "rewrite_count": state["rewrite_count"] + 1,
            "retrieved_docs": [],  # IMPORTANT rewrite
            "messages": msgs,
            "critique": None,  # IMPORTANT rewrite
        }

    if plan == "answer":
        draft = generate_answer(state["question"], state["retrieved_docs"])
        msgs = msgs + ["[executor] answer draft generated"]
        return {"answer_draft": draft, "messages": msgs}

    msgs = msgs + ["[executor] plan=end (no-op)"]
    return {"messages": msgs}


def critic_node(state: AgentState) -> dict:
    """
    Critic：only criti
    """
    plan = state["plan"]
    docs = state["retrieved_docs"]
    msgs = state["messages"]

    if plan == "end":
        verdict: Verdict = "end"
        return {
            "verdict": verdict,
            "messages": msgs + ["[critic] planner ended -> end"],
        }

    # 1) retrieve
    if plan == "retrieve":
        if not docs:
            if state["rewrite_count"] < MAX_REWRITES:
                verdict = "revise"
                critique = "need_rewrite"
                msgs = msgs + ["[critic] retrieve miss -> revise (need_rewrite)"]
                return {"verdict": verdict, "critique": critique, "messages": msgs}
            else:
                verdict = "end"
                msgs = msgs + ["[critic] retrieve miss and no rewrites left -> end"]
                return {"verdict": verdict, "final_answer": None, "messages": msgs}

        # if low quality -> revise
        score = doc_quality_score(docs)
        if score < MIN_DOC_TOTAL_CHARS and state["rewrite_count"] < MAX_REWRITES:
            verdict = "revise"
            critique = "need_rewrite"
            msgs = msgs + [
                f"[critic] weak evidence score={score}, min :{MIN_DOC_TOTAL_CHARS} -> revise"
            ]
            return {"verdict": verdict, "critique": critique, "messages": msgs}

        # let planner to answer
        verdict = "revise"
        critique = None
        msgs = msgs + [
            f"[critic] retrieve hit -> proceed (planner will answer) score={score}"
        ]
        return {"verdict": verdict, "critique": critique, "messages": msgs}

    # 2) answer: approve / revise / end
    if plan == "answer":
        if len(docs) < MIN_DOCS_TO_ANSWER:
            # evidence low -> revise
            if state["rewrite_count"] < MAX_REWRITES:
                verdict = "revise"
                critique = "need_rewrite"
                msgs = msgs + ["[critic] answer has insufficient docs -> revise"]
                return {"verdict": verdict, "critique": critique, "messages": msgs}
            verdict = "end"
            msgs = msgs + ["[critic] answer insufficient and no rewrites left -> end"]
            return {"verdict": verdict, "messages": msgs}

        verdict = "approve"
        msgs = msgs + ["[critic] approved"]
        return {
            "verdict": verdict,
            "final_answer": state["answer_draft"],
            "messages": msgs,
        }

    # rewrite
    if plan == "rewrite":
        verdict = "revise"
        msgs = msgs + ["[critic] rewrite done -> continue"]
        return {"verdict": verdict, "messages": msgs}

    # fallback
    return {"verdict": "end", "messages": msgs + ["[critic] unknown state -> end"]}


# ========= Routers =========
def route_after_planner(state: AgentState) -> Plan:
    return state["plan"]


def route_after_critic(state: AgentState) -> Verdict:
    return state["verdict"]


# ========= Build Graph =========
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner_node", planner_node)
    g.add_node("executor_node", executor_node)
    g.add_node("critic_node", critic_node)

    g.add_edge(START, "planner_node")

    g.add_conditional_edges(
        "planner_node",
        route_after_planner,
        {
            "retrieve": "executor_node",
            "rewrite": "executor_node",
            "answer": "executor_node",
            "end": END,
        },
    )

    g.add_edge("executor_node", "critic_node")

    g.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {
            "approve": END,
            "revise": "planner_node",
            "end": END,
        },
    )

    return g.compile()


# ========= Scenarios =========
def run_case(title: str, init: AgentState):
    print(f"\n\n==================== {title} ====================")
    app = build_graph()
    out = app.invoke(init)
    for m in out["messages"]:
        print(m)
    print("\nFINAL ANSWER:\n", out.get("final_answer"))


if __name__ == "__main__":

    # CASE 1:（retrieve -> answer -> approve）
    run_case(
        "CASE 1: direct hit",
        {
            "question": "What is LangGraph?",
            "query": "langgraph",
            "turn_count": 0,
            "rewrite_count": 0,
            "retrieved_docs": [],
            "answer_draft": "",
            "final_answer": None,
            "plan": "retrieve",
            "verdict": "revise",
            "critique": None,
            "messages": [],
        },
    )

    # CASE 2: miss at first，rewrite to hit（retrieve miss -> revise -> rewrite -> retrieve hit -> answer -> approve）
    # run_case(
    #     "CASE 2: miss then rewrite hit",
    #     {
    #         "question": "Explain how agent loops work in LangGraph",
    #         "query": "loops",  # KB lack keyword "loops"
    #         "turn_count": 0,
    #         "rewrite_count": 0,
    #         "retrieved_docs": [],
    #         "answer_draft": "",
    #         "final_answer": None,
    #         "plan": "retrieve",
    #         "verdict": "revise",
    #         "critique": None,
    #         "messages": [],
    #     },
    # )

    # CASE 3: miss forever, end（retrieve miss -> rewrite ... -> end）
    # run_case(
    #     "CASE 3: always miss -> end",
    #     {
    #         "question": "Explain embeddings in detail",
    #         "query": "embeddings",  # KB lack keyword "embeddings"
    #         "turn_count": 0,
    #         "rewrite_count": 0,
    #         "retrieved_docs": [],
    #         "answer_draft": "",
    #         "final_answer": None,
    #         "plan": "retrieve",
    #         "verdict": "revise",
    #         "critique": None,
    #         "messages": [],
    #     },
    # )

    # CASE 4: evidence weak， critic verdict rewrite（retrieve hit but weak -> rewrite -> retrieve -> answer）
    # run_case(
    #     "CASE 4: weak evidence -> revise -> rewrite",
    #     {
    #         "question": "What is weak in this KB?",
    #         "query": "weak",
    #         "turn_count": 0,
    #         "rewrite_count": 0,
    #         "retrieved_docs": [],
    #         "answer_draft": "",
    #         "final_answer": None,
    #         "plan": "retrieve",
    #         "verdict": "revise",
    #         "critique": None,
    #         "messages": [],
    #     },
    # )
