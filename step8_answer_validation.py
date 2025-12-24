from __future__ import annotations
from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph

# ========= 常量 =========
Route = Literal["pass", "fail", "retry", "end"]
MAX_REWRITES = 2
MIN_DOC_LENGTH = 50  # validation example


# ========= State =========
class AgentState(TypedDict):
    question: str

    # coming from Step 7
    query: str | None
    rewrite_count: int
    retrieved_docs: list[str]

    # Answer & Validation
    answer_draft: str | None
    validation_reason: str | None

    messages: list[str]
    route: Route


# ========= Mock Answer Generator =========
def generate_answer(docs: list[str]) -> str:
    return " ".join(docs)


# ========= Nodes =========
def answer_draft_node(state: AgentState) -> dict:
    answer = generate_answer(state["retrieved_docs"])
    messages = state["messages"] + [
        f"[answer_draft] generated answer draft, answer={answer}, len={len(answer)}"
    ]

    return {
        "answer_draft": answer,
        "messages": messages,
    }


def validate_answer_node(state: AgentState) -> dict:
    answer = state["answer_draft"] or ""
    docs = state["retrieved_docs"]

    if not docs:
        route: Route = "fail"
        reason = "no supporting documents"
    elif len(answer) < MIN_DOC_LENGTH:
        route = "fail"
        reason = "answer too short / weak evidence"
    else:
        route = "pass"
        reason = "answer validated"

    messages = state["messages"] + [f"[validate] route={route}, reason={reason}"]

    return {
        "route": route,
        "validation_reason": reason,
        "messages": messages,
    }


def retry_or_end_node(state: AgentState) -> dict:
    if state["rewrite_count"] < MAX_REWRITES:
        route: Route = "retry"
        messages = state["messages"] + ["[retry] validation failed, retrying"]
    else:
        route = "end"
        messages = state["messages"] + ["[end] validation failed, giving up"]

    return {
        "route": route,
        "messages": messages,
    }


# ========= Routers =========
def route_after_validation(state: AgentState) -> Route:
    return state["route"]


def route_after_retry(state: AgentState) -> Route:
    return state["route"]


# ========= Build Graph =========
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("answer_draft_node", answer_draft_node)
    graph.add_node("validate_answer_node", validate_answer_node)
    graph.add_node("retry_or_end_node", retry_or_end_node)

    graph.add_edge(START, "answer_draft_node")
    graph.add_edge("answer_draft_node", "validate_answer_node")

    graph.add_conditional_edges(
        "validate_answer_node",
        route_after_validation,
        {
            "pass": END,
            "fail": "retry_or_end_node",
        },
    )

    graph.add_conditional_edges(
        "retry_or_end_node",
        route_after_retry,
        {
            "retry": END,  # In real project：use Step7 rewrite
            "end": END,
        },
    )

    return graph.compile()


# ========= Run =========
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "question": "What is LangGraph?",
            "query": "LangGraph",
            "rewrite_count": 1,
            "retrieved_docs": [
                "LangGraph builds stateful,",
                "multi-step agent workflows as graphs",
            ],
            "answer_draft": None,
            "validation_reason": None,
            "messages": [],
            "route": "fail",
        }
    )

    print("\n=== TRACE ===")
    for m in result["messages"]:
        print(m)

    print("\nFINAL ANSWER:", result.get("answer_draft"))
    print("VALIDATION:", result.get("validation_reason"))
