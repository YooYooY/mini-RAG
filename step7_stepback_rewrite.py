from __future__ import annotations

from email import message
from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph

# ========= Constant =========
Route = Literal["hit", "miss", "retry", "end"]
MAX_REWRITES = 5


# ========= State =========
class AgentState(TypedDict):
    question: str

    # StepBack / Rewrite
    stepback_question: str | None
    query: str | None
    rewrite_count: int

    # RAG
    retrieved_docs: list[str]

    # Out and trace
    messages: list[str]
    route: Route
    final_answer: str | None


# ========= Mock Knowledge Base =========
KB = {
    "langgraph_mock": "LangGraph builds stateful, multi-step agent workflows as graphs.",
    "retrieval augmented generation": "RAG retrieves relevant context then generates an answer.",
    "react agent": "ReAct combines reasoning (thought) and acting (tool use) in loops.",
    "query rewrite": "Query rewrite transforms a user question into a better retrieval query.",
    "What is the definition": "From custom defintion.",
}


def rag_retrieve(query: str) -> list[str]:
    q = query.lower()
    hits = []
    for key, doc in KB.items():
        if key in q:
            hits.append(doc)
    return hits


# ========= Mock “LLM” for StepBack/Rewrite =========
def stepback(question: str) -> str:
    q = question.lower()

    if "how" in q or "enable" in q:
        return "What is LangGraph and how does it work?"
    if "difference" in q:
        return "What are the difference between related concepts?"
    return "What is the definition and key idea of the concept?"


def rewrite_query(question: str, stepback_q: str, attempt: int) -> str:
    q = question.lower()

    if attempt == 0:
        if "langgraph" in q:
            return "LangGraph"
        if "rag" in q:
            return "retrieval augmented generation"
        if "react" in q or "re-act" in q:
            return "ReAct agent"

    if attempt == 1:
        if "langgraph" in q:
            return "LangGraph stateful multi-step agent workflows"
        return stepback_q + " definition"

    return stepback_q + " langgraph_mock"


# ========= Nodes =========
def plan_query_node(state: AgentState) -> dict:
    attempt = state["rewrite_count"]
    sb = state["stepback_question"] or stepback(state["question"])
    query = rewrite_query(state["question"], sb, attempt)

    messages = state["messages"] + [
        f"========= 🌰 START ============",
        f"[attempt] {attempt}",
        f"[stepback] {sb}",
        f"[rewrite] {query}",
    ]

    return {
        "messages": messages,
        "query": query,
        "stepback_question": sb,
        "rewrite_count": attempt + 1,
    }


def retrieve_node(state: AgentState) -> dict:
    query = state["query"] or state["question"]
    docs = rag_retrieve(query)

    messages = state["messages"] + [f"[retrieve] query='{query}' hits={len(docs)}"]

    return {"retrieved_docs": docs, "messages": messages}


def judge_hit_node(state: AgentState) -> dict:
    if state["retrieved_docs"]:
        route = "hit"
        messages = state["messages"] + ["[judge] hit!✌️"]
    else:
        route = "miss"
        messages = state["messages"] + ["[judge] miss!😭"]

    return {"messages": messages, "route": route}


def maybe_retry_node(state: AgentState) -> dict:
    if state["rewrite_count"] <= MAX_REWRITES:
        route: Route = "retry"
        messages = state["messages"] + ["[retry] will rewrite and try again"]
    else:
        route = "end"
        messages = state["messages"] + ["[retry] exceeded max rewrites, ending"]

    return {"route": route, "messages": messages}


def answer_node(state: AgentState) -> dict:
    final_answer = (
        " ".join(state["retrieved_docs"]) if state["retrieved_docs"] else None
    )
    messages = state["messages"] + ["[answer] composed from retrieved docs"]

    return {"final_answer": final_answer, "messages": messages, "route": "end"}


# ========= Routers =========
def route_after_judge(state: AgentState) -> Route:
    return state["route"]


def route_after_retry(state: AgentState) -> Route:
    return state["route"]


# ========= Build Graph =========
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan_query_node", plan_query_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("judge_hit_node", judge_hit_node)
    graph.add_node("answer_node", answer_node)
    graph.add_node("maybe_retry_node", maybe_retry_node)

    # Linear edges
    graph.add_edge(START, "plan_query_node")
    graph.add_edge("plan_query_node", "retrieve_node")
    graph.add_edge("retrieve_node", "judge_hit_node")

    # Conditional: hit/miss
    graph.add_conditional_edges(
        "judge_hit_node",
        route_after_judge,
        {
            "hit": "answer_node",
            "miss": "maybe_retry_node",
        },
    )

    # Conditional: retry/end
    graph.add_conditional_edges(
        "maybe_retry_node",
        route_after_retry,
        {
            "retry": "plan_query_node",
            "end": END,
        },
    )

    # answer -> end
    graph.add_edge("answer_node", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "question": "Explain how LangGraph enables agent loops",
            "stepback_question": None,
            "query": None,
            "rewrite_count": 0,
            "retrieved_docs": [],
            "messages": [],
            "route": "miss",
            "final_answer": None,
        }
    )

    print("\n=== TRACE ===")
    for msg in result["messages"]:
        print(msg)

    print("\nFINAL ANSWER:", result["final_answer"])
