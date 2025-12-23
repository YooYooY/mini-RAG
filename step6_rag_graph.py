from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph

# ========= Constant =========
Route = Literal["answer", "rewrite", "end"]
MAX_STEPS = 10


# ========= State =========
class AgentState(TypedDict):
    question: str
    rewritten_question: str | None

    retrieved_docs: list[str]
    messages: list[str]

    step_count: int
    route: Route
    final_answer: str | None


# ========= Mock Knowledge Base =========
KB = {
    "langgraph": "LangGraph is a library for building stateful, multi-step agents.",
    "rag": "RAG stands for Retrieval-Augmented Generation.",
    "agent": "An agent can think, act, and decide using tools.",
}


def rag_retrieve(query: str) -> list[str]:
    hits = []
    for key, doc in KB.items():
        if key in query.lower():
            hits.append(doc)
    return hits


# ========= Nodes =========
def thinking_node(state: AgentState) -> dict:
    query = state["rewritten_question"] or state["question"]
    messages = state["messages"] + [f"[Thingking question] {query}"]

    return {"messages": messages, "step_count": state["step_count"] + 1}


def retrieve_node(state: AgentState) -> dict:
    query = state["rewritten_question"] or state["question"]
    retrieved_docs = rag_retrieve(query)
    messages = state["messages"] + [f"[retrieve] Found {len(retrieved_docs)} docs"]

    return {
        "retrieved_docs": retrieved_docs,
        "messages": messages,
        "step_count": state["step_count"] + 1,
    }


def observation_node(state: AgentState) -> dict:

    retrieved_docs = state["retrieved_docs"]

    if retrieved_docs:
        route = "answer"
        final_answer = " ".join(state["retrieved_docs"])
        messages = state["messages"] + ["[observation] Docs are relevant."]
    else:
        route = "rewrite"
        final_answer = None
        messages = state["messages"] + ["[observation] No docs found."]

    return {
        "route": route,
        "final_answer": final_answer,
        "messages": messages,
        "step_count": state["step_count"] + 1,
    }


def rewrite_node(state: AgentState) -> dict:
    """
    Step Back / Query Rewrite (mock)
    """
    rewritten = state["question"] + " agent"

    messages = state["messages"] + [f"[rewrite] rewrite question as {rewritten}"]

    return {
        "rewritten_question": rewritten,
        "messages": messages,
        "step_count": state["step_count"] + 1,
    }


def decide_node(state: AgentState) -> dict:
    route = state["route"]
    if route == "answer":
        route = "end"
        messages = state["messages"] + [f"[decide] {route}"]
    elif state["step_count"] >= MAX_STEPS:
        route = "end"
        messages = state["messages"] + ["[decide] Max steps reached. Fallback."]
    else:
        route = "rewrite"
        messages = state["messages"] + ["[decide] continue retrieved."]

    return {
        "route": route,
        "messages": messages,
        "step_count": state["step_count"] + 1,
    }


# ========= Router =========
def route_after_decide(state: AgentState) -> Route:
    return state["route"]


# ========= Build Graph =========
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("thinking_node", thinking_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("observation_node", observation_node)
    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("decide_node", decide_node)

    graph.add_edge(START, "thinking_node")
    graph.add_edge("thinking_node", "retrieve_node")
    graph.add_edge("retrieve_node", "observation_node")
    graph.add_edge("observation_node", "decide_node")
    graph.add_edge("rewrite_node", "thinking_node")

    graph.add_conditional_edges(
        "decide_node",
        route_after_decide,
        {"answer": END, "end": END, "rewrite": "rewrite_node"},
    )

    return graph.compile()


# ========= Run =========
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            # "question": "What is LangGraph?",
            "question": "What is your name?",
            "rewritten_question": None,
            "retrieved_docs": [],
            "messages": [],
            "step_count": 0,
            "route": "rewrite",
            "final_answer": None,
        }
    )

    print("\n=== TRACE ===")
    for msg in result["messages"]:
        print(msg)

    print("\nFINAL ANSWER:", result["final_answer"])
