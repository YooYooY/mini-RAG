from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph

Route = Literal["direct_answer", "needs_thinking"]

ROUTE_TO_NODE = {
    "direct_answer": "direct_answer",
    "needs_thinking": "thinking",
}


class AgentState(TypedDict):
    input: str
    messages: list[str]
    step_count: int
    route: Route


def analyze_input_node(state: AgentState) -> dict:
    text = state["input"]
    messages = state["messages"] + [f"[analyze] input = {text}"]

    route = "direct_answer" if len(text) < 15 else "needs_thinking"

    return {"messages": messages, "route": route, "step_count": state["step_count"] + 1}


def direct_answer_node(state: AgentState) -> dict:
    messages = state["messages"] + ["[answer] This is a direct answer."]
    return {"messages": messages, "step_count": state["step_count"] + 1}


def thinking_node(state: AgentState) -> dict:
    messages = state["messages"] + ["[thinking] I need to think step by step..."]
    return {"messages": messages, "step_count": state["step_count"] + 1}


def route_after_analysis(
    state: AgentState,
) -> Route:
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_input", analyze_input_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("thinking", thinking_node)

    graph.add_edge(START, "analyze_input")

    graph.add_conditional_edges(
        "analyze_input",
        route_after_analysis,
        ROUTE_TO_NODE,
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("thinking", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "input": "You can direct to answer my question",
            "messages": [],
            "step_count": 0,
            "route": "direct_answer",
        }
    )
    print(result)
