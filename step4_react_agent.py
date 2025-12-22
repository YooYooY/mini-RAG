from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph

Route = Literal["continue", "end"]
ActionType = Literal["search", "calculator", "none"]

MAX_STEPS = 10


# ========= State =========
class AgentState(TypedDict):
    input: str
    messages: list[str]
    step_count: int
    route: Route

    # ReAct key props
    thought: str
    action: ActionType
    observation: str


# ========= Mock Tool =========
def search_tool(query: str) -> str:
    return f"[tool: seatch] Found results about: '{query}'"


def calculator_tool(expression: str) -> str:
    return f"[tool: calculator] Result of '{expression}' is 42"


# ========= Nodes =========
def thinking_nodes(state: AgentState) -> dict:
    thought = f"I need to answer the question: {state['input']}"

    if "calculate" in state["input"]:
        action: ActionType = "calculator"
    elif "search" in state["input"]:
        action: ActionType = "search"
    else:
        action = "none"

    messages = state["messages"] + [f"[thought] {thought}"]

    return {
        "messages": messages,
        "thought": thought,
        "action": action,
        "step_count": state["step_count"] + 1,
    }


def action_node(state: AgentState) -> dict:
    action = state["action"]
    if action == "calculator":
        observation = calculator_tool(state["input"])
    elif action == "search":
        observation = search_tool(state["input"])
    else:
        observation = "No tool needed."

    messages = state["messages"] + [f"[action] {action}"]

    return {
        "messages": messages,
        "observation": observation,
        "step_count": state["step_count"] + 1,
    }


def observation_node(state: AgentState) -> dict:
    # return observation result
    messages = state["messages"] + [f"[observation] {state['observation']}"]

    return {
        "messages": messages,
        "step_count": state["step_count"] + 1,
    }


def decide_node(state: AgentState) -> dict:
    if state["step_count"] >= MAX_STEPS or state["action"] == "none":
        route: Route = "end"
        messages = state["messages"] + ["[decide] I can answer now."]
    else:
        route: Route = "continue"
        messages = state["messages"] + ["[decide] I need another round."]

    return {
        "messages": messages,
        "route": route,
        "step_count": state["step_count"] + 1,
    }


def route_after_decide(state: AgentState) -> Route:
    return state["route"]


# ========= Build Graph =========
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("thinking_nodes", thinking_nodes)
    graph.add_node("action_node", action_node)
    graph.add_node("observation_node", observation_node)
    graph.add_node("decide_node", decide_node)

    graph.add_edge(START, "thinking_nodes")
    graph.add_edge("thinking_nodes", "action_node")
    graph.add_edge("action_node", "observation_node")
    graph.add_edge("observation_node", "decide_node")

    graph.add_conditional_edges(
        "decide_node", route_after_decide, {"continue": "thinking_nodes", "end": END}
    )

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "input": "search LangGraph ReAct example",
            "messages": [],
            "step_count": 0,
            "route": "continue",
            "thought": "",
            "action": "none",
            "observation": "",
        }
    )

    for m in result["messages"]:
        print(m)
