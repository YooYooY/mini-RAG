from langgraph.graph import StateGraph, END
from core.types import TaskState

from critic.query_rewrite_node import query_rewrite_node
from planner.planner_node import planner_node
from retriever.retriever_node import retriever_node
from executor.executor_node import executor_node
from critic.critic_node import critic_node
from fallback.fail_answer_node import fail_answer_node


def build_graph():
    graph = StateGraph(TaskState)

    graph.add_node("planner_node", planner_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("executor_node", executor_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("query_rewrite_node", query_rewrite_node)
    graph.add_node("fail_answer_node", fail_answer_node)

    graph.set_entry_point("planner_node")

    graph.add_edge("planner_node", "retriever_node")
    graph.add_edge("retriever_node", "executor_node")
    graph.add_edge("executor_node", "critic_node")

    graph.add_edge("query_rewrite_node", "retriever_node")

    graph.add_conditional_edges(
        "critic_node",
        lambda s: s["critic_result"]["status"],
        {
            "pass": END,
            "revise_retry": "retriever_node",
            "revise_rewrite": "query_rewrite_node",
            "fail": "fail_answer_node",
        },
    )

    return graph.compile()
