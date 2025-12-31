from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import TaskState


def query_rewrite_node(state: TaskState):
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    critic = state["critic_result"]
    rewrite_query = critic.get("rewrite_query")
    reason = critic.get("reason", "")

    if not rewrite_query:
        raise ValueError("query_rewrite_node called without rewrite_query")

    prev_rc = task_memory["retrieval_context"]

    # 归档上一轮
    hist = task_memory.setdefault("retrieval_history", [])
    hist.append(
        {
            "round": prev_rc["round"],
            "query": prev_rc["query"],
            "query_source": prev_rc["query_source"],
            "hits": prev_rc.get("retriever_hits", []),
            "critic": critic,
        }
    )

    # 新一轮
    new_round = prev_rc["round"] + 1

    task_memory["retrieval_context"] = {
        "round": new_round,
        "query": rewrite_query,
        "query_source": "query_rewrite",
        "retriever_hits": [],
    }

    state["retrieval_context"] = task_memory["retrieval_context"]

    append_trace(
        task_id,
        "query_rewrite_node",
        "critic_query_rewrite",
        {
            "new_round": new_round,
            "rewrite_query": rewrite_query,
            "rewrite_reason": reason,
        },
        {"current_query": rewrite_query},
        next_step="retriever_node",
    )

    save_checkpoint(task_id, state, "query_rewrite_node", "retriever_node")

    return state
