from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import TaskState


def fail_answer_node(state: TaskState, *, config):
    critic = state["critic_result"]

    reason = critic.get("reason", "")
    fail_type = critic.get("fail_type", "unknown")

    state[
        "answer"
    ] = f"""
❌ 检索失败（终止）

原因：{reason}
类型：{fail_type}
"""

    append_trace(
        state["task_id"],
        "fail_answer_node",
        "fail_safe_termination",
        {"fail_type": fail_type},
        {"reason": reason},
        next_step="end",
    )

    return state
