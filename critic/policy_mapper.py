MAX_RETRY = 2


def map_critic_policy(semantic_result, critic_count: int):
    status = semantic_result["status"]
    action = semantic_result.get("action")
    rewrite_query = semantic_result.get("rewrite_query")

    # 超过最大 retry → fail
    if critic_count >= MAX_RETRY:
        return {
            "status": "fail",
            "reason": "critic retry limit exceeded",
            "critic_count": critic_count,
        }

    # --- PASS ---
    if status == "pass":
        return {
            "status": "pass",
            "reason": semantic_result.get("reason", ""),
            "critic_count": 0,
        }

    # --- QUERY REWRITE ---
    if action == "query_rewrite" and rewrite_query:
        return {
            "status": "revise_rewrite",
            "reason": semantic_result.get("reason", ""),
            "rewrite_query": rewrite_query,
            "critic_count": critic_count + 1,
        }

    # --- RETRIEVER RETRY ---
    if action == "redo_retriever":
        return {
            "status": "revise_retry",
            "reason": semantic_result.get("reason", ""),
            "critic_count": critic_count + 1,
        }

    # --- OTHERWISE FAIL ---
    return {
        "status": "fail",
        "reason": semantic_result.get("reason", "unrecoverable"),
        "critic_count": critic_count + 1,
    }
