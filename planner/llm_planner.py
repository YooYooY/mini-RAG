import json
from core.types import IntentContext


def run_llm_planner(llm, user_query: str) -> IntentContext:

    prompt = f"""
You are a Task Planner for an AskMyDocs RAG Agent.

Your job:
Understand the user query and produce a structured task plan.

User Query:
{user_query}

You MUST output JSON only, in this schema:

topic: short name of subject domain (Chinese allowed)

intent: concise natural language description

Example output:
{{
  "topic": "订单系统",
  "intent": "查询订单接口并理解参数含义",
}}

Output ONLY valid JSON.
"""

    resp = llm.invoke(prompt)

    try:
        result = json.loads(resp.content)
    except Exception:
        result = {
            "topic": "未知主题",
            "intent": user_query,
        }

    return result
