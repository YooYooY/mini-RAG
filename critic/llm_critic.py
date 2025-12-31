import dotenv
import json
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

dotenv.load_dotenv()


def run_llm_critic(
    user_query: str,
    intent: str,
    hits: list[dict],
    draft_answer: str,
    llm,
):
    """
    返回一个结构化裁决结果（LLM 助手角色 only）
    - 支持 query_rewrite
    """

    evidence_text = "\n\n".join(
        f"[{h['title']}]\n{h['chunk']}\n(score={h['score']})"
        for h in hits[:3]  # 控制长度
    )

    prompt = f"""
You are a retrieval quality auditor for an AskMyDocs RAG system.

Task:
Evaluate whether the retrieved evidence is semantically relevant
to the user query and the planned task intent.
If the evidence is clearly from the wrong domain (for example,
APIs about orders but the user is asking about UI layout),
you should prefer to fix the query (query_rewrite),
not just redo the retriever.

User Query:
{user_query}

Task Intent:
{intent}

Draft Answer (generated from evidence):
{draft_answer}

Retrieved Evidence Chunks:
{evidence_text}

You must reason carefully and output a structured JSON with fields:

- status:
  - "pass"   → evidence matches query & task domain
  - "revise" → evidence seems partially relevant or mismatched, try to repair
  - "fail"   → critically wrong retrieval or cannot recover

- reason:
  short natural language justification

- action:
  - "redo_retriever" → keep the same query, but maybe change top_k or other params
  - "query_rewrite"  → the current query is not suitable, propose a better short query
  - "stop"           → give up, cannot be repaired
  - null             → when status="pass"

- rewrite_query:
  - when action="query_rewrite", a SHORT rewritten query string
  - otherwise null

Guidelines:
- If the evidence is from a clearly different domain than the user query,
  prefer action="query_rewrite" with a concise rewritten query.
- The rewritten query should be short (<= 20 Chinese characters or <= 10 English words),
  and optimized for retrieval (keywords only, no explanation).
"""

    resp = llm.invoke(prompt)

    try:
        result = json.loads(resp.content)
    except Exception:
        # 兜底：至少维持一个可执行的 revise
        result = {
            "status": "revise",
            "reason": "LLM critic returned invalid response, fallback to revise",
            "action": "redo_retriever",
            "rewrite_query": None,
        }

    # 补全缺失字段，避免 KeyError
    if "action" not in result:
        result["action"] = None
    if "rewrite_query" not in result:
        result["rewrite_query"] = None

    return result
