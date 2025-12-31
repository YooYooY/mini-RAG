from __future__ import annotations

SUMMARIZE_KEYWORDS_ZH = ["总结", "概括", "归纳", "精简一下", "提炼一下"]
SUMMARIZE_KEYWORDS_EN = ["summarize", "summary", "tl;dr"]

COMPARE_KEYWORDS_ZH = ["对比", "比较", "区别", "差异", "不同点", "相同点"]
COMPARE_KEYWORDS_EN = ["compare", "difference", "similarities", "vs "]

REWRITE_KEYWORDS_ZH = ["改写", "润色", "优化这段话", "换一种说法", "翻译成"]
REWRITE_KEYWORDS_EN = ["rewrite", "paraphrase", "polish", "rephrase", "translate"]


def detect_intent(query: str) -> dict:
    """
    AskMyDocs Planner 级 intent 检测（领域无关）

    - 不绑定“是什么文档”
    - 只管“用户想对这堆知识做什么”
    - 返回尽量简单，让后端容易用
    """

    q = query.strip().lower()

    # Summarize
    if any(k in query for k in SUMMARIZE_KEYWORDS_ZH) or any(
        k in q for k in SUMMARIZE_KEYWORDS_EN
    ):
        return {"intent": "summarize"}

    # Compare
    if any(k in query for k in COMPARE_KEYWORDS_ZH) or any(
        k in q for k in COMPARE_KEYWORDS_EN
    ):
        return {"intent": "compare"}

    # Rewrite / Translate
    if any(k in query for k in REWRITE_KEYWORDS_ZH) or any(
        k in q for k in REWRITE_KEYWORDS_EN
    ):
        return {"intent": "rewrite"}

    # 默认：问答型
    return {"intent": "qa"}
