from langchain_core.tools import tool
from src.rag import build_retriever

retriever = build_retriever()


@tool
def rag_search(query: str) -> str:
    """Search background knowledge via RAG."""
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)
