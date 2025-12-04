from fastapi import Request, HTTPException
from src.rag.service import RagService


def get_rag_service(request: Request) -> RagService:
    rag = request.app.state.rag
    if rag is None:
        raise HTTPException(500, "RAG 服务尚未初始化")
    return rag
