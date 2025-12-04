from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from src.core.dependencies import get_rag_service
from src.rag.service import RagService
from src.core.config import UPLOAD_DIR

api_router = APIRouter()


@api_router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    rag: RagService = Depends(get_rag_service),
):
    save_path = UPLOAD_DIR / file.filename

    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        chunks = rag.index_file(save_path)
    except Exception as e:
        raise HTTPException(400, f"文件索引失败：{e}")

    return {"indexed_chunks": chunks}


@api_router.post("/ask")
async def ask_question(
    query: str,
    rag: RagService = Depends(get_rag_service),
):
    answer = rag.ask(query)
    return {"answer": answer}
