from fastapi import FastAPI
from src.api.router import api_router
from src.rag.service import RagService

app = FastAPI(title="Mini RAG API")

# 初始化 RagService → 单例，全局共享
app.state.rag = RagService()

# 注册路由
app.include_router(api_router, prefix="/api")
