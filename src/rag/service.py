from typing import List
from pathlib import Path
from threading import Lock

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import (
    PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EMBEDDING_MODEL,
    CHAT_MODEL,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from src.rag.pdf_parser import PDFParser


class RagService:
    """
    RAG 主业务服务：
    - 文档索引
    - 文本切分
    - Embedding
    - 向量库加载/更新
    - 文档检索（retriever）
    - QA 推理
    """

    def __init__(self):
        self.persist_dir = PERSIST_DIR
        self._lock = Lock()

        # 初始化 OCR / 文档解析器
        # self.parser = PDFParser()

        # 初始化 Embedding
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # 初始化向量库（不存在则自动创建）
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir), embedding_function=self.embeddings
        )

        # 初始化文本切分器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        # 初始化 retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        # 初始化 LLM
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    # -----------------------------
    # 1. 文档索引
    # -----------------------------
    def index_file(self, file_path: Path) -> int:
        """
        统一索引入口：
        - 自动判断 txt / pdf
        - 自动 OCR fallback
        - 切分 chunk
        - 入向量库
        """

        suffix = file_path.suffix.lower()

        # 文本文件
        if suffix == ".txt":
            docs = self._load_txt(file_path)
        # PDF 文件（文字版或扫描版）
        # elif suffix == ".pdf":
        #     docs = self.parser.parse(file_path)
        # else:
        #     raise ValueError(f"不支持的文件类型: {suffix}")

        # 切分 chunk
        splits = self.splitter.split_documents(docs)

        if not splits:
            raise ValueError("文档解析后没有有效文本可用于索引。")

        # 写入向量库加锁（避免并发冲突）
        with self._lock:
            self.vectorstore.add_documents(splits)
            self.vectorstore.persist()

        return len(splits)

    def _load_txt(self, path: Path) -> List[Document]:
        """解析 txt 文件为 Document 列表"""
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text)]

    # -----------------------------
    # 2. 文档检索
    # -----------------------------
    def search(self, query: str) -> List[Document]:
        """返回相关文本 Document 列表"""
        return self.retriever.get_relevant_documents(query)

    # -----------------------------
    # 3. QA 推理
    # -----------------------------
    def ask(self, query: str) -> str:
        """使用 LCEL 完成检索增强问答"""

        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的知识问答助手。请严格基于提供的上下文回答用户问题，
不能添加不存在的内容。如果答案无法从上下文中找到，请明确告知“当前知识库中未找到相关信息”。

[上下文]
{context}

[问题]
{question}

请给出准确、简洁的回答。
"""
        )

        # LCEL Pipeline： retriever → prompt → LLM → parser
        chain = (
            {
                "context": self.retriever
                | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": lambda x: x,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # 执行 LCEL
        answer = chain.invoke(query)
        return answer
