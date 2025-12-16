from dotenv import load_dotenv

load_dotenv()

import re
from typing import List

from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# ============================================================
# 1. Business-level schema
# ============================================================


class Answer(BaseModel):
    answer: str
    sources: List[str]
    used_rag: bool
    error: str | None = None  # ← 新增：失败原因（可选）


# ============================================================
# 2. Model-level schema
# ============================================================


class RawRAGAnswer(BaseModel):
    definition: str
    uses: List[str]


# ============================================================
# 3. Adapter
# ============================================================


def adapt_raw_to_answer(
    raw: RawRAGAnswer,
    *,
    used_rag: bool,
    sources: List[str],
) -> Answer:
    text = raw.definition + "\n\nUses:\n" + "\n".join(f"- {u}" for u in raw.uses)

    return Answer(answer=text, sources=sources, used_rag=used_rag, error=None)


# ============================================================
# 4. JSON utilities
# ============================================================


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ============================================================
# 5. RAG retriever
# ============================================================


def build_retriever():
    loader = TextLoader("data/langchain_intro.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 2})


# ============================================================
# 6. Core logic: generate raw semantic JSON
# ============================================================


def generate_raw_answer(llm, question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Return a JSON with fields: summary and examples",
                # "Return a JSON object with EXACTLY these fields:\n"
                # "- definition: string\n"
                # "- uses: array of strings\n"
                # "Do NOT include markdown fences.",
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    chain = prompt | llm
    return chain.invoke({"question": question, "context": context}).content


# ============================================================
# 7. Repair prompt (when parsing fails)
# ============================================================


def repair_json(llm, raw_output: str, error_msg: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "The previous output was invalid JSON for the required schema.\n"
                "Fix it and return ONLY valid JSON.\n"
                "Required fields:\n"
                "- definition: string\n"
                "- uses: array of strings\n"
                "Do NOT add extra fields.",
            ),
            ("human", "Previous output:\n{raw}\n\nError:\n{error}"),
        ]
    )
    print("error_msg=>", error_msg)
    chain = prompt | llm
    return chain.invoke({"raw": raw_output, "error": error_msg}).content


# ============================================================
# 8. High-level API: answer with retry / repair / fallback
# ============================================================


def answer_with_rag_safe(question: str, docs) -> Answer:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    context_text = "\n\n".join(d.page_content for d in docs)

    # -------- First attempt --------
    raw_text = generate_raw_answer(llm, question, context_text)

    try:
        # cleaned = extract_json(raw_text)
        print("raw_text=>", raw_text)
        raw = RawRAGAnswer.model_validate_json(raw_text)
        return adapt_raw_to_answer(raw, used_rag=True, sources=["langchain_intro.txt"])

    except ValidationError as e:
        # -------- Repair attempt --------
        print("-------- Repair attempt --------")
        repaired_text = repair_json(llm, raw_text, str(e))

        try:
            cleaned = extract_json(repaired_text)
            raw = RawRAGAnswer.model_validate_json(cleaned)
            return adapt_raw_to_answer(
                raw, used_rag=True, sources=["langchain_intro.txt"]
            )

        except ValidationError:
            # -------- Fallback --------
            return Answer(
                answer="Failed to generate a structured answer.",
                sources=["langchain_intro.txt"],
                used_rag=True,
                error="Structured output validation failed after retry.",
            )


# ============================================================
# 9. Main
# ============================================================


def main():
    retriever = build_retriever()

    question = "What is LangChain and what can it be used for?"
    docs = retriever.invoke(question)

    result = answer_with_rag_safe(question, docs)

    print("\n=== Final Answer (Safe) ===")
    print(result.model_dump())


if __name__ == "__main__":
    main()
