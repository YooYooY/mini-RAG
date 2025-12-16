from dotenv import load_dotenv

load_dotenv()

from typing import Dict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# ============================================================
# 1. Build RAG (Vector Store)
# ============================================================


def build_retriever():
    loader = TextLoader("data/langchain_intro.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 2})


retriever = build_retriever()


# ============================================================
# 2. Tools (RAG + Vision)
# ============================================================


@tool
def rag_search(query: str) -> str:
    """Search background knowledge using RAG."""
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)


@tool
def classify_image(label: str) -> str:
    """Classify the main object in an image."""
    return f"Image classification result: {label}"


TOOLS: Dict[str, any] = {
    "rag_search": rag_search,
    "classify_image": classify_image,
}


# ============================================================
# 3. Multimodal LLM with Tools
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(list(TOOLS.values()))


# ============================================================
# 4. System Prompt (ENGINEERING VERSION)
# ============================================================

SYSTEM_PROMPT = """
You are a multimodal ReACT agent.

You can:
- See images
- Use tools to retrieve information

Rules:
- Use tools when external knowledge is required.
- When enough information is collected, provide a final answer.
"""


# ============================================================
# 5. ReACT + RAG + Multimodal Agent Loop
# ============================================================


def react_rag_multimodal_agent(
    user_query: str,
    image_url: str | None = None,
) -> str:
    # Initial message
    content: List[dict] = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": user_query},
    ]

    if image_url:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )

    messages = [HumanMessage(content=content)]

    while True:
        response: AIMessage = llm.invoke(messages)

        # ----------------------------
        # Tool Calls (ReACT - Act)
        # ----------------------------
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                args = tc["args"]

                print(f"\n→ Calling tool: {tool_name}")
                print(f"→ Args: {args}")

                tool = TOOLS.get(tool_name)
                result = tool.invoke(args)

                print(f"→ Observation:\n{result}")

                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    )
                )
            continue

        # ----------------------------
        # Final Answer (ReACT - Stop)
        # ----------------------------
        text = response.content.strip()
        if text:
            return text

        raise RuntimeError("LLM returned neither tool_calls nor content.")


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    question = "What is LangChain and what can it be used for?"
    image = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"

    answer = react_rag_multimodal_agent(
        user_query=question,
        image_url=image,
    )

    print("\n=== FINAL ANSWER ===")
    print(answer)
