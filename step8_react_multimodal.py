from dotenv import load_dotenv

load_dotenv()

from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    AIMessage,
)

# ============================================================
# 1. Tool
# ============================================================


@tool
def classify_image(label: str) -> str:
    """Return classification result of an image."""
    return f"The image is classified as: {label}"


TOOLS: Dict[str, any] = {
    "classify_image": classify_image,
}


# ============================================================
# 2. Multimodal LLM (bind tools)
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([classify_image])


# ============================================================
# 3. System Prompt (NO text-ReACT)
# ============================================================

SYSTEM_PROMPT = """
You are a multimodal ReACT agent.

Rules:
- You can see images.
- Use tools when they help.
- When you are done, provide a final answer.
"""


# ============================================================
# 4. ReACT Multimodal Agent Loop
# ============================================================


def react_multimodal_agent(image_url: str, user_query: str) -> str:
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "text", "text": user_query},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        )
    ]

    while True:
        response: AIMessage = llm.invoke(messages)

        # ----------------------------
        # Case 1: tool calls
        # ----------------------------
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                args = tc["args"]

                print(f"\n→ Calling tool: {tool_name}")
                print(f"→ Args: {args}")

                tool = TOOLS.get(tool_name)
                result = tool.invoke(args)

                print(f"→ Observation: {result}")

                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    )
                )
            continue

        # ----------------------------
        # Case 2: final answer
        # ----------------------------
        text = response.content.strip()
        if text:
            return text

        raise RuntimeError("LLM returned neither tool_calls nor content.")


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    image_url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
    query = "Look at the image and tell me what it is."

    answer = react_multimodal_agent(image_url, query)

    print("\n=== FINAL ANSWER ===")
    print(answer)
