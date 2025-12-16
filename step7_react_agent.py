from dotenv import load_dotenv

load_dotenv()

import json
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    AIMessage,
)

# ============================================================
# 1. Tools
# ============================================================


@tool
def search(query: str) -> str:
    """Search information about a topic."""
    return "LangChain is a framework for building applications powered by large language models."


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


TOOLS: Dict[str, any] = {
    "search": search,
    "add": add,
}


# ============================================================
# 2. LLM (IMPORTANT: bind tools)
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([search, add])


# ============================================================
# 3. System Prompt (ReACT Style)
# ============================================================

SYSTEM_PROMPT = """
You are a ReACT agent.

Rules:
- Use tools when external information or calculation is required.
- When you decide to use a tool, CALL IT (do not describe it).
- When you have enough information, respond with a final answer.

Do not explain your reasoning to the user.
"""


# ============================================================
# 4. ReACT Agent Loop (FINAL VERSION)
# ============================================================


def react_agent(user_query: str) -> str:
    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]

    while True:
        response: AIMessage = llm.invoke(messages)

        # ====================================================
        # Case 1: LLM wants to call tools (content MAY be empty)
        # ====================================================
        if response.tool_calls:
            print("\n=== Tool Calls Detected ===")

            for tc in response.tool_calls:
                tool_name = tc["name"]
                args = tc["args"]

                print(f"→ Calling tool: {tool_name}")
                print(f"→ Args: {args}")

                tool = TOOLS.get(tool_name)
                if not tool:
                    raise ValueError(f"Unknown tool: {tool_name}")

                result = tool.invoke(args)

                print(f"→ Observation: {result}")

                # IMPORTANT:
                # 1. Append the original AIMessage (with tool_calls)
                # 2. Append ToolMessage as observation
                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    )
                )

            # Continue the loop after tool execution
            continue

        # ====================================================
        # Case 2: Final answer (normal text)
        # ====================================================
        text = response.content.strip()

        if text:
            print("\n=== Final Answer Generated ===")
            return text

        # ====================================================
        # Case 3: Defensive fallback (should rarely happen)
        # ====================================================
        raise RuntimeError("LLM returned neither tool_calls nor content.")


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    question = "What is LangChain? Also calculate 15 + 27."
    final_answer = react_agent(question)

    print("\n=== FINAL OUTPUT ===")
    print(final_answer)
