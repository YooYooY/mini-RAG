from langchain_core.tools import tool

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

load_dotenv()


# 1️⃣ Define a simple tool
@tool
def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b


def main():
    # Creat LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Bind tools to LLM
    agent = llm.bind_tools([add])

    # User input
    user_query = "You MUST call the add tool to calculate 12 + 30."

    # First LLM call
    first_response = agent.invoke(user_query)

    print("\n=== First LLM Response ===")
    print(first_response)

    # Check if the model decided to call a tool
    tool_calls = first_response.tool_calls

    if not tool_calls:
        print("\n ❌ No tool was called by the model.")
        return

    # Execute the tool(s)
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\n→ Executing tool: {tool_name}")
        print(f"→ Tool arguments: {tool_args}")

        if tool_name == "add":
            result = add.invoke(tool_args)
        else:
            raise ValueError(f"Unknow tool: {tool_name}")

        print(f"-> Tool result: {result}")

        # Wrap tool result into ToolMessage
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    # Second LLM call (final answer)
    final_response = llm.invoke(
        [{"role": "user", "content": user_query}, first_response, *tool_messages]
    )

    print("\n=== Final LLM Response ===")
    print(final_response.content)


if __name__ == "__main__":
    main()
