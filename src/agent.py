from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from src.tools import rag_search

TOOLS = {
    "rag_search": rag_search,
}

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).bind_tools(list(TOOLS.values()))

SYSTEM_PROMPT = """
You are an AI agent.
Use tools when needed.
Return a final answer when ready.
"""


def run_agent(query: str) -> str:
    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    while True:
        response: AIMessage = llm.invoke(messages)
        if response.tool_calls:
            print("TOOL CALLS", response.tool_calls)
            for tc in response.tool_calls:
                tool = TOOLS[tc["name"]]
                result = tool.invoke(tc["args"])

                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    )
                )
            continue

        if response.content:
            print("FINAL ANSERT", response.content)
            return response.content

        raise RuntimeError("Unexpected agent state")
