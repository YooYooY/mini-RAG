from dotenv import load_dotenv
import requests

load_dotenv()

import os

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage


# ----------------------
# Tool 1: Math (add)
# ----------------------
@tool
def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city using AMap API.
    City should be a Chinese city name or adcode.
    """
    api_key = os.getenv("AMAP_KEY")
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "city": city,
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()

    if data.get("status") != "1":
        return f"Weather API Error: {data}"

    info = data["lives"][0]

    return (
        f"{info['city']} weather: {info['weather']}, "
        f"{info['temperature']}°C, humidity {info['humidity']}%"
    )


# ----------------------
# Tool 3: Search
# ----------------------
search_tool = DuckDuckGoSearchRun()


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [add, get_weather]
    agent = llm.bind_tools(tools)

    user_query = (
        # "Please search what LangChain is, "
        "then calculate 15+ 27, "
        "and also tell me today's weather in Beijing."
    )

    first_response = agent.invoke(user_query)

    print("\n=== Planning (LLM decides tools) ===")
    print(first_response)

    tool_messages = []

    for tool_call in first_response.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        call_id = tool_call["id"]

        print(f"\n→ Executing tool: {name}")
        print(f"→ Arguments: {args}")

        if name == "add":
            result = add.invoke(args)
        elif name == "get_weather":
            result = get_weather.invoke("110101")
        elif name == search_tool.name:
            result = search_tool.invoke(args)
        else:
            raise ValueError(f"Unknow tool: {name}")

        print(f"-> Result: {result}")

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

        # -------- Step 3: final response --------
    final_response = llm.invoke(
        [
            {"role": "user", "content": user_query},
            first_response,
            *tool_messages,
        ]
    )

    print("\n=== Final Answer ===")
    print(final_response.content)


if __name__ == "__main__":
    main()
