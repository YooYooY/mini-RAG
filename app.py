import os
from dotenv import load_dotenv
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain.agents import ToolExecutor

load_dotenv()


AMAP_KEY = os.getenv("AMAP_KEY")


# use decoration tool
@tool
def multiply(a: int, b: int) -> int:
    """Return a*b"""
    return a * b


# WeatherTool
class WeatherTool(BaseTool):
    name = "weather_query"
    description = "查询指定城市的天气"

    def _run(self, city: str):
        url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={AMAP_KEY}"
        return requests.get(url).json()


# DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

# convert to openai tool
tools = [multiply, search, WeatherTool()]

llm = ChatOpenAI(model="gpt-3.5-turbo")

agent = llm.bind_tools(tools)

query = "calcute 13 * 21"
result = agent.invoke(query)
print("result:", result)


tool_executor = ToolExecutor(tools)

tool_calls = result.additional_kwargs.get("tool_calls", [])

for tool_call in tool_calls:
    name = tool_call["name"]
    args = tool_call["args"]
    output = tool_executor.invoke(name=name, input=args)
    print("工具执行结果: ", output)
