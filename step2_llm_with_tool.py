from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = llm.bind_tools([add])

    query = "You MUST call the add tool to calculate 12 + 30."

    response = agent.invoke(query)

    print("\n=== 模型回复对象 ===")
    print(response)

    print("\n=== 原始模型返回（包含是否调用工具） ===")
    print(response.additional_kwargs)


if __name__ == "__main__":
    main()
