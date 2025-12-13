from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def main():
    llm = ChatOpenAI(temperature=0.2)

    response = llm.invoke("Hello! Please introduce yourself briefly.")

    print("LLM output: ")
    print(response)


if __name__ == "__main__":
    main()
