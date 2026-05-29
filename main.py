import os

from dotenv import load_dotenv
from importlib.metadata import version

from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()
core_version = version("langchain-core")
lg_version = version("langgraph")

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # llm = ChatOpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.environ.get("OPENROUTER_API_KEY"),
    #     #model="deepseek/deepseek-chat-v3-0324:free",
    #     #model="meta-llama/llama-3.3-8b-instruct:free",
    #     model="qwen/qwen3-32b:free",
    #     temperature=0
    # )

    # response = llm.invoke("Say 'setup complete!' in one word")
    # print(response.content)

    llm_GenAI = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2
    )

    response_GenAI = llm_GenAI.invoke("Say, 'setup complete!' in one word")
    print(f"Response from ChatGoogleGenerativeAI: {response_GenAI}")

    llm_groq = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,       # Low temperature for precise code/logic tasks
        max_retries=3          # Crucial for Groq to handle occasional rate limits
    )

    response_groq = llm_groq.invoke("Say, 'setup complete!' in one word")
    print(f"Response from ChatGroq: {response_groq}")

    print("Setup complete!")


if __name__ == "__main__":
    main()

