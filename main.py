from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core import __version__ as core_version
from langgraph.version import __version__ as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # response = llm.invoke("Say, 'setup complete!' in one word")
    # print(f"Response from ChatOpenAI: {response}")

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     max_retries=2
    # )

    # response = llm.invoke("Say, 'setup complete!' in one word")
    # print(f"Response from ChatGoogleGenerativeAI: {response}")

    llm_groq = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,       # Low temperature for precise code/logic tasks
        max_retries=3          # Crucial for Groq to handle occasional rate limits
    )

    response_groq = llm_groq.invoke("Say, 'setup complete!' in one word")
    print(f"Response from ChatGroq: {response_groq}")

    llm_openrouter = ChatOpenAI(
        base_url="https://openrouter.ai",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model="meta-llama/llama-3.1-8b-instruct:free", # Explicitly request the free tier model variant
        temperature=0
    )

    response_openrouter = llm_groq.invoke("Say, 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response_openrouter}")


    print("Setup complete!")


if __name__ == "__main__":
    main()

