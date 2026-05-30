from dotenv import load_dotenv
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model # A general llm initializer without importing too many llm providers
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

# llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)

def create_knowledge_base():
    '''Create a vector store from knowledge base'''

    # Split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE, metadata={"source":"langchain_knowledge_base.md"})

    chunks = splitter.split_documents([doc])

    vectorstore = Chroma.from_documents(chunks, embedding=embeddings_model, persist_directory=tempfile.mkdtemp())
    
    return vectorstore

def demo_basic_rag():
    vectorstore = create_knowledge_base()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
    llm1 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,       # Low temperature for precise code/logic tasks
        max_retries=3          # Crucial for Groq to handle occasional rate limits
    )

    # RAG Prompt Template
    prompt_template = '''
Answer the question based only on the following context:

{context}

Question: {question}

Answer:

Make sure to answer in a concise manner, and if you don't know the answer, just say "I don't know.'''

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test the RAG chain
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?"
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

    
if __name__ == "__main__":
    demo_basic_rag()

'''
1) Runnable is an Langchain interface that represent a single execution step (like an LLM, 
a prompt, or a custom Python function) that can be chained

2) | is Langchain expression language (LCEL) to chain up the entire pipeline that is Runnable

3) RunnablePassthrough() is a temparory container that return the same input, meaning that the exactly 
same question string passed will be returned back unchanged so that the question value in the dictionary 
is still the same during chaining

4) StrOutputParser() extract just the text message from LLM response (wihtout metadata, number of input 
token etc, only text output) during chaining
'''