'''
Vector Database with Chroma
'''

from pathlib import Path
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.embeddings import JinaEmbeddings
# from langchain_cohere import CohereEmbeddings
# from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()
DB_PATH = "./chroma_db"
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Some other free embeddings api from google ai (yet to verify)
'''
cohere_embeddings = CohereEmbeddings(model="embed-english-v3.0")
voyage_embeddings = VoyageAIEmbedding(model="yoyage-3")
jina_embeddings = JinaEmbeddings(model_name="jina-embeddings-v3")
'''

# Original Source for indexing
SAMPLE_DOCS = [
    Document(
        page_content="Langchain is a framework for developing applications powered by language models.",
        metadata={"source":"langchain_docs", "topic":"overview"}
    ),

    Document(
        page_content="LangGraph is a library building stateful, multi-actor applications with LLMs.",
        metadata={"source":"langgraph_docs", "topic":"overview"}
    ),

    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source":"rag_guide", "topic":"architecture"}
    ),

    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source":"embeddings_guide", "topic":"fundementals"}
    ),

    Document(
        page_content="Chroma is an open-source embedding database for AI application",
        metadata={"source":"chroma_docs", "topic":"database"}
    ),

    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source":"faiss_docs", "topic":"database"}
    ),

    Document(
        page_content="Pinecone is managed vector database service for production workloads.",
        metadata={"source":"pinecone_docs", "topic":"database"}
    )
]

def reset_db():
    '''Helper function to delete existing chroma_db/ to prevent adding the same source in the db for every calls'''
    if Path(DB_PATH).exists():
        shutil.rmtree(DB_PATH)

def chroma_basics():
    k = 2

    # create vector store from documents
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=DB_PATH
    )

    print(f"Vector store created {vectorstore._collection.count()} documents and persisted")

    # perform similarity search
    query = "What is LangChain?"
    results = vectorstore.similarity_search(query, k=k)

    print(f"Top {k} results for query '{query}':")

    for i, doc in enumerate(results):
        print(
            f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
        )
    print()

def similarity_search_with_scores():
    # create vector store from documents
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=DB_PATH
    )

    # Perform similarity search with scores
    query = "Explain vector stores."
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    print(f"Top 3 results with scores for query '{query}':")
    for i, (doc, score) in enumerate(results_with_scores):
        final_score = 1 / (1 + score)  # Convert distance to similarity
        print(
            f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
        )
    # The score here refer to distance score not similarity score (score closer to 0 means more relevant), 
    # and not similarity score, since the closer to 1 means more similar
    print()

# To narrow down the search
def metadata_filtering():
    # create vector store from documents
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=DB_PATH
    )

    query = "What databases are available?"

    # Wihtout metadata filtering
    results = vectorstore.similarity_search(query, k=5)
    print(f"Results without metadata filtering for query '{query}':")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    print()

    #With metadata filtering
    filter_criteria = {'topic':"database"}
    filter_result = vectorstore.similarity_search(query, k=5, filter=filter_criteria)
    print(f"Results with metadata filtering for query '{query}':")
    for i, doc in enumerate(filter_result):
        print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")
        
    print()

def as_retriever():
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=DB_PATH
    )
     
     # Basic retrieval usage
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k":3}
     )

     # Use retrieval to get relevant documents
    query = "How do i build AI applications?"
    docs = retriever.invoke(query)

    print("Retriever results:")
    for i, doc in enumerate(docs):
        print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    print()

    # Fetch 5 docs and return 3 diverse
    mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3, "fetch_k":5})

    query = "Vector databases and embeddings"
    mmr_docs = mmr_retriever.invoke(query)
    print("MMR Retriever results:")
    for i, doc in enumerate(mmr_docs):
        print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

    print()

# Setup chroma + retrieval 
def exercise_vector_store_setup():
    '''
    EXERCISE: Create a complete vector store setup that:
    1. Takes a list of text strings
    2. Splits them into chunks
    3. Stores in Chroma
    4. Returns a configured retriever

    Test with sample documents
    '''
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    def create_retriever(document: list[str], chunk_size: int = 500, chunk_overlap:int = 50, k: int = 5):
        doc = [Document(page_content=text) for text in document]
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_doc = splitter.split_documents(doc)

        # Create a vectore store 
        vectorstore = Chroma.from_documents(documents=split_doc, embedding=embeddings_model, persist_directory=DB_PATH)

        # Return retriever
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":k})
    
    # test the function
    sample_texts = [
        "Python is a versatile programming language used in web development, "
        "data science machine learning, and automation. It has a simple syntax "
        "that makes it easy to learn and read. ",
        
        "JavaScript is the language of web. It runs in browsers and on "
        "servers with Node.js. Modern frameworks like React and Vue make "
        "building web applications efficient. ",

        "Rust is a systems programming language focused on safety and "
        "performance. It prevents common bugs like null pointer dereferences "
        "and data races at compile time. "
    ]

    retriever = create_retriever(sample_texts, chunk_size=200, chunk_overlap=20, k=3)

    print("Testing retriever:\n")
    queries = [
        "What's good for web development?",
        "Which language is safest?"
    ]

    for query in queries:
        print(f"Query: {query}")
        result = retriever.invoke(query)
        for doc in result:
            print(f"    - {doc.page_content[:60]}...")
        print()
            

if __name__ == "__main__":
    reset_db()

    # ===========================
    # Direct/ manual query
    # ============================
    #chroma_basics()
    #similarity_search_with_scores()
    #metadata_filtering()

    # ===========================
    # Retrieval query
    # ===========================
    #as_retriever()
    
    # ===========================
    # Sample testing flow of retrieval
    # ===========================
    exercise_vector_store_setup()


'''
1) .NamedTemporaryFile() create tempfile, .TemporaryDirectory() create tempdirectory
2) .from_documents() store data to Chroma database
3) k=3 return top 3 most similar documents when you query a vector database
Too small -> Might miss relevant info
Too large -> includes irrelevant stuff, wastes tokens

'''
