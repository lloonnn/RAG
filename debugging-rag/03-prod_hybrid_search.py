'''
To solve retrieval noise, use hybrid search
'''

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Document with both semantic content AND specific indentifiers
documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports "
                    " gigabit speeds and advanced QoS features.",
        metadata={"type":"products"}
    ),

    Document(
        page_content="For network connectivity issues, first check the "
                    "ethernet cable and router status lights.",
        metadata={"type":"troubleshooting"}
    ),

    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
                    "rejected the connection. Check firewall settings. ",
        metadata={"type":"error"}
    ),

    Document(
        page_content="The authentication process requires valid credentials. "
                    "Use OAuth2 for secure API access. ",
        metadata={"type":"auth"}
    ),

    Document(
        page_content="Router configuration guide: Access the admin panel "
                    "at 192.168.1.1 to modify settings. ",
        metadata={"type":"config"}
    ),

    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
                    "alt text and sufficient color contrast. ",
        metadata={"type":"compliance"}
    ),
]

print(f"Loaded {len(documents)} documents")

# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="hybrid_test"
)

# Create vector retriever
vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)

print("Vector retriever ready")

# BM25 works on the raw test
bm25_retriever = BM25Retriever.from_documents(
    documents, 
    k=3
)

print("BM25 retriever ready")

# Combine with EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5] # equal weight to both
)

print("Hybrid retriever ready")

def test_query(query, name, retriever):
    '''Test a query and show results'''
    results = retriever.invoke(query)
    print(f"\\n{name} - Query: \"{query}\"")
    for i, doc in enumerate(results[:3]):
        preview= doc.page_content[:80] + "..."
        print(f"    {i+1}. {preview}")
    
    return results

# Test queries designed to challenge to challenge vector search
test_queries = [
    "SKU-7742 specifications", # Exact product code
    "E_CONN_REFUSED error", # Error code
    "How do I authenticate?", # Semantic question
    "WCAG compliance", # Accronym
    "router configuration" # General semantic
]

for query in test_queries:
    print("=" *68)

    # Vector only
    vector_results = test_query(query, 'VECTOR', vector_retriever)

    # BM25 only
    bm25_results = test_query(query, 'BM25', bm25_retriever)

    # Hybrid
    hybrid_results = test_query(query, 'HYBIRD', ensemble_retriever)

