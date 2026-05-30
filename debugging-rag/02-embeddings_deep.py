'''
To solve embedding mismatch, use similarity
'''

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def basic_embeddings():
    
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Norm of the vector: {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?"
    ]

    batch_embedding = embeddings.embed_documents(text) # documents is a list of text
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimension: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")
        print()

def similarity_search():
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets"
    ]

    query = "What programming language exist?"

    # embed documents and query
    doc_vec = embeddings.embed_documents(docs)
    query_vec = embeddings.embed_query(query)

    # cos similarity
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    similarities = [cosine_similarity(query_vec, d_vec) for  d_vec in doc_vec]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"    {score:.4f} : {doc}")


if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()