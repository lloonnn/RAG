import os
import tempfile
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)

from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    # Create a temporory text file for documentation 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file: 
        temp_file.write(b"Hello, this is a sample text file.\nThis is used to demonstrate the TextLoader.") #.write()expect binary from NamedTemporaryFile
        temp_file_path = temp_file.name

    try:
        # load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        
        print(f"Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: {documents[0].metadata}")

        # # Print the loaded documents
        # for doc in documents:
        #     print("Document Content:")
        #     print(doc)
        #     print(doc.page_content)

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load();

    print("=" * 35)
    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")
        print()
    print("=" * 35)
    

if __name__ == "__main__":
    # load_text_file()
    pdf_loader("./resources/docs/langchain_demo.pdf")

