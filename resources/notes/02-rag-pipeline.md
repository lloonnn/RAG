# Complete RAG Pipeline

## Phase 1: Indexing
Load Documents $\rightarrow$ Chunk $\rightarrow$ Embed $\rightarrow$ Store into vector DB  
The embedding vectors in DB represent each chunk of the document.

## Phase 2: Query
Query (User's question) $\rightarrow$ Embed the query $\rightarrow$ Search the embedded query with the vector DB to find similar vectors $\rightarrow$ Retrieve the original text in the documents with the chunked text $\rightarrow$ Generate response from LLM $\rightarrow$ Answer query  

Note: The Embedding of documents (During indexing) and query embedding (During query) must use the same embedding model to ensure same dimension of vectors. 