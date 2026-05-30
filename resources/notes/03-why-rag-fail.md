# Why 90% RAG Projects Fail especially during Production?
## 1) Bad Chunking
Bad chunking could caused retrieve partial context and mostly get wrong answer

## 2) Embedding Mismatch
When user's query use different words with the words found in the documents, semantic search fails. 

## 3) Retrieval Noise
Use hybrid search to solve the issue  
- Traditional vector search is looking for semantic

### When vector search fails (Suppose user query or documents has the following characteristic)
1) Product Codes
product codes like SKU-7742x has no semantic meaning

2) Acronyms
Model does not know abbreviation

3) Exact names
If query is about John Smith accounting, the vector search might find something about accounting and people name John, but missed something specific mentioning about John Smith, beacuse the searching is looking for semantic, not the exact name

4) Error Codes or Typo
The embedding model will treat it as just characters

Solution: BM25 search (Looking for exact words and does not understand meaning) + vector search

||Vector Search|BM25 Search|
|---------|--------------------|--------------------|
|Good At|Semantic Similarity|Exact Matches|
||Synonyms|Rare terms|
||Natural Questions|COdes & IDs|
|Bad At|What BM25 Search Good At|What Vector Search Good At|


## 4) Context Overflow

## 5) Hallucination