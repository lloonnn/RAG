# Chunking
## Document Processing Pipeline- RAG Indexing Pipieline:
### Document Loaders:
- Extract text from files
- Handle different formats

### Text splitters
- Chunk into 500-1000 char pieces
- Preserve sentence boundaries
- Add overlap (100-200 chars)

### Embedding Generation
- Convert each chunk to vector
- Use OpenAI/ Cohere API

### Vector Storage
- Store in Chroma/ Pinecone
- Index for fast search

### Ready for queries!

Same input but different chunking will affect the retrieval and end up different results.
For instance, if the chucks poorly split text and break the context, the retrieval will retireve incomplete context for the embeddings that send to llm.  

## Types of chunking (From easiest to advanced)
### 1) Fixed-size chunking
How it works:  
- Cutting at exact intervals (e.g. every 500 chars)  

When to use:
- When you are just implementing something really simple 
- You just want fast processing
- Predictable sizes

Cons:
- Destroy meaning (Because it could truncate at critical cut which the text could lose its meaning or meaningless if some words are splitted. e.g The quick brown fox jum | ps over the la | zy dog.)
- Inaccurate retrieval
- Frustrating user experience

**DO NOT USE IN PRODUCTION**

### 2) Recursive Chunking
The reliable defualt  
It tries to split at natural boundaris in order of preference.   
For instances, first split is based on paragrpah  

If too big:  
Within each paragraph, split it in new lines  

If still too big:  
Within each line, split it in sentences (full stop)  

If still too big:  
Within each sentence, split it in clauses  

If still too big:  
Within each clauses, split it in words  

If still too big:  
Within each word, split it in characters  

### 3) Semantic chunking
A premium structure because it splits everything based on meaning not based on text structure. Similar topics have similiar embedding.

- First, embed each sentence
- Second, compare adjacent embeddings
- Third, split when similarity drops

**Use this when high quality matters** e.g. legal documents, technical manuals and knowledge bases. When accuracy more than speed.

### 4)Late chunking
Traditional chunking:
- First chunking with fixed chunking
- Second, embed each chunk into embeddings, where each embedding is independent to each other

Issue for traditional chunking:  
- Chunk 5 has no idea what chunk 1-4 contains and so on  

Late chunking embed the full document hence, each chunk understand the context of other chunks. Within the emded document, token boundaries are decided to perform chunking. 

Cons:  
- Complex implemantation  

Use when doing cutting edge RAG system  

---

### Quick Reference Table
|Content Type|Strategy|Chunk Size|
|-------------|-------------|------------|
|General docs|Recursive|500 - 1000|
|Technical|Semantic|Auto|
|Code|Code Splitter|Function|
|Markdown|MD Splitter|Headers|




 
