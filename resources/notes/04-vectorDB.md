The vector database use HNSW algorithm.  

HNSW is a graph-based algorithm used in vector databases to perform high-speed Approximate Nearest Neighbor (ANN) searches between nodes.  

It has 2 main parameter $M$ (Max Connection, the number of nodes connected to a particular node), and $ef$ (Search effort).  

A lower $M$ (8-16) means lesser node connected and give fast search but low accuracy.  
A higher $M$ (32-64) means more node connected and give slow search but high accuract.  

More M = more connections = more memory = better accuracy    
More ef = more search effort = slower = better accuracy  

Hence in production, if the database is very large these two parameters are critical for performance.  

|Use Case| $M$ value|$ef$ value | Priority|
|--------------|------|------|----------|
|Prototype|16|40|Speed|
|Production|16|100|Balanced|
|High accuracy|32|200|Accuracy|

## When and How to scale
When the query takes lots of time (>100 ms) then you likely have index that is too large for memory.  

### 1) Vectical Scale (try this first)
Add more RAM, more CPU
- Pros: Simple, no code changes
- Cons: Has limit
- Best for: Under 5-10 million Vector

### 2) Horizontal Scale
Split data across multiple instances (Shard)
- Pros: Unlimited scale
- Cons: Complex, merge results
- Best for: Over 10 million vectors

## Cost Optimzation Strategies (When Scaling)
### 1) Reduce Vector Dimension
- 1536 -> 512 dims (Set as a parameter in code)
- Savings: 30-60%
- Effort: Low

### 2) Quantization
Reduce the amount of bytes per dimension
- convert float32 -> int8
- Savings: 50-75%
- Effort: Medium

### 3) Batch Queries
Instead of making individual query at every call, collect multiple query into batch, so that there are fewer round trips. 
- Bundle requests
- Savings: 10-30%
- Effort: low

### 4) Caching
- Cache queries (and response) that are repeatitive or similar, so it won't head to the database everytime
- Savings: 10-40%
- Effort: medium

### 5) Right-Size
- Don't over-provision, scale only when neccessary, review your cost monthly to see if scaling is needed
- Savings: 20-50%
- Effort: low

## The bottom line: Which to choose?

|Scale|Best Choice|Why|
|---------|------------|------------|
|<100k|Chroma\ Local|Free, simple|
|100k - 1m|Pinecone serverless|Low cost, zero ops|
|1m-10m|pgvector managed|Cost-effective|
|10m+|pgvector self-hosted|Significant savings| 
