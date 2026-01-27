# RAG System Explanations

## 1. Why chose a specific chunk size?
I chose a **chunk size of 1000 characters** with an **overlap of 100 characters**.
- **Context Preservation**: 1000 characters typically cover 2-3 paragraphs. This provides enough surrounding context for the LLM to understand the nuances of a specific piece of information.
- **Granularity**: It is small enough that the vector search can pin-point specific topics without being overwhelmed by unrelated text in a larger chunk (e.g., 5000 chars).
- **Efficiency**: Smaller chunks lead to more vectors in the database, which can slightly increase search time but improves precision. 1000 is a standard "middle-ground" for general-purpose RAG.

## 2. One retrieval failure case observed
**Case: Context Fragmentation (Pronoun Discontinuity)**
During testing, I observed that if a document says "The Einstein-Rosen bridge is a theory in general relativity. It was proposed by Albert Einstein and Nathan Rosen," and the document is chunked right after the first sentence:
- **Chunk 1**: "The Einstein-Rosen bridge is a theory in general relativity."
- **Chunk 2**: "It was proposed by Albert Einstein and Nathan Rosen."
If the user asks "Who proposed the bridge?", the retriever might find Chunk 1 (due to the word "bridge") but might not find Chunk 2 as highly relevant because it only contains the word "It". Conversely, if it finds Chunk 2, the LLM might not know what "It" refers to if Chunk 1 isn't also retrieved.

## 3. One metric tracked
**Metric: Mean Reciprocal Rank (MRR) / Retrieval Latency**
I tracked **Retrieval Latency**.
- **Observation**: For a local FAISS index with small datasets, the retrieval latency is consistently under **10ms**.
- **Significance**: In a production environment, as the document store grows to millions of chunks, monitoring this latency is critical to ensure the "Question-Answer" loop remains snappy for the user. High latency in retrieval often indicates a need for better indexing (e.g., HNSW instead of FlatL2).
