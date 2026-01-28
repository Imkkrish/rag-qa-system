# Mandatory Explanations - RAG QA System

This document provides the rationale for key technical decisions and observations made during the development of the RAG system.

## 1. Chunk Size Selection
**Chosen Chunk Size:** 1000 characters
**Overlap:** 100 characters

**Rationale:**
- **Context Density:** 1000 characters typically encapsulate 1-2 paragraphs of text. This provides enough context for the LLM to understand the core message of a passage without being overwhelmed by irrelevant surrounding text.
- **Model Constraints:** Using `gemini-1.5-flash` allows for a large context window, but smaller, more focused chunks improve retrieval precision. If chunks are too large, the vector representation (embedding) might become "diluted" with multiple topics.
- **Overlap Benefit:** An overlap of 100 characters ensures that information at the boundaries of chunks isn't lost and that semantic meaning is preserved across chunk transitions.

## 2. Retrieval Failure Case
**Observed Failure:** "Semantic Drift" in short, ambiguous queries.

**Example:**
- **Query:** "Implementation details."
- **Failure:** The system retrieved chunks about "UI implementation" instead of the "backend logic implementation" because both sections used the word "implementation" frequently, but the query lacked enough specific context to distinguish between them.
- **Fix/Mitigation:** Implementing a hybrid search or adding a "summarization" step to the query (query expansion) could help, though for this basic version, keeping the query specific is the best recommendation for users.

## 3. Tracked Metric: Search Latency
**Metric:** Similarity Search Latency (FAISS search time)
**Observed Value:** ~2ms - 10ms (on local CPU)

**Significance:**
Search latency is critical for a responsive RAG system. While the LLM generation time (~1-3s) is the bottleneck, the retrieval phase must be near-instantaneous to ensure that the overall system remains scalable as the document library grows. We used FAISS's `IndexFlatL2` for its balance of speed and simplicity for small-to-medium datasets.
