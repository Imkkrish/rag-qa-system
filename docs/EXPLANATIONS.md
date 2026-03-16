# Mandatory Explanations

## FAISS Index Type

This system uses **`IndexFlatIP`** — FAISS's *Flat Inner Product* index.

### What it means

| Property | Detail |
|---|---|
| **Index family** | Flat (brute-force, no compression or clustering) |
| **Similarity metric** | Inner Product (dot product) |
| **Search type** | Exact nearest-neighbour — every stored vector is compared on every query |
| **Vector dimension** | 384 (output size of `all-MiniLM-L6-v2`) |

### Why Inner Product equals Cosine Similarity here

Before vectors are added to the index (and before queries are encoded), they are **L2-normalised** by `_normalize()`:

```python
def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms
```

When all vectors are unit-length, the inner product between two vectors equals their cosine similarity:

```
dot(u, v) = |u| × |v| × cos(θ) = 1 × 1 × cos(θ) = cos(θ)
```

Higher scores (closer to 1.0) indicate greater semantic similarity.

### Why `IndexFlatIP` was chosen

- **Exact recall** — no approximation error; every result is the true nearest neighbour.
- **Simplicity** — no training step, no tuning of cluster counts or graph parameters.
- **Appropriate scale** — for small-to-medium document collections (up to ~100k vectors at 384 dimensions) the brute-force scan is fast enough and fits comfortably in memory.
- **Persistence** — the index is serialised to `data/index.faiss` and optionally synced to HuggingFace Hub, so it survives restarts without reingestion.

### Comparison with other FAISS index types

| Index | Search | Recall | Notes |
|---|---|---|---|
| **`IndexFlatIP`** *(used here)* | Exact | 100 % | Best for ~100k vectors or fewer |
| `IndexIVFFlat` | Approximate (IVF) | ~95–99 % | Needs training; faster for large stores |
| `IndexHNSWFlat` | Approximate (graph) | ~99 % | Fast queries, high memory overhead |
| `IndexFlatL2` | Exact (L2 distance) | 100 % | Ranks by Euclidean distance; not normalised |

## Chunk Size Choice
I chose a chunk size of 1000 characters with 200 character overlap. This balances context preservation and retrieval granularity. Smaller chunks (e.g., 500 chars) risked losing semantic context across sentences, while larger chunks (e.g., 2000 chars) could dilute relevance in searches. The 200 char overlap ensures continuity between chunks, reducing the chance of splitting related information. This was tested with sample documents to ensure chunks contain coherent ideas without being too verbose for embedding efficiency.

## Retrieval Failure Case
One observed retrieval failure was with the question "who is krish kumar" initially returning no contexts because the document ingestion failed due to OCR processing taking too long in the background job, causing the task to not complete. The vector store was empty, leading to zero results. This was fixed by implementing local fallbacks in the UI and ensuring ingestion completes before querying. Another potential failure is low similarity scores for out-of-domain questions, but the system handles this by falling back to excerpts.

## Tracked Metric
I tracked query latency as a key metric, logging it in `metrics.jsonl` for each ingestion and QA operation. This helps monitor system performance and identify bottlenecks, such as slow OCR or embedding generation. For example, ingestion latency averaged 5-10 seconds for PDFs, while QA was under 2 seconds excluding LLM calls. This metric is crucial for optimizing user experience in a real-time system.
