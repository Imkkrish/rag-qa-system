# Mandatory Explanations

## Chunk Size Choice

The project uses `CHUNK_SIZE = 800` characters with `CHUNK_OVERLAP = 120` characters (configurable via the `CHUNK_SIZE` and `CHUNK_OVERLAP` environment variables).

**Why 800 characters?**  
800 characters translates to roughly 150–200 tokens — safely within the 256-token optimal range of `all-MiniLM-L6-v2` (hard limit 512 tokens). This is large enough to contain a complete idea (typically a paragraph) while keeping each chunk focused enough that its embedding accurately reflects a single topic.

- Chunks smaller than ~400 chars often lack enough context for the embedding to be distinctive — a single sentence such as *"It was built in 1990"* is ambiguous without surrounding text.
- Chunks larger than ~1500 chars start to contain multiple unrelated topics, diluting the embedding and reducing cosine similarity precision at retrieval time.

**Why 120 characters of overlap?**  
120 chars ensures that sentences straddling a boundary appear in at least one complete chunk. Without overlap, a two-sentence answer split across two chunks could be missed entirely if the retriever returns only one of them. The overlap is intentionally small (≈15 % of chunk size) to avoid excessive duplication and index bloat.

**Trade-off summary:**

| Setting | Effect |
|---------|--------|
| Smaller chunks | More precise retrieval; risk of missing context |
| Larger chunks | More context per result; diluted embeddings |
| More overlap | Fewer boundary-split misses; larger index |
| Less overlap | Smaller index; higher boundary-split risk |

## Retrieval Failure Case

**Observed failure — empty vector store after ingestion timeout:**  
When the question *"who is krish kumar"* was asked, retrieval returned no contexts. Root cause: the background ingestion job timed out during OCR processing of a scanned PDF, so no chunks were added to the FAISS index. The fix was to (a) use PyPDF2 as the primary extraction path and OCR only as a fallback when fewer than 100 characters are extracted, and (b) surface ingestion status via the `/documents/status/{job_id}` endpoint so the UI can signal incomplete ingestion before a user queries.

**Observed failure — low similarity scores for out-of-domain queries:**  
Queries with no topical overlap to ingested documents produce scores near 0 for all chunks. The system now propagates all retrieved contexts to the UI so the user can see why the LLM answer is weak, and the generator prompt instructs the LLM to say "I don't know" when context is insufficient rather than hallucinating.

**General mitigation strategies:**
- Log every query together with the top-k similarity scores to `metrics.jsonl` for post-hoc analysis.
- Set a minimum score threshold below which the system declines to answer.
- Use hybrid search (BM25 + vector) to improve recall for keyword-heavy queries.

## Tracked Metric

**Primary metric: end-to-end query latency (`latency_ms`)**

Each ingestion and QA operation appends a JSON record to `data/metrics.jsonl`:

```json
{"event": "ingest", "doc_id": "...", "chunks": 42, "latency_ms": 6200}
{"event": "query", "latency_ms": 340}
```

**Why latency?**  
Latency is the most user-visible performance indicator in an interactive QA system. Tracking it per operation (ingest vs. query) pinpoints exactly which stage is the bottleneck:

- Ingestion latency > 10 s indicates OCR or embedding on CPU is the bottleneck → fix: GPU or async worker.
- Query latency > 2 s typically points to the LLM API call → fix: caching, smaller model, or streaming.

**Observed values during development:**

| Operation | Typical latency |
|-----------|----------------|
| PDF ingestion (digital, ~10 pages) | 4–8 s |
| PDF ingestion (scanned, OCR) | 15–30 s |
| Query (embedding + FAISS search) | < 200 ms |
| Query (end-to-end including Gemini) | 1–3 s |

**Additional metrics worth tracking in production:**

| Metric | Purpose |
|--------|---------|
| Retrieval score distribution | Detect domain drift |
| Chunk count per document | Monitor index growth |
| LLM token usage | Cost control |
| Answer relevance (RAGAS) | Quality assurance |

For a full discussion of RAG evaluation metrics, see [RAG_QA.md — Section 7](RAG_QA.md#7️⃣-real-production-problems).
