# RAG Interview Questions & Answers

A comprehensive reference covering all aspects of Retrieval-Augmented Generation (RAG) — from fundamentals to production-level system design.

---

## 1️⃣ Basic RAG Questions

### What is Retrieval Augmented Generation (RAG)?

RAG is an architecture that enhances a Large Language Model (LLM) by supplying it with relevant documents retrieved at query time rather than relying solely on knowledge encoded during pre-training. The pipeline has two stages:

1. **Retrieve** – given a user question, search a document store for the most relevant passages.
2. **Generate** – feed those passages together with the question into an LLM to produce a grounded answer.

This grounds LLM responses in up-to-date, verifiable external knowledge without modifying the model weights.

### Why do we use RAG instead of fine-tuning an LLM?

| Consideration | RAG | Fine-tuning |
|---|---|---|
| Knowledge updates | Add/remove documents instantly | Requires expensive retraining |
| Cost | Cheap (embedding + retrieval) | GPU-hours of training |
| Hallucination risk | Lower – answers cite sources | Higher – model bakes in facts |
| Domain specificity | Retrieved at runtime | Baked into weights |
| Transparency | Sources are inspectable | Opaque |

RAG is preferred when documents change frequently, budgets are limited, or auditability is important.

### What are the main components of a RAG pipeline?

1. **Document Ingestion** – parse, clean, and chunk source documents.
2. **Embedding Model** – encode chunks and queries into dense vectors.
3. **Vector Store** – persist and index embeddings (e.g., FAISS, Pinecone).
4. **Retriever** – search the vector store for the top-k most relevant chunks.
5. **Prompt Builder** – combine the question and retrieved chunks into a prompt.
6. **LLM / Generator** – produce the final answer from the prompt.
7. **Metrics & Logging** – track retrieval quality, latency, and answer relevance.

### What role does the retriever play in RAG?

The retriever is the gatekeeper of information. It converts the user query into a vector and performs a similarity search against the stored document embeddings to surface the most relevant passages. Retrieval quality directly determines the maximum quality of the final answer — if the right context is not retrieved, the LLM cannot produce a correct response.

### What role does the generator (LLM) play in RAG?

The generator reads the question and the retrieved passages and synthesises a coherent, factual answer. Its job is to:

- Reason over multiple passages.
- Synthesise information when the answer spans several chunks.
- Acknowledge uncertainty when the context is insufficient.

The LLM does **not** need to memorise facts; the retriever supplies the facts at query time.

### What is an embedding?

An embedding is a fixed-length numerical vector that encodes the semantic meaning of a piece of text. Texts with similar meaning are mapped to nearby points in vector space, regardless of exact wording. For example:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("What is RAG?")
# embedding.shape → (384,)
```

### Why do we convert text into vectors?

Computers cannot directly compare the meaning of two strings. Vectors allow:

- **Semantic similarity** – find passages that mean the same thing even if the wording differs.
- **Fast lookup** – approximate nearest-neighbour (ANN) algorithms operate on numeric arrays orders of magnitude faster than text matching.
- **Composability** – vector operations (cosine similarity, dot product) generalise across languages and domains.

### What is vector similarity search?

Given a query vector **q**, vector similarity search finds the stored vectors that are closest to **q** under a chosen distance or similarity metric. The result is an ordered list of the most semantically relevant documents.

### What similarity metrics are used in vector search?

| Metric | Formula | Notes |
|---|---|---|
| **Cosine similarity** | `(A·B) / (‖A‖ ‖B‖)` | Direction only; scale-invariant. Most common in NLP. |
| **Dot product (inner product)** | `A·B` | Equivalent to cosine when vectors are normalised. |
| **Euclidean distance (L2)** | `‖A−B‖₂` | Sensitive to magnitude; used in FAISS `IndexFlatL2`. |
| **Manhattan distance (L1)** | `Σ|Aᵢ−Bᵢ|` | Less common; robust to outliers. |

This project normalises embeddings and uses `IndexFlatIP` (inner product), making it equivalent to cosine similarity.

### What is cosine similarity?

Cosine similarity measures the angle between two vectors, ignoring their magnitude:

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

- Range: −1 (opposite) to +1 (identical direction).
- A value of 1 means the two texts have the same semantic direction; 0 means orthogonal (unrelated).

---

## 2️⃣ Embeddings & Vector DB

### How are embeddings generated?

Embeddings are produced by a neural encoder (usually a Transformer) trained with contrastive objectives (e.g., SBERT, SimCSE) so that semantically similar inputs produce nearby vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(["chunk 1", "chunk 2"], show_progress_bar=False)
# vectors.shape → (2, 384)
```

### What embedding model did you use and why?

This project uses **`sentence-transformers/all-MiniLM-L6-v2`** (384 dimensions):

- **Small & fast** – 22 M parameters; encodes 14 000+ sentences/second on CPU.
- **Good quality** – consistently strong on semantic similarity benchmarks (SBERT leaderboard).
- **Open-source** – no API key or cost; runs locally.
- Alternatives like `text-embedding-ada-002` (OpenAI) give higher accuracy but add latency and cost.

### Why is vector dimensionality important?

Dimensionality determines:

- **Storage** – each float32 dimension uses 4 bytes. 1 M vectors × 384 dims = ~1.5 GB.
- **Speed** – higher dimensions make ANN searches slower.
- **Expressiveness** – too few dimensions lose semantic nuance.

384 dimensions offers a practical balance for most QA tasks.

### What happens if embeddings are very high dimensional?

The **curse of dimensionality** causes:

1. Distance metrics become less discriminative — all points appear equally distant.
2. ANN index construction and query time grow substantially.
3. Memory use increases proportionally.

Mitigation: dimensionality reduction (PCA, MRL), quantisation (int8/binary), or purpose-built models with smaller output dims.

### What is a vector database?

A vector database stores embedding vectors alongside metadata and exposes fast ANN search, filtering, and CRUD operations. Examples: FAISS (library), Pinecone, Weaviate, Qdrant, Chroma, Milvus.

### Why did you use FAISS instead of Pinecone or Weaviate?

| Criterion | FAISS | Pinecone / Weaviate |
|---|---|---|
| Cost | Free, runs locally | Managed service, usage fees |
| Deployment | No extra infra | Separate server / SaaS |
| Scalability | Single-node (millions of vectors) | Multi-node, hosted |
| Features | Indexing + search only | Metadata filtering, namespaces, auto-embed |
| Use case | Prototypes, local deployments | Production at scale |

For a self-contained demo on Hugging Face Spaces, FAISS avoids external dependencies and keeps the stack simple.

### What are FAISS indexes?

FAISS (Facebook AI Similarity Search) provides a family of indexes:

| Index | Description | When to use |
|---|---|---|
| `IndexFlatL2` / `IndexFlatIP` | Exact brute-force | Small datasets (<100 k vectors) |
| `IndexIVFFlat` | Inverted file; approximate | Medium datasets |
| `IndexIVFPQ` | IVF + product quantisation | Large datasets, memory constrained |
| `IndexHNSWFlat` | Graph-based ANN | High-recall, fast query |

### Difference between Flat index and IVF index

- **Flat** (`IndexFlatL2`/`IndexFlatIP`): scans every vector exhaustively. 100% recall. O(n) query time. Impractical for millions of vectors.
- **IVF** (`IndexIVFFlat`): clusters vectors into Voronoi cells (inverted lists). At query time only `nprobe` cells are searched. Sub-linear query time. Slight recall trade-off.

### What is HNSW indexing?

**Hierarchical Navigable Small World** (HNSW) builds a multi-layer proximity graph. Each layer is a sparser "highway" graph. Search starts at the top layer and navigates greedily down to the bottom, finding nearest neighbours efficiently.

- Near O(log n) query time.
- Very high recall (>99 % achievable).
- Higher memory and build time than IVF.
- Used by default in Qdrant and Weaviate.

### What is approximate nearest neighbor (ANN) search?

ANN search trades a small amount of recall for dramatically lower latency and memory. Instead of comparing the query to every stored vector (exact NN), algorithms like IVF, HNSW, or ScaNN partition or graph-index the space so only a subset of vectors is examined. The result is a list that contains the true nearest neighbours with high probability (e.g., 95–99 % recall).

---

## 3️⃣ Document Processing

### How do you process PDFs for RAG?

1. **Extract text** – use `PyPDF2` (or `pdfplumber`/`pdfminer`) for digital PDFs.
2. **OCR fallback** – if extracted text is sparse (<100 chars/page), convert pages to images with `pdf2image` and run `pytesseract`.
3. **Clean** – remove headers/footers, normalise whitespace.
4. **Chunk** – split into overlapping segments (see below).
5. **Embed & store** – encode chunks and add to the vector index.

```python
from PyPDF2 import PdfReader

reader = PdfReader("document.pdf")
text = "\n".join(page.extract_text() or "" for page in reader.pages)
```

### What is OCR and when is it needed?

**Optical Character Recognition** (OCR) converts images of text (scanned documents, image-only PDFs) into machine-readable strings. It is needed when:

- A PDF contains scanned pages rather than embedded text.
- PDF text extraction yields empty or garbled output.
- Source materials are photos, screenshots, or handwritten notes.

This project uses `pytesseract` + `pdf2image` as a fallback when `PyPDF2` extracts fewer than 100 characters.

### What challenges arise in PDF text extraction?

- **Multi-column layouts** – text streams interleave columns incorrectly.
- **Tables and figures** – rendered as images; text extractors skip them.
- **Scanned/image PDFs** – no embedded text layer; OCR required.
- **Encoding issues** – non-standard fonts produce garbled characters.
- **Headers & footers** – repeated boilerplate pollutes chunks.
- **Hyphenation and ligatures** – words split across lines/glyphs.

### Why is text chunking necessary in RAG?

LLM context windows are finite. Feeding entire documents would exceed token limits and drown the relevant passage in noise. Chunking:

1. Keeps each unit within the embedding model's max sequence length.
2. Makes retrieval more precise — a small, focused chunk scores higher than a noisy full document.
3. Reduces LLM prompt size and therefore latency/cost.

### What chunk size did you use and why?

`CHUNK_SIZE = 800` characters, `CHUNK_OVERLAP = 120` characters (configurable via environment variables):

- 800 chars ≈ 150–200 tokens — fits comfortably within most embedding models (512 token limit for `all-MiniLM-L6-v2`).
- Large enough to contain a complete idea (a paragraph or two).
- Overlap of 120 chars prevents important sentences from being cut at a chunk boundary.

### What happens if chunks are too large?

- A chunk may contain multiple unrelated topics, reducing embedding specificity.
- The cosine similarity between a narrow query and a broad chunk is diluted.
- The chunk may exceed the embedding model's token limit, causing truncation.
- The LLM prompt becomes long and expensive; irrelevant content crowds out signal.

### What happens if chunks are too small?

- Individual chunks lack context — a single sentence is often ambiguous without surrounding text.
- More chunks per document increases index size and retrieval time.
- The same logical answer may be split across two non-retrieved chunks.
- Each embed-and-store call adds overhead, slowing ingestion.

### What is overlapping chunking?

Overlapping chunking creates chunks where the end of one chunk repeats the beginning of the next. For example with `chunk_size=800, overlap=120`:

```
Chunk 1: chars  0–800
Chunk 2: chars  680–1480
Chunk 3: chars  1360–2160
```

The overlap ensures that sentences at a boundary appear in at least one complete chunk, preventing information loss at chunk edges.

---

## 4️⃣ Retrieval Optimization

### How do you improve retrieval accuracy?

- **Better embeddings** – use a domain-specific or larger model.
- **Optimal chunk size** – tune for the document type.
- **Metadata filtering** – restrict search to relevant document subsets.
- **Hybrid search** – combine dense vector search with sparse BM25.
- **Reranking** – apply a cross-encoder to reorder the top-k candidates.
- **Query rewriting** – expand or rephrase the query to surface more relevant chunks.
- **HyDE** – generate a hypothetical answer and embed it for retrieval.

### What is top-k retrieval?

Top-k retrieval returns the **k** chunks with the highest similarity score to the query. In this project the default is `TOP_K_DEFAULT = 4`. A higher k improves recall (less chance of missing the answer) but increases LLM prompt size and latency.

### How do you choose k value?

- Start with k = 4–5 for typical QA tasks.
- Increase k if the answer often spans many chunks or documents are fragmented.
- Decrease k if the LLM context window is small or latency is critical.
- Empirically evaluate with a labelled question set, measuring answer quality vs. latency at each k.

### What is reranking in RAG?

After the retriever returns k candidates (first-stage retrieval using fast ANN), a **cross-encoder reranker** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each (query, chunk) pair jointly, reading both texts together for higher accuracy. The reranked top-n are passed to the LLM. Cost: O(k) cross-encoder forward passes — acceptable because k is small (e.g., 20).

### What is hybrid search?

Hybrid search combines **dense retrieval** (embedding similarity) with **sparse retrieval** (keyword matching, e.g., BM25). The two score lists are merged via reciprocal rank fusion (RRF) or a weighted sum. It handles both semantic relevance (dense) and exact keyword matches (sparse).

### Difference between BM25 and vector search

| | BM25 | Vector Search |
|---|---|---|
| Representation | Sparse TF-IDF-like term weights | Dense semantic embedding |
| Matches | Exact/near-exact keywords | Synonyms, paraphrases |
| Out-of-vocabulary | Handles novel terms well | Novel terms mapped to similar vectors |
| Speed | Very fast (inverted index) | Requires ANN index |
| Recall for synonyms | Poor | Good |

### When would you combine keyword search + embeddings?

- When queries contain rare proper nouns or technical terms that semantic embeddings may not distinguish from common words.
- When exact-match recall is critical (legal, medical).
- When the vector index is not yet trained on the domain vocabulary.
- General best practice for production RAG to maximise recall.

---

## 5️⃣ Prompt Engineering in RAG

### How do you structure the prompt for the LLM?

A typical RAG prompt:

```
You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't know."

Context:
[Source: doc1.pdf] <chunk text>

[Source: doc2.pdf] <chunk text>

Question: <user question>
Answer:
```

Key elements: role instruction, strict grounding constraint, labelled context passages, clear question/answer delimiter.

### How do you insert retrieved documents into prompts?

Each chunk is prefixed with its source identifier and appended sequentially. In this project:

```python
context_text = "\n\n".join(
    f"[Source: {c['source']}] {c['chunk']}" for c in contexts
)
```

Best practices:
- Most relevant chunk first (highest score) so the LLM reads it early.
- Keep total context within the model's context window.
- Separate chunks with clear delimiters.

### How do you avoid hallucination in RAG?

1. **Strict system instruction** – "Answer ONLY from the context. Do not use prior knowledge."
2. **Citation enforcement** – ask the LLM to cite the source for each claim.
3. **Confidence threshold** – if the top retrieval score is below a threshold, return "I don't know."
4. **Answer verification** – check the answer against the retrieved text post-generation.
5. **Low temperature** – set `temperature=0` for deterministic, fact-based generation.

### How do you ensure answers come only from retrieved documents?

- Explicit prompt instruction: *"Use only the context below."*
- Instruct the model to output "not found in provided context" when the answer is absent.
- Post-process: verify each sentence of the answer against source chunks.
- Use structured output (e.g., JSON with `answer` + `source_ids`) to enforce traceability.

### What is context window limitation?

The context window is the maximum number of tokens an LLM can process in a single forward pass. Examples:

| Model | Context window |
|---|---|
| GPT-3.5-turbo | 16 k tokens |
| GPT-4o | 128 k tokens |
| Gemini 2.0 Flash | 1 M tokens |
| Llama 3 8B | 8 k tokens |

In RAG, the entire prompt (system message + retrieved chunks + question) must fit within this window. Retrieving many large chunks can exceed it, requiring truncation or smarter context selection.

---

## 6️⃣ System Design Questions

### How would you scale a RAG system for millions of documents?

1. **Distributed vector store** – Weaviate / Qdrant cluster or Pinecone for horizontal scaling.
2. **IVF or HNSW indexes** – avoid brute-force search.
3. **Async ingestion pipeline** – Kafka/Celery workers for background embedding and indexing.
4. **Read replicas** – shard the vector index across nodes; route queries with a load balancer.
5. **Embedding server** – dedicated GPU service (e.g., TGI, Triton) to handle burst traffic.
6. **Caching** – cache embeddings and frequent query results (see below).

### How would you handle frequent document updates?

- **Soft delete** – mark stale chunks in metadata; filter them out at retrieval time.
- **Delta indexing** – re-embed and re-add only changed chunks; remove old chunk IDs.
- **Versioning** – store document version in metadata; query only the latest version.
- **Scheduled re-ingestion** – background job polls for changes and updates the index.
- FAISS does not support in-place deletion; use IVF with `remove_ids` or switch to Qdrant/Weaviate which support native updates.

### How would you reduce RAG latency?

| Technique | Typical savings |
|---|---|
| Cache frequent queries | Eliminates retrieval + LLM for warm cache hits |
| Smaller embedding model | 2–5× faster encoding |
| GPU for embedding | 10–50× faster than CPU |
| ANN index (HNSW/IVF) vs Flat | Sub-linear vs linear search |
| Reduce top-k | Fewer chunks → shorter prompt → faster LLM |
| Streaming responses | First token faster; perceived latency lower |
| Async generation | Non-blocking I/O in FastAPI |

### How do you handle large context documents?

- **Hierarchical chunking** – create summary chunks at paragraph/section level for a first coarse retrieval pass, then refine with fine-grained chunks.
- **Map-reduce** – split the document into windows, summarise each, then combine summaries.
- **Long-context models** – Gemini 1.5 / Claude 3 support 1 M+ token windows.
- **Selective extraction** – use rule-based extraction (tables, sections) before embedding.

### How would you implement caching in RAG?

- **Query-level cache** – hash the query; if the hash exists return the cached answer (exact-match).
- **Semantic cache** – embed the query; if a stored query is within cosine distance ε, return its cached answer (fuzzy match). Libraries: GPTCache.
- **Embedding cache** – cache the embedding for each unique text to avoid re-encoding.
- **Redis** or in-memory dict for hot cache; Postgres/S3 for cold storage.

### How would you deploy a RAG system in production?

1. **Containerise** – Docker image with the FastAPI app (as in this project's `Dockerfile`).
2. **Orchestrate** – Kubernetes or AWS ECS for auto-scaling.
3. **Persistent vector store** – external managed service (Pinecone, Weaviate) or a StatefulSet with persistent volumes.
4. **Secrets management** – inject API keys via environment variables or Vault; never bake into images.
5. **CI/CD** – automated tests, image build, and deploy on each push.
6. **Observability** – structured logging, Prometheus metrics, distributed tracing (OpenTelemetry).
7. **Rate limiting & auth** – API gateway (Kong, AWS API GW) before the service.

---

## 7️⃣ Real Production Problems

### What if the retriever returns irrelevant documents?

- **Symptom**: low cosine scores for all retrieved chunks.
- **Causes**: domain mismatch, poor chunk quality, or a mis-posed query.
- **Fixes**:
  - Lower similarity threshold and return "I don't know" for low-scoring results.
  - Improve chunking (better boundaries, size tuning).
  - Use a domain-adapted embedding model.
  - Add hybrid (BM25 + vector) search.
  - Log queries and retrieved chunks for periodic human review.

### What if the LLM hallucinates even with context?

- Use a lower temperature (0 or 0.1).
- Add chain-of-thought instruction: "Quote the passage that supports your answer."
- Apply a post-generation grounding check (NLI model to verify entailment).
- Switch to a model fine-tuned for faithfulness (e.g., Llama-3-RAG-instruct).

### What if multiple documents contradict each other?

- Include all contradicting passages in the context and instruct the LLM to acknowledge conflicting information.
- Tag documents with date/version metadata; instruct the model to prefer the most recent.
- Return multiple candidate answers with source attribution and let the user decide.

### How do you evaluate RAG system performance?

#### Retrieval metrics
| Metric | Description |
|---|---|
| **Recall@k** | Fraction of relevant chunks in the top-k results |
| **MRR** | Mean reciprocal rank of the first relevant result |
| **NDCG@k** | Normalised discounted cumulative gain |

#### Generation metrics
| Metric | Description |
|---|---|
| **Answer Relevance** | Does the answer address the question? (LLM judge or ROUGE) |
| **Faithfulness** | Are claims in the answer supported by the context? |
| **Context Precision** | Fraction of retrieved chunks that are actually relevant |
| **Context Recall** | Fraction of relevant information covered by retrieved chunks |

#### Operational metrics
- **Latency** (p50/p95/p99 end-to-end and per stage)
- **Throughput** (queries/second)
- **Cost** (tokens used, API calls)

Frameworks: **RAGAS**, **TruLens**, **DeepEval**.

---

## 8️⃣ Advanced Questions

### What is multi-hop retrieval?

Multi-hop retrieval answers questions that require chaining multiple pieces of evidence. For example, "Who is the CEO of the company that acquired Slack?" requires first retrieving "Salesforce acquired Slack" and then retrieving "Marc Benioff is the CEO of Salesforce". The system iteratively retrieves and reasons until the answer is assembled.

### What is query rewriting in RAG?

Query rewriting uses an LLM to reformulate the user's question before retrieval to improve recall:

- **HyDE (Hypothetical Document Embeddings)** – generate a hypothetical answer and embed that for retrieval instead of the question.
- **Expansion** – add synonyms and related terms to the query.
- **Decomposition** – split complex multi-part questions into simpler sub-queries retrieved separately.
- **Step-back prompting** – rephrase to a more general form.

### What is self-RAG?

Self-RAG (Asai et al., 2023) trains a single LLM to adaptively decide:

1. Whether retrieval is needed for a given token span.
2. Whether a retrieved passage is relevant.
3. Whether the generated segment is supported by the retrieved passage.
4. Whether the response is useful overall.

Special reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) are generated inline, allowing the model to self-critique and selectively retrieve, producing higher faithfulness with fewer unnecessary retrievals.

### What is agentic RAG?

Agentic RAG wraps the retriever and generator in an autonomous agent loop (e.g., ReAct, LangChain Agent):

1. The agent decides whether to retrieve, what query to issue, and how many times.
2. It can call multiple tools (web search, SQL, calculator) alongside vector retrieval.
3. Results are fed back to the agent iteratively until it determines the answer is complete.

This handles complex, multi-step research tasks that single-shot RAG cannot.

### What is graph RAG?

Graph RAG (Microsoft Research, 2024) indexes documents as a **knowledge graph** (entities + relationships) in addition to text chunks. At query time:

- Global queries leverage graph community summaries for high-level synthesis.
- Local queries combine entity-centric subgraph retrieval with traditional chunk retrieval.

Graph RAG outperforms plain RAG on questions requiring cross-document reasoning and holistic understanding of a large corpus.

### What is knowledge distillation for RAG?

Knowledge distillation for RAG transfers the behaviour of a large, accurate RAG system (teacher) into a smaller, faster model (student):

- The student learns to generate answers directly from the question, internalising common retrieval patterns.
- Reduces inference latency by avoiding real-time retrieval for frequent queries.
- Approaches include fine-tuning the student on (question, teacher-answer) pairs where teacher answers were produced with full RAG context.

---

## 9️⃣ Python / Implementation Questions

### How would you generate embeddings in Python?

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("What is RAG?")
# numpy array of shape (384,)
```

### How do you store vectors in FAISS?

```python
import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)          # exact L2 index
vectors = np.random.rand(1000, dimension).astype("float32")
index.add(vectors)                            # add 1000 vectors
print(index.ntotal)                           # 1000

# Persist
faiss.write_index(index, "index.faiss")

# Reload
index = faiss.read_index("index.faiss")
```

For inner-product / cosine (requires normalised vectors):

```python
index = faiss.IndexFlatIP(dimension)
norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
vectors_norm = (vectors / norms).astype("float32")
index.add(vectors_norm)
```

### How do you retrieve top results?

```python
query_vector = model.encode(["What is RAG?"]).astype("float32")
# Normalise for cosine
query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True) + 1e-12

D, I = index.search(query_vector, k=5)
# D → similarity scores, shape (1, 5)
# I → indices into the stored vectors, shape (1, 5)
for score, idx in zip(D[0], I[0]):
    print(f"Score: {score:.4f}, Chunk index: {idx}")
```

---

## 🔥 Bonus: Key Distinctions

### RAG vs Fine-tuning vs Agents vs Knowledge Graphs

| | RAG | Fine-tuning | Agents | Knowledge Graphs |
|---|---|---|---|---|
| **How it works** | Retrieve relevant text at query time; feed to LLM | Update model weights on domain data | LLM calls tools iteratively in a loop | Structured entity-relation graph queried symbolically |
| **Knowledge update** | Instant (add documents) | Requires retraining | Instant (update tool data) | Update graph nodes/edges |
| **Strengths** | Low cost, auditable, current knowledge | Deep domain adaptation, no retrieval latency | Complex multi-step reasoning, tool use | Precise relational queries, no hallucination |
| **Weaknesses** | Retrieval quality bottleneck | Expensive, knowledge becomes stale | High latency, unpredictable | Knowledge graph construction cost |
| **Best for** | Document QA, enterprise search | Specialised text style/format tasks | Research assistants, workflow automation | Structured data, compliance |
| **Hallucination risk** | Low (grounded in sources) | Medium | Medium | Very low |
| **Example tools** | LangChain, LlamaIndex, FAISS | LoRA, QLoRA, full fine-tune | LangGraph, AutoGen | Neo4j, Microsoft GraphRAG |

---

*For implementation details specific to this project, see [EXPLANATIONS.md](EXPLANATIONS.md).*
