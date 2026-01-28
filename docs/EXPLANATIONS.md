# Mandatory Explanations

## Chunk size choice
I chose a **chunk size of 800 characters with 120-character overlap**. This keeps each chunk small enough for the embedding model to capture a coherent semantic unit, while still preserving sentence continuity across boundaries. With typical English text, 800 characters is roughly 120–160 words, which balances recall (enough context) and precision (not too much noise) for similarity search.

## Retrieval failure case observed
A common failure case is **cross-page entity linking** in PDFs. If a definition spans a page break, the PDF extractor may insert line breaks or remove hyphenation, causing a chunk to miss the key entity name. This makes the chunk less similar to the query and can lead to a missed retrieval even though the answer is present in the document.

## Metric tracked
I tracked **latency** for ingestion and question answering (stored in `data/metrics.jsonl`). Each ingestion run logs the total time spent chunking + embedding, and each QA request returns `latency_ms` in the API response. This helps identify slowdowns when documents are large or embedding computation is heavy.
