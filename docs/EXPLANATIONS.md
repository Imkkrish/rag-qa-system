# Mandatory Explanations

## Chunk Size Choice
I chose a chunk size of 1000 characters with 200 character overlap. This balances context preservation and retrieval granularity. Smaller chunks (e.g., 500 chars) risked losing semantic context across sentences, while larger chunks (e.g., 2000 chars) could dilute relevance in searches. The 200 char overlap ensures continuity between chunks, reducing the chance of splitting related information. This was tested with sample documents to ensure chunks contain coherent ideas without being too verbose for embedding efficiency.

## Retrieval Failure Case
One observed retrieval failure was with the question "who is krish kumar" initially returning no contexts because the document ingestion failed due to OCR processing taking too long in the background job, causing the task to not complete. The vector store was empty, leading to zero results. This was fixed by implementing local fallbacks in the UI and ensuring ingestion completes before querying. Another potential failure is low similarity scores for out-of-domain questions, but the system handles this by falling back to excerpts.

## Tracked Metric
I tracked query latency as a key metric, logging it in `metrics.jsonl` for each ingestion and QA operation. This helps monitor system performance and identify bottlenecks, such as slow OCR or embedding generation. For example, ingestion latency averaged 5-10 seconds for PDFs, while QA was under 2 seconds excluding LLM calls. This metric is crucial for optimizing user experience in a real-time system.
