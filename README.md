---
title: RAG QA System
emoji: "🧠"
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.37.1"
python_version: "3.10"
app_file: streamlit_app.py
pinned: false
---

# RAG-Based Question Answering System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using FastAPI for the backend API, FAISS for vector storage, Sentence Transformers for embeddings, and Google Gemini for answer generation. It supports PDF and TXT document ingestion with OCR fallback for scanned PDFs, background job processing, and a Streamlit UI for user interaction.

## Features
- Document ingestion (PDF, TXT) with OCR support for scanned PDFs
- Text chunking and embedding using Sentence Transformers
- FAISS vector store for similarity search
- Background job processing for ingestion
- FastAPI with Pydantic validation and basic rate limiting
- Streamlit UI for document upload and question answering
- Metrics logging (latency, chunks, etc.)

## Architecture Diagram
```
User (Browser) <-> Streamlit UI <-> FastAPI Backend
                                      |
                                      v
                                Document Ingestion (Background)
                                      |
                                      v
                            Text Extraction (PyPDF2 + OCR)
                                      |
                                      v
                            Chunking (size: 800, overlap: 120)
                                      |
                                      v
                            Embedding (SentenceTransformers)
                                      |
                                      v
                            FAISS Index + Metadata Storage
                                      |
                                      v
                            Query Processing -> Retrieval -> Generation (Gemini)
```

## Mandatory Explanations

### Why Chunk Size of 800 with 120 Overlap?
- **Chunk Size (800)**: Balances providing sufficient context for coherent answers while keeping chunks small enough for precise retrieval. Larger chunks (e.g., 2000) might include irrelevant information, reducing retrieval accuracy; smaller chunks (e.g., 500) may lose context across sentences.
- **Overlap (120)**: Ensures continuity between chunks, preventing loss of information at boundaries. This helps maintain semantic flow, especially for questions spanning multiple sentences.

### One Retrieval Failure Case Observed
- **Case**: When querying "who is krish kumar" on a resume PDF, the top retrieved chunks had low similarity scores (e.g., 0.087), leading to potential irrelevant or incomplete answers. This occurred because the query embedding didn't closely match the document chunks due to variations in phrasing or focus on specific sections.
- **Mitigation**: Increasing top_k from 1 to 4 improved coverage, and the system falls back to showing excerpts if LLM fails.

### One Metric Tracked
- **Latency**: Tracked for both ingestion (time to process and embed document) and QA (time for retrieval + generation). Logged in `metrics.jsonl` for monitoring performance. For example, ingestion latency averaged 5-10 seconds for PDFs, QA around 2-10 seconds depending on API response.

## Setup
1. Clone the repository:
   - `git clone <repo-url>`
2. Create a virtual environment:
   - `python -m venv venv`
   - `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Set up environment variables (create `.env` file):
   - `GOOGLE_API_KEY=<your-gemini-api-key>`
5. Run the FastAPI backend:
   - `uvicorn app.main:app --reload`
6. Run the Streamlit UI:
   - `streamlit run streamlit_app.py`

## API Endpoints
- `POST /documents/upload` (multipart file upload) → Returns `job_id` for background ingestion
- `GET /documents/status/{job_id}` → Job status and details
- `POST /qa` → Answer and retrieved contexts (JSON body: `{"question": "str", "top_k": 4}`)
- `GET /health` → Health check

## Usage
1. Start the backend and UI as above.
2. In the Streamlit UI, upload a PDF or TXT file.
3. Wait for ingestion (check job status if needed).
4. Ask questions; the system retrieves relevant chunks and generates answers using Gemini.

## Evaluation Criteria Alignment
- **Chunking Strategy**: 800/120 overlap for optimal context vs. precision.
- **Retrieval Quality**: FAISS similarity search with normalized embeddings; fallback to excerpts.
- **API Design**: RESTful with Pydantic, background jobs, rate limiting.
- **Metrics Awareness**: Latency logging for performance monitoring.
- **System Explanation**: Detailed in this README and code comments.

## GitHub Repository
[Link to GitHub Repo](https://github.com/your-username/rag-qa-system)  # Replace with actual link
3. Ask questions via the UI or `POST /qa`.

## Mandatory Explanations
See [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md).

## Notes
- If `HUGGINGFACEHUB_API_TOKEN` is not set, the system returns retrieved chunks instead of an LLM answer.
- This implementation avoids heavy frameworks and uses a lightweight local FAISS index.
