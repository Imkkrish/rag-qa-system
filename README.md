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
This project provides a minimal RAG pipeline using FastAPI + FAISS for retrieval and a Streamlit UI for interaction. Documents are ingested in the background, chunked, embedded, and stored locally. Queries retrieve relevant chunks and generate answers using Google Gemini API.

## Features
- PDF + TXT ingestion (with OCR support for scanned PDFs)
- Chunking + embeddings (SentenceTransformers)
- FAISS vector store
- Background ingestion job
- FastAPI endpoints with Pydantic validation
- Basic rate limiting
- Streamlit UI
- Metrics logging (latency)

## Architecture Diagram
See [docs/architecture.drawio](docs/architecture.drawio).

## Setup
1. Create a Hugging Face dataset repository for storing vector data (e.g., `Imkkrish/rag-data`).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure environment variables (copy `.env.example` to `.env`):
   - `GOOGLE_API_KEY` (required for LLM generation)
   - `HF_REPO` (your HF dataset repo, e.g., `Imkkrish/rag-data`)
   - `HF_TOKEN` (your HF token with write access)
4. Run the API:
   - `uvicorn app.main:app --reload`
5. Run Streamlit:
   - `streamlit run streamlit_app.py`

## API Endpoints
- `POST /documents/upload` (multipart file) → returns `job_id`
- `GET /documents/status/{job_id}` → job status
- `POST /qa` → answer + contexts
- `GET /health`

## Usage
1. Upload a document using the Streamlit UI or `POST /documents/upload`.
2. Wait for job completion using `/documents/status/{job_id}`.
3. Ask questions via the UI or `POST /qa`.

## Mandatory Explanations
See [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md).

## Notes
- If `GOOGLE_API_KEY` is not set, the system returns retrieved chunks instead of an LLM answer.
- This implementation avoids heavy frameworks and uses a lightweight local FAISS index.
- For HF Spaces deployment, set `HF_REPO` and `HF_TOKEN` in secrets to persist data across sessions.
