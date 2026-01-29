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
This project provides a minimal RAG pipeline using FastAPI with a built-in web interface. Documents are ingested, chunked, embedded, and stored locally or on HF Hub. Queries retrieve relevant chunks and generate answers using Google Gemini API.

## Features
- PDF + TXT ingestion (with OCR support for scanned PDFs)
- Chunking + embeddings (SentenceTransformers)
- FAISS vector store with HF Hub persistence
- Background ingestion job
- FastAPI with web UI (HTML templates)
- API endpoints with Pydantic validation
- Basic rate limiting
- Metrics logging (latency)

## Architecture Diagram
See [docs/architecture.drawio](docs/architecture.drawio).

## Setup
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Configure environment variables (copy `.env.example` to `.env`):
   - `GOOGLE_API_KEY` (required for LLM generation)
   - `HF_REPO` (your HF dataset repo, e.g., `imkrish/rag-data`)
   - `HF_TOKEN` (your HF token with write access)
3. Run the app:
   - `uvicorn app.main:app --reload`
4. Open http://localhost:8000 in your browser

## Docker
- Build: `docker build -t rag-qa .`
- Run: `docker run -p 8000:8000 rag-qa`

## API Endpoints
- `GET /` - Web interface
- `POST /upload` - Upload and ingest document (web form)
- `POST /ask` - Ask question (web form)
- `POST /documents/upload` (multipart file) → returns `job_id`
- `GET /documents/status/{job_id}` → job status
- `POST /qa` → answer + contexts
- `GET /health`

## Usage
1. Open http://localhost:8000
2. Upload a document using the web form
3. Ask questions via the web interface

## Mandatory Explanations
See [docs/EXPLANATIONS.md](docs/EXPLANATIONS.md).

## Notes
- If `GOOGLE_API_KEY` is not set, the system returns retrieved chunks instead of an LLM answer.
- This implementation avoids heavy frameworks and uses a lightweight local FAISS index.
- For HF Spaces deployment, set `HF_REPO` and `HF_TOKEN` in secrets to persist data across sessions.
