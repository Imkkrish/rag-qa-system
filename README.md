---
title: RAG QA System
emoji: "🧠"
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.37.1"
python_version: "3.9"
app_file: streamlit_app.py
pinned: false
---

# RAG-Based Question Answering System

## Overview
This project provides a minimal RAG pipeline using FastAPI + FAISS for retrieval and a Streamlit UI for interaction. Documents are ingested in the background, chunked, embedded, and stored locally. Queries retrieve relevant chunks and optionally call the Hugging Face Inference API to generate answers.

## Features
- PDF + TXT ingestion
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
1. Create a virtual environment and install deps:
   - `pip install -r requirements.txt`
2. Configure environment variables (copy `.env.example` to `.env`):
   - `HUGGINGFACEHUB_API_TOKEN` (optional)
   - `HF_MODEL` (optional)
3. Run the API:
   - `uvicorn app.main:app --reload`
4. Run Streamlit:
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
- If `HUGGINGFACEHUB_API_TOKEN` is not set, the system returns retrieved chunks instead of an LLM answer.
- This implementation avoids heavy frameworks and uses a lightweight local FAISS index.
