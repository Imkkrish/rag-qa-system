---
title: RAG QA System
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
python_version: "3.11"
app_file: app.py
pinned: false
---

# Knowledge Nexus - RAG QA System

Knowledge Nexus is a Retrieval-Augmented Generation (RAG) system built with FastAPI and Streamlit. It allows users to upload PDF and TXT documents, indexes them using FAISS and SentenceTransformers, and provides an interface to query the documents using Google's Gemini 1.5 Flash.

## ✨ Features

- **Multi-Format Support**: Upload PDF and TXT files.
- **Efficient Retrieval**: Uses FAISS for incredibly fast similarity search.
- **LLM Powered**: Synthesizes answers using Gemini 1.5 Flash.
- **Robust API**:
  - Request validation using **Pydantic**.
  - Rate limiting via **SlowAPI**.
  - Background task processing for document ingestion.
- **Premium UI**: Sleek, text-focused Streamlit interface.

## 🚀 Setup & Usage

### 1. Requirements
Ensure you have Python 3.9+ installed and a `GOOGLE_API_KEY`.

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Running the App
```bash
streamlit run app.py
```
This will start:
- **Streamlit UI**: Typically on `http://localhost:8501`
- **FastAPI Backend**: On `http://localhost:8000`

### 4. API Documentation
- **POST `/upload`**: Upload a document (PDF/TXT).
- **POST `/ask`**: Query the index. 
  Example Payload: `{"question": "How do I setup?", "k": 3}`

## 📖 Mandatory Explanations
For details on chunking strategies, retrieval failures, and metrics, see [explanations.md](explanations.md).

## 📐 Architecture
Visulize the system flow in [architecture.md](architecture.md).
