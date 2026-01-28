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

# RAG-Based Question Answering System

This system uses **Streamlit** for its user interface and **FastAPI** for its background API, running on **ZeroGPU**.

## 🚀 How it works

1. **Upload**: Use the sidebar to upload PDF or TXT files.
2. **Retrieve**: FAISS finds the most relevant chunks from your documents.
3. **Generate**: Gemini 1.5 Flash synthesizes an answer using **ZeroGPU**.

## 🛠️ API Access

The FastAPI endpoints are available at:

- `POST /upload`: Upload documents.
- `POST /ask`: Query the documents.

_Note: The API runs on a background thread within the Streamlit process._

Make sure to set `GOOGLE_API_KEY` in your Space Secrets.
