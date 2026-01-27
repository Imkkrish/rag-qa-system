---
title: RAG QA System
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.2
python_version: "3.10"
app_file: app.py
pinned: false
---

# RAG-Based Question Answering System

This system uses **FastAPI** for its API and **Gradio** for its user interface, running on **ZeroGPU**.

## 🚀 How it works

1. **Upload**: Send PDF or TXT files.
2. **Retrieve**: FAISS finds the most relevant chunks.
3. **Generate**: Gemini 1.5 Flash synthesizes an answer using **ZeroGPU**.

## 🛠️ API Access

The FastAPI endpoints are available at:

- `POST /upload`: Upload documents.
- `POST /ask`: Query the documents.

Make sure to set `GOOGLE_API_KEY` in your Space Secrets.
