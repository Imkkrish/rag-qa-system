---
title: RAG QA System
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RAG-Based Question Answering System

This is a Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, and Google's Gemini LLM. It allows uploading PDF/TXT documents and asking questions based on their content.

## Features

- **FastAPI**: Modern, fast web framework for APIs.
- **FAISS**: Efficient similarity search for document chunks.
- **Gemini 1.5 Flash**: High-performance LLM for generating accurate answers.
- **Background Jobs**: Asynchronous document processing to prevent API blocking.
- **ZeroGPU Support**: Configured for Hugging Face Spaces.

## Setup & Configuration

Add `GOOGLE_API_KEY` to your Hugging Face Space Secrets.
