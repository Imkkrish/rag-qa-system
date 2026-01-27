# RAG-Based Question Answering System

This is a Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, and Google's Gemini LLM. It allows uploading PDF/TXT documents and asking questions based on their content.

## Features

- **FastAPI**: Modern, fast web framework for APIs.
- **FAISS**: Efficient similarity search for document chunks.
- **Gemini 1.5 Flash**: High-performance LLM for generating accurate answers.
- **Background Jobs**: Asynchronous document processing to prevent API blocking.
- **Rate Limiting**: Integrated using `slowapi` to prevent abuse.
- **Pydantic Validation**: Robust input/output validation.

## Prerequisites

- Python 3.9+
- Google API Key (for Gemini)

## Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd banao-Tech
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the root directory:

   ```env
   GOOGLE_API_KEY=your_key_here
   ```

5. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

## Usage

### 1. Upload a Document

- **Endpoint**: `POST /upload`
- **Body**: Form-data with key `file` (PDF or TXT)
- **Example (cURL)**:
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/upload' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@your_document.pdf'
  ```

### 2. Ask a Question

- **Endpoint**: `POST /ask`
- **Body**: JSON
  ```json
  {
    "question": "What is the main topic of the document?"
  }
  ```
- **Example (cURL)**:
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/ask' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "What is the main topic of the document?"
  }'
  ```

## Architecture

See `architecture.md` for more details.

## Explanations

See `explanations.md` for mandatory technical explanations.
