# System Architecture

## Overview

The system follows a classic RAG architecture, leveraging FastAPI for the web layer, FAISS for the retrieval layer, and Gemini for the generation layer.

## Component Diagram (Mermaid)

```mermaid
graph TD
    User((User))
    API[FastAPI Interface]
    BG[Background Worker]
    DS[Document Storage]
    VS[(FAISS Vector DB)]
    ST[Sentence Transformers]
    LLM[Gemini 1.5 Flash]

    User -- "1. Uploads PDF/TXT" --> API
    API -- "2. Saves File" --> DS
    API -- "3. Triggers Task" --> BG
    BG -- "4. Extracts Text" --> DS
    BG -- "5. Chunks Text" --> BG
    BG -- "6. Embeds Chunks" --> ST
    ST -- "7. Stores Vectors" --> VS

    User -- "8. Asks Question" --> API
    API -- "9. Embeds Query" --> ST
    ST -- "10. Similarity Search" --> VS
    VS -- "11. Retrieves Chunks" --> API
    API -- "12. Sends Question + Context" --> LLM
    LLM -- "13. Generates Answer" --> API
    API -- "14. Returns Answer" --> User
```

## Workflow Detail

1.  **Ingestion Phase**:
    - The user uploads a file.
    - The system saves it temporarily and returns a "success" response immediately.
    - A background task handles the extraction of text from PDF/TXT using `PyMuPDF`.
    - Text is broken into manageable chunks (1000 chars).
    - Each chunk is converted into an embedding (384-dim vector).
    - Vectors and metadata are stored in a local FAISS index.

2.  **Query Phase**:
    - The user sends a natural language question.
    - The question is converted into the same embedding space.
    - FAISS performs an L2 distance search to find the top 4 most similar chunks.
    - These chunks are formatted into a prompt for the Gemini LLM.
    - The LLM synthesizes an answer based _only_ on the provided context.
    - The answer and sources are returned to the user.
