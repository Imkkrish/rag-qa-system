import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import List, Dict

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from .config import (
    INDEX_PATH,
    METADATA_PATH,
    METRICS_PATH,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HF_MODEL,
    HF_API_TOKEN,
)

_index_lock = Lock()
_model = None


def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _load_metadata() -> List[Dict]:
    _ensure_dirs()
    if not METADATA_PATH.exists():
        return []
    return json.loads(METADATA_PATH.read_text())


def _save_metadata(metadata: List[Dict]):
    _ensure_dirs()
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


def _load_index(dimension: int):
    _ensure_dirs()
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return faiss.IndexFlatIP(dimension)


def _save_index(index):
    _ensure_dirs()
    faiss.write_index(index, str(INDEX_PATH))


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".txt":
        return _read_txt(path)
    raise ValueError("Unsupported file type")


def add_document_chunks(chunks: List[str], doc_id: str, source: str) -> int:
    if not chunks:
        return 0
    model = _load_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = _normalize(np.array(embeddings).astype("float32"))

    with _index_lock:
        metadata = _load_metadata()
        index = _load_index(embeddings.shape[1])
        start_id = len(metadata)
        index.add(embeddings)
        for i, chunk in enumerate(chunks):
            metadata.append({
                "id": start_id + i,
                "doc_id": doc_id,
                "source": source,
                "chunk": chunk,
            })
        _save_index(index)
        _save_metadata(metadata)
    return len(chunks)


def search(query: str, top_k: int) -> List[Dict]:
    model = _load_model()
    query_vec = model.encode([query], show_progress_bar=False)
    query_vec = _normalize(np.array(query_vec).astype("float32"))

    metadata = _load_metadata()
    if not metadata:
        return []

    with _index_lock:
        index = _load_index(query_vec.shape[1])
        scores, ids = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = metadata[int(idx)]
        results.append({
            "score": float(score),
            "chunk": item["chunk"],
            "source": item["source"],
            "doc_id": item["doc_id"],
        })
    return results


def _log_metric(payload: Dict):
    _ensure_dirs()
    with METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def generate_answer(question: str, contexts: List[Dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['source']}] {c['chunk']}" for c in contexts
    )
    if not HF_API_TOKEN:
        return (
            "No HF token configured. Here are the most relevant excerpts:\n\n"
            + context_text
        )

    prompt = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        return (
            "LLM generation failed. Here are the most relevant excerpts:\n\n"
            + context_text
        )
    data = response.json()
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "").strip() or context_text
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip() or context_text
    return context_text


def ingest_document(path: Path, doc_id: str, source: str) -> Dict:
    started = time.time()
    text = parse_document(path)
    chunks = chunk_text(text)
    count = add_document_chunks(chunks, doc_id=doc_id, source=source)
    _log_metric({
        "event": "ingest",
        "doc_id": doc_id,
        "chunks": count,
        "latency_ms": int((time.time() - started) * 1000),
    })
    return {"chunks": count}
