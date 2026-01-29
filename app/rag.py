import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import List, Dict

import faiss
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from huggingface_hub import HfApi, hf_hub_download

from .config import (
    INDEX_PATH,
    METADATA_PATH,
    METRICS_PATH,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    GOOGLE_API_KEY,
    HF_REPO,
    HF_TOKEN,
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
    if not METADATA_PATH.exists() and HF_REPO:
        try:
            hf_hub_download(repo_id=HF_REPO, filename="metadata.json", local_dir=DATA_DIR, token=HF_TOKEN)
        except Exception:
            pass  # If download fails, start empty
    if not METADATA_PATH.exists():
        return []
    return json.loads(METADATA_PATH.read_text())


def _save_metadata(metadata: List[Dict]):
    _ensure_dirs()
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    if HF_REPO and HF_TOKEN:
        try:
            api = HfApi()
            api.upload_file(path_or_fileobj=str(METADATA_PATH), path_in_repo="metadata.json", repo_id=HF_REPO, token=HF_TOKEN)
            print("Uploaded metadata to HF Hub")
        except Exception as e:
            print(f"Failed to upload metadata: {e}")


def _load_index(dimension: int):
    _ensure_dirs()
    if not INDEX_PATH.exists() and HF_REPO:
        try:
            hf_hub_download(repo_id=HF_REPO, filename="index.faiss", local_dir=DATA_DIR, token=HF_TOKEN)
        except Exception:
            pass
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return faiss.IndexFlatIP(dimension)


def _save_index(index):
    _ensure_dirs()
    faiss.write_index(index, str(INDEX_PATH))
    if HF_REPO and HF_TOKEN:
        try:
            api = HfApi()
            api.upload_file(path_or_fileobj=str(INDEX_PATH), path_in_repo="index.faiss", repo_id=HF_REPO, token=HF_TOKEN)
            print("Uploaded index to HF Hub")
        except Exception as e:
            print(f"Failed to upload index: {e}")


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
    """Read PDF - use OCR if text extraction yields little content."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    text = "\n".join(pages).strip()
    
    # If very little text extracted, try OCR
    if len(text) < 100:
        try:
            from pdf2image import convert_from_path
            import pytesseract
            images = convert_from_path(str(path))
            ocr_pages = []
            for img in images:
                ocr_pages.append(pytesseract.image_to_string(img))
            text = "\n".join(ocr_pages)
        except Exception:
            pass  # Fall back to original text
    return text


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
    if not GOOGLE_API_KEY:
        return (
            "No Gemini API key configured. Here are the most relevant excerpts:\n\n"
            + context_text
        )

    prompt = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return response.text.strip() or context_text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            return (
                f"LLM generation failed: {error_str}. Here are the most relevant excerpts:\n\n"
                + context_text
            )


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
