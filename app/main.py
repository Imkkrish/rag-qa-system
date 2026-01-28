import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from .config import UPLOAD_DIR, TOP_K_DEFAULT
from .models import UploadResponse, JobStatus, QARequest, QAResponse
from .storage import create_job, update_job, get_job
from .rag import ingest_document, search, generate_answer

app = FastAPI(title="RAG-Based Question Answering System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


@app.post("/documents/upload", response_model=UploadResponse)
@limiter.limit("10/minute")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT are supported")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid4())
    dest_path = UPLOAD_DIR / f"{doc_id}{suffix}"
    content = await file.read()
    dest_path.write_bytes(content)

    job_id = create_job(file.filename)
    update_job(job_id, "processing")

    def _run():
        try:
            ingest_document(dest_path, doc_id=doc_id, source=file.filename)
            update_job(job_id, "completed")
        except Exception as exc:
            update_job(job_id, "failed", error=str(exc))

    background_tasks.add_task(_run)

    return UploadResponse(job_id=job_id, status="processing")


@app.get("/documents/status/{job_id}", response_model=JobStatus)
@limiter.limit("30/minute")
async def document_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


@app.post("/qa", response_model=QAResponse)
@limiter.limit("30/minute")
async def ask_question(payload: QARequest):
    started = time.time()
    contexts = search(payload.question, payload.top_k or TOP_K_DEFAULT)
    answer = generate_answer(payload.question, contexts)
    latency_ms = int((time.time() - started) * 1000)
    return QAResponse(answer=answer, contexts=contexts, latency_ms=latency_ms)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
