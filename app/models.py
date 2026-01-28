from pydantic import BaseModel, Field
from typing import List, Optional


class UploadResponse(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    id: str
    filename: str
    status: str
    error: Optional[str]


class QARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(4, ge=1, le=10)


class ContextChunk(BaseModel):
    score: float
    source: str
    doc_id: str
    chunk: str


class QAResponse(BaseModel):
    answer: str
    contexts: List[ContextChunk]
    latency_ms: int
