from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system", min_length=1)
    k: Optional[int] = Field(4, description="Number of relevant chunks to retrieve", ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class UploadResponse(BaseModel):
    message: str
    filename: str
