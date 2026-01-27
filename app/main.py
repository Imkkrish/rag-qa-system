from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import shutil
import os
import uuid

from app.core.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service

# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG-Based QA System")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

def process_document_task(file_path: str, original_filename: str):
    try:
        text = DocumentProcessor.extract_text(file_path)
        chunks = DocumentProcessor.chunk_text(text)
        vector_store.add_documents(chunks, original_filename)
    except Exception as e:
        print(f"Error processing document: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload")
@limiter.limit(settings.RATE_LIMIT)
async def upload_document(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.UPLOADS_DIR, f"{file_id}_{file.filename}")
    
    # Ensure directory exists
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    background_tasks.add_task(process_document_task, temp_path, file.filename)
    
    return {"message": "File uploaded successfully. Processing in background.", "filename": file.filename}

@app.post("/ask", response_model=QueryResponse)
@limiter.limit(settings.RATE_LIMIT)
async def ask_question(request: Request, query_req: QueryRequest):
    relevant_chunks = vector_store.search(query_req.question)
    
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="No relevant context found in documents.")
    
    answer = llm_service.generate_answer(query_req.question, relevant_chunks)
    sources = list(set([chunk['source'] for chunk in relevant_chunks]))
    
    return QueryResponse(answer=answer, sources=sources)

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG-Based QA System API"}

if __name__ == "__main__":
    import uvicorn
    # Use port assigned by HF or default to 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
