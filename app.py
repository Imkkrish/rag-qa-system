try:
    import streamlit as st
    # Check if we are actually running with 'streamlit run'
    from streamlit.runtime.scriptrunner import get_script_run_context
    HAS_STREAMLIT = get_script_run_context() is not None
except ImportError:
    HAS_STREAMLIT = False

import os
import shutil
import uuid
from typing import List
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
import threading
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import schemas
from core_logic.models.schemas import QueryRequest, QueryResponse, UploadResponse

# Import services
from core_logic.core.config import settings
from core_logic.services.document_processor import DocumentProcessor
from core_logic.services.vector_store import vector_store
from core_logic.services.llm_service import llm_service

# --- FASTAPI APP DEFINITION (Exposed at top level) ---
limiter = Limiter(key_func=get_remote_address)
api_app = FastAPI(title="RAG API")
api_app.state.limiter = limiter
api_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
api_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@api_app.post("/upload", response_model=UploadResponse)
@limiter.limit(settings.RATE_LIMIT)
async def api_upload(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Basic extension check
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    file_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.UPLOADS_DIR, f"{file_id}_{file.filename}")
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    def process_task(path, name):
        try:
            text = DocumentProcessor.extract_text(path)
            chunks = DocumentProcessor.chunk_text(text)
            vector_store.add_documents(chunks, name)
        finally:
            if os.path.exists(path): 
                os.remove(path)
                
    background_tasks.add_task(process_task, temp_path, file.filename)
    return {"message": "Success", "filename": file.filename}

@api_app.post("/ask", response_model=QueryResponse)
@limiter.limit(settings.RATE_LIMIT)
async def api_ask(request: Request, query: QueryRequest):
    relevant_chunks = vector_store.search(query.question, k=query.k)
    if not relevant_chunks: 
        return {"answer": "I don't have enough information in the documents to answer that.", "sources": []}
    
    answer = llm_service.generate_answer(query.question, relevant_chunks)
    sources = list(set([c['source'] for c in relevant_chunks]))
    return {"answer": answer, "sources": sources}

# --- BACKGROUND API RUNNER ---
def run_api_server():
    # Run on a different port than Streamlit (7860)
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

# Start API thread automatically if in Streamlit
if HAS_STREAMLIT:
    if "api_started" not in st.session_state:
        st.session_state.api_started = True
        threading.Thread(target=run_api_server, daemon=True).start()

# --- STREAMLIT UI ---
if HAS_STREAMLIT:
    st.set_page_config(
        page_title="Knowledge Nexus",
        page_icon="📚",
        layout="wide"
    )
    
    # Premium Text-Only CSS
    st.markdown("""
    <style>
        .main { background-color: #f8fafc; }
        .stChatFloatingInputContainer { bottom: 20px; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    if HAS_SPACES:
        @spaces.GPU
        def get_llm_response(query: str, chunks: List):
            return llm_service.generate_answer(query, chunks)
    else:
        def get_llm_response(query: str, chunks: List):
            return llm_service.generate_answer(query, chunks)

    # Sidebar
    with st.sidebar:
        st.title("Knowledge Nexus")
        st.markdown("---")
        st.header("Document Upload")
        st.write("Upload PDF/TXT to link them to the AI.")
        
        uploaded_file = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed")
        
        if uploaded_file and st.button("Index Document"):
            with st.spinner("Processing..."):
                file_id = str(uuid.uuid4())
                temp_path = os.path.join(settings.UPLOADS_DIR, f"{file_id}_{uploaded_file.name}")
                os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    text = DocumentProcessor.extract_text(temp_path)
                    chunks = DocumentProcessor.chunk_text(text)
                    vector_store.add_documents(chunks, uploaded_file.name)
                    st.success(f"Linked: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    if os.path.exists(temp_path): os.remove(temp_path)

        st.markdown("---")
        st.write("Model: Gemini 1.5 Flash")
        st.write("Retrieval: FAISS")
        st.write("Compute: ZeroGPU")

    # Chat UI
    st.title("Document Interaction")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                chunks = vector_store.search(prompt)
                if not chunks:
                    response = "No relevant information found."
                else:
                    ans = get_llm_response(prompt, chunks)
                    sources = list(set([c['source'] for c in chunks]))
                    response = f"{ans}\n\nSources: {', '.join(sources)}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    if not HAS_STREAMLIT:
        print("Starting FastAPI Server (Independent Mode)...")
        run_api_server()
    else:
        print("FastAPI Server will start in a background thread via Streamlit.")
