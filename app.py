import streamlit as st
import os
import shutil
import uuid
from typing import List
import spaces
import threading
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import time

# Import services
from app.core.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service

# --- API THREAD ---
def run_api():
    api_app = FastAPI(title="RAG API")
    api_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @api_app.post("/upload")
    async def api_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        file_id = str(uuid.uuid4())
        temp_path = os.path.join(settings.UPLOADS_DIR, f"{file_id}_{file.filename}")
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        def process_task(path, name):
            text = DocumentProcessor.extract_text(path)
            chunks = DocumentProcessor.chunk_text(text)
            vector_store.add_documents(chunks, name)
            if os.path.exists(path): os.remove(path)
        background_tasks.add_task(process_task, temp_path, file.filename)
        return {"message": "Success", "filename": file.filename}

    @api_app.post("/ask")
    async def api_ask(request: Request):
        data = await request.json()
        question = data.get("question")
        relevant_chunks = vector_store.search(question)
        if not relevant_chunks: return {"answer": "No info", "sources": []}
        # In API context, ZeroGPU might be tricky from thread, 
        # but for internal use it should work if called correctly.
        answer = llm_service.generate_answer(question, relevant_chunks)
        sources = list(set([c['source'] for c in relevant_chunks]))
        return {"answer": answer, "sources": sources}

    uvicorn.run(api_app, host="0.0.0.0", port=8000)

# Start API in background if not already running
if "api_thread" not in st.session_state:
    thread = threading.Thread(target=run_api, daemon=True)
    thread.start()
    st.session_state.api_thread = True

# Page Configuration
st.set_page_config(
    page_title="Knowledge Nexus | RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for Premium Look (No Icons)
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4f46e5;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        color: white;
    }
    .upload-box {
        border: 2px dashed #cbd5e1;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .status-text {
        color: #64748b;
        font-size: 0.9em;
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Zero-GPU Decorated Function
@spaces.GPU
def get_llm_response(query: str, chunks: List):
    return llm_service.generate_answer(query, chunks)

# Sidebar for Ingestion
with st.sidebar:
    st.title("Knowledge Nexus")
    st.markdown("---")
    st.header("Upload Section")
    st.write("Link your PDF or TXT documents to the system brain.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Analyzing and indexing..."):
                # Save temporarily
                file_id = str(uuid.uuid4())
                temp_path = os.path.join(settings.UPLOADS_DIR, f"{file_id}_{uploaded_file.name}")
                os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    text = DocumentProcessor.extract_text(temp_path)
                    chunks = DocumentProcessor.chunk_text(text)
                    vector_store.add_documents(chunks, uploaded_file.name)
                    st.success(f"Successfully linked: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing: {str(e)}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    st.markdown("---")
    st.header("System Status")
    st.info("Model: Gemini 1.5 Flash\nRetrieval: FAISS\nCompute: ZeroGPU")

# Main Chat Interface
st.title("Document Interaction")
st.write("Ask questions based on your uploaded knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Enter your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            relevant_chunks = vector_store.search(prompt)
            
            if not relevant_chunks:
                response = "I could not find any relevant information in the linked documents."
            else:
                response_text = get_llm_response(prompt, relevant_chunks)
                sources = list(set([c['source'] for c in relevant_chunks]))
                response = f"{response_text}\n\nSources: {', '.join(sources)}"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
