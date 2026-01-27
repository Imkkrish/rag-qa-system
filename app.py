import gradio as gr
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import spaces

# Import our existing services
from app.core.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service

# --- FASTAPI SETUP ---
fastapi_app = FastAPI(title="RAG API")
fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@spaces.GPU
def generate_answer_with_gpu(question, chunks):
    return llm_service.generate_answer(question, chunks)

@fastapi_app.post("/upload")
async def api_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT supported.")
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
    return {"message": "Processing in background", "filename": file.filename}

@fastapi_app.post("/ask")
async def api_ask(request: Request):
    data = await request.json()
    question = data.get("question")
    relevant_chunks = vector_store.search(question)
    if not relevant_chunks: return {"answer": "No relevant info found.", "sources": []}
    answer = generate_answer_with_gpu(question, relevant_chunks)
    sources = list(set([c['source'] for c in relevant_chunks]))
    return {"answer": answer, "sources": sources}

# --- GRADIO UI SETUP ---
def chatbot_response(message, history):
    relevant_chunks = vector_store.search(message)
    if not relevant_chunks:
        yield "I couldn't find any relevant information in the uploaded documents."
        return
    answer = generate_answer_with_gpu(message, relevant_chunks)
    sources = list(set([c['source'] for c in relevant_chunks]))
    response = f"{answer}\n\n**Sources:** {', '.join(sources)}"
    yield response

def upload_file(file):
    if file is None: return "No file selected."
    try:
        text = DocumentProcessor.extract_text(file.name)
        chunks = DocumentProcessor.chunk_text(text)
        vector_store.add_documents(chunks, os.path.basename(file.name))
        return f"✅ Linked: {os.path.basename(file.name)}"
    except Exception as e: return f"❌ Error: {str(e)}"

# Premium "Modern Glass" CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
}

.gradio-container {
    background: #f8fafc !important;
    font-family: 'Outfit', sans-serif !important;
}

.main-header {
    background: var(--primary-gradient);
    padding: 2.5rem;
    border-radius: 24px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
}

.main-header h1 {
    font-weight: 600;
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    color: white !important;
}

.sidebar-box {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.chat-container {
    border-radius: 24px !important;
    border: 1px solid #e2e8f0 !important;
    background: white !important;
    overflow: hidden;
}

.upload-btn {
    background: #f1f5f9 !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    color: #475569 !important;
    transition: all 0.3s ease !important;
}

.upload-btn:hover {
    border-color: #6366f1 !important;
    background: #eef2ff !important;
    color: #4f46e5 !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", radius_size="lg"), css=custom_css) as demo:
    with gr.Group(elem_classes="main-header"):
        gr.Markdown("# 📚 Knowledge Nexus")
        gr.Markdown("Unlock the hidden insights in your documents with AI-powered retrieval.")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="sidebar-box"):
                gr.Markdown("### 📥 Ingest Documents")
                gr.Markdown("Support for PDF and TXT files. Your data remains private.")
                file_status = gr.Textbox(label="Last Action", interactive=False, placeholder="Waiting for upload...")
                upload_btn = gr.UploadButton(
                    "➕ Add Document", 
                    file_types=[".pdf", ".txt"],
                    elem_classes="upload-btn"
                )
                upload_btn.upload(upload_file, upload_btn, file_status)
                
                gr.Markdown("---")
                gr.Markdown("### 🛠️ Configuration")
                gr.Markdown("Model: **Gemini 1.5 Flash**\nEngine: **FAISS Vector Core**\nScaling: **ZeroGPU Optimized**")
            
        with gr.Column(scale=2):
            with gr.Group(elem_classes="chat-container"):
                gr.ChatInterface(
                    fn=chatbot_response,
                    type="messages",
                    examples=["What's in the document?", "Give me a summary."],
                    fill_height=True
                )

# API Prefixing
app = FastAPI()
app.mount("/api", fastapi_app)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
