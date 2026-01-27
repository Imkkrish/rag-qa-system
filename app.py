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
        try:
            text = DocumentProcessor.extract_text(path)
            chunks = DocumentProcessor.chunk_text(text)
            vector_store.add_documents(chunks, name)
        finally:
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
        yield "I could not find any relevant information in the uploaded documents."
        return
    answer = generate_answer_with_gpu(message, relevant_chunks)
    sources = list(set([c['source'] for c in relevant_chunks]))
    response = f"{answer}\n\nSources: {', '.join(sources)}"
    yield response

def upload_file(file):
    if file is None: return "No file selected."
    try:
        text = DocumentProcessor.extract_text(file.name)
        chunks = DocumentProcessor.chunk_text(text)
        vector_store.add_documents(chunks, os.path.basename(file.name))
        return f"File linked: {os.path.basename(file.name)}"
    except Exception as e: return f"Error: {str(e)}"

# Minimal CSS to avoid layout breakages
# Explicitly hiding SVG icons that might be causing the massive zoom issue
custom_css = """
footer {display: none !important;}
.gradio-container {max-width: 1000px !important; margin: 0 auto !important;}
button svg {display: none !important;}
.header-area {
    text-align: center;
    padding: 20px;
    background: #4f46e5;
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo"), css=custom_css) as demo:
    with gr.Column(elem_classes="header-area"):
        gr.Markdown("# Knowledge Nexus")
        gr.Markdown("RAG-Powered Document Q&A System")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Section")
            file_status = gr.Textbox(label="Status", interactive=False, placeholder="No documents uploaded")
            upload_btn = gr.UploadButton("Upload Document", file_types=[".pdf", ".txt"])
            upload_btn.upload(upload_file, upload_btn, file_status)
            
            gr.Markdown("---")
            gr.Markdown("### System Info")
            gr.Markdown("Model: Gemini 1.5 Flash\nRetrieval: FAISS\nCompute: ZeroGPU")
            
        with gr.Column(scale=2):
            gr.ChatInterface(
                fn=chatbot_response,
                type="messages",
                examples=["Summarize the document", "Key points of the file"],
                fill_height=True
            )

# API Prefixing
app = FastAPI()
app.mount("/api", fastapi_app)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
