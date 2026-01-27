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

# Zero-GPU Decorated Function
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
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    
    relevant_chunks = vector_store.search(question)
    if not relevant_chunks:
        return {"answer": "No relevant info found.", "sources": []}
    
    answer = generate_answer_with_gpu(question, relevant_chunks)
    sources = list(set([c['source'] for c in relevant_chunks]))
    return {"answer": answer, "sources": sources}

# --- GRADIO UI SETUP ---
def chatbot_response(message, history):
    # message: str, history: list of messages
    relevant_chunks = vector_store.search(message)
    if not relevant_chunks:
        yield "I couldn't find any relevant information in the uploaded documents."
        return
    
    answer = generate_answer_with_gpu(message, relevant_chunks)
    sources = list(set([c['source'] for c in relevant_chunks]))
    
    response = f"{answer}\n\n**Sources:** {', '.join(sources)}"
    yield response

def upload_file(file):
    if file is None:
        return "No file selected."
    try:
        text = DocumentProcessor.extract_text(file.name)
        chunks = DocumentProcessor.chunk_text(text)
        vector_store.add_documents(chunks, os.path.basename(file.name))
        return f"✅ Successfully processed: {os.path.basename(file.name)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Premium CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.sidebar-info {
    font-size: 0.9em;
    color: #666;
    margin-top: 10px;
}
.header-container {
    text-align: center;
    margin-bottom: 20px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"), css=custom_css) as demo:
    with gr.Row(elem_classes="header-container"):
        gr.Markdown("# 📚 RAG-Based Knowledge Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Document Upload")
            gr.Markdown("Upload your PDFs or TXT files here to build your knowledge base.", elem_classes="sidebar-info")
            file_output = gr.Textbox(label="Status", interactive=False)
            upload_button = gr.UploadButton("📁 Select File", file_types=[".pdf", ".txt"])
            upload_button.upload(upload_file, upload_button, file_output)
            
            gr.Markdown("---")
            gr.Markdown("### 💡 Tips")
            gr.Markdown("- Ask specific questions about your docs.\n- Upload multiple files for a broader context.\n- Responses are generated using Gemini 1.5 Flash.")
            
        with gr.Column(scale=3):
            # Using ChatInterface with modernized types
            chat = gr.ChatInterface(
                fn=chatbot_response,
                type="messages",
                examples=["What is the main topic of the document?", "Summarize the uploaded file."],
            )

# Mount Gradio at root and FastAPI at /api
# This helps avoid routing conflicts
app = FastAPI()
app.mount("/api", fastapi_app)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
