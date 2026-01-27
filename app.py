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

@fastapi_app.post("/upload")
async def api_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT supported.")
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.UPLOADS_DIR, f"{file_id}_{file.filename}")
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Internal function for background processing
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
    if not relevant_chunks:
        return {"answer": "No relevant info found.", "sources": []}
    
    answer = generate_answer_with_gpu(question, relevant_chunks)
    sources = list(set([c['source'] for c in relevant_chunks]))
    return {"answer": answer, "sources": sources}

# --- ZERO-GPU DECORATED FUNCTION ---
@spaces.GPU
def generate_answer_with_gpu(question, chunks):
    return llm_service.generate_answer(question, chunks)

# --- GRADIO UI SETUP ---
def chatbot_response(message, history):
    relevant_chunks = vector_store.search(message)
    if not relevant_chunks:
        return "I couldn't find any relevant information in the uploaded documents."
    
    answer = generate_answer_with_gpu(message, relevant_chunks)
    sources = list(set([c['source'] for c in relevant_chunks]))
    return f"{answer}\n\n**Sources:** {', '.join(sources)}"

def upload_file(file):
    if file is None:
        return "No file selected."
    
    # Process immediately for the UI
    text = DocumentProcessor.extract_text(file.name)
    chunks = DocumentProcessor.chunk_text(text)
    vector_store.add_documents(chunks, os.path.basename(file.name))
    return f"Successfully processed: {os.path.basename(file.name)}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 📚 RAG-Based Question Answering System")
    gr.Markdown("Upload documents (PDF/TXT) and ask questions. Powered by **Gemini 1.5 Flash** and **ZeroGPU**.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_output = gr.Textbox(label="Upload Status")
            upload_button = gr.UploadButton("Click to Upload PDF/TXT", file_types=[".pdf", ".txt"])
            upload_button.upload(upload_file, upload_button, file_output)
            
        with gr.Column(scale=2):
            chat = gr.ChatInterface(chatbot_response)

# --- MOUNTING AND LAUNCHING ---
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
