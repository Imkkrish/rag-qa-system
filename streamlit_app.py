import time
import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from app.config import UPLOAD_DIR
from app.rag import ingest_document, search, generate_answer

st.set_page_config(page_title="RAG QA", page_icon="🧠", layout="wide")

st.title("RAG-Based Question Answering")

with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded is not None:
        if st.button("Ingest Document"):
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            suffix = Path(uploaded.name).suffix.lower()
            doc_id = str(uuid4())
            dest_path = UPLOAD_DIR / f"{doc_id}{suffix}"
            dest_path.write_bytes(uploaded.getvalue())
            ingest_document(dest_path, doc_id=doc_id, source=uploaded.name)
            st.success("Document ingested locally.")

st.subheader("Ask a Question")
question = st.text_input("Question")
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Top K", 1, 10, 4)
with col2:
    if st.button("Ask"):
        if not question:
            st.warning("Enter a question")
        else:
            payload = {"question": question, "top_k": top_k}
            start = time.time()
            contexts = search(question, top_k)
            answer = generate_answer(question, contexts)
            latency_ms = int((time.time() - start) * 1000)
            data = {"answer": answer, "contexts": contexts, "latency_ms": latency_ms}

            st.markdown("### Answer")
            st.write(data["answer"])
            st.caption(f"Latency: {data['latency_ms']} ms")
            st.markdown("### Retrieved Contexts")
            for ctx in data["contexts"]:
                st.markdown(f"**{ctx['source']}** (score: {ctx['score']:.3f})")
                st.write(ctx["chunk"])
