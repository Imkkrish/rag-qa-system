import time
import os
import requests
import streamlit as st

st.set_page_config(page_title="RAG QA", page_icon="🧠", layout="wide")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.title("RAG-Based Question Answering")

with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded is not None:
        if st.button("Ingest Document"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            res = requests.post(f"{API_BASE}/documents/upload", files=files)
            if res.ok:
                data = res.json()
                st.success(f"Job queued: {data['job_id']}")
                st.session_state["last_job_id"] = data["job_id"]
            else:
                st.error(res.text)

    if "last_job_id" in st.session_state:
        if st.button("Check Last Job"):
            jid = st.session_state["last_job_id"]
            res = requests.get(f"{API_BASE}/documents/status/{jid}")
            if res.ok:
                st.json(res.json())
            else:
                st.error(res.text)

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
            res = requests.post(f"{API_BASE}/qa", json=payload)
            if res.ok:
                data = res.json()
                st.markdown("### Answer")
                st.write(data["answer"])
                st.caption(f"Latency: {data['latency_ms']} ms")
                st.markdown("### Retrieved Contexts")
                for ctx in data["contexts"]:
                    st.markdown(f"**{ctx['source']}** (score: {ctx['score']:.3f})")
                    st.write(ctx["chunk"])
            else:
                st.error(res.text)
