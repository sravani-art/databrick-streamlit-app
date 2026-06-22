# streamlit_app.py
import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8007")

st.set_page_config(page_title="RAG Process Maps UI")

st.title("RAG Agent — Process Map Uploader & Chat")

with st.expander("Upload files"):
    process_id = st.text_input("process_id (group files by process map)", value="proc-123")
    uploaded = st.file_uploader("Upload JSON", accept_multiple_files=True)
    if st.button("Upload"):
        if not uploaded:
            st.warning("Please select at least one file.")
        else:
            files = []
            for f in uploaded:
                files.append(("files", (f.name, f.getvalue(), f.type)))
            payload = {"process_id": process_id}
            headers = {}
            resp = requests.post(f"{API_BASE}/upload", files=files, data=payload, headers=headers)
            st.write(resp.json())

st.markdown("---")
st.header("Chat with a process map")
chat_proc = st.text_input("process_id to chat with", value="proc-123")
query = st.text_input("Your question about the process map")
top_k = st.slider("top_k retrievals", 1, 10, 5)
col1, col2 = st.columns([4, 1])
with col1:
    if st.button("Ask"):
        if not query:
            st.warning("Type a question")
        else:
            payload = {"process_id": chat_proc, "query": query, "top_k": top_k}
            headers = {"Content-Type": "application/json"}
            resp = requests.post(f"{API_BASE}/chat", json=payload, headers=headers)
            if resp.status_code == 200:
                j = resp.json()
                st.subheader("Answer")
                st.write(j.get("answer"))
                st.subheader("Sources")
                st.write(j.get("sources"))
            else:
                st.error(f"Error: {resp.status_code} {resp.text}")

with col2:
    if st.button("❌ Close Chat"):
        headers = {}
        resp = requests.delete(f"{API_BASE}/process/{chat_proc}", headers=headers)
        if resp.status_code == 200:
            st.success(f"Deleted data for process_id: {chat_proc}")
        else:
            st.error(f"Error deleting data: {resp.status_code} {resp.text}")
