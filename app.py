import streamlit as st
import os
import json
import shutil
from datetime import datetime
import pytz
from pathlib import Path
import time
import gc

# Direct imports
from create_database import generate_data_store
from query_data import query_rag

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Zero-Shot RAG Demo", layout="wide")

DATA_ROOT = "data"
BOOKS_DIR = os.path.join(DATA_ROOT, "books")
LEGAL_DIR = os.path.join(DATA_ROOT, "legal")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
CHROMA_PATH = "chroma"
CHAT_HISTORY_FILE = "chat_history.json"

# Ensure folders exist
for d in [BOOKS_DIR, LEGAL_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- UI ----------------
st.title("📄 Zero-Shot RAG System")
st.caption("Upload documents → Ask questions → Answers grounded in your data")

# ---------------- Chat History ----------------
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

# ---------------- File Routing Logic ----------------
def route_and_save_file(uploaded_file):
    name = uploaded_file.name.lower()
    suffix = Path(name).suffix

    # Images
    if suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"]:
        target_dir = IMAGES_DIR

    # Legal docs (filename heuristic)
    elif any(k in name for k in [
         "policy",
         "policies",
         "terms",
         "conditions",
         "agreement",
         "contract",
         "nda",
         "privacy",
         "compliance",
         "license",
         "licence",
         "gdpr",
         "regulation",
         "bylaws",
         "governance",
         "legal",
         "disclaimer"
    ]):
        target_dir = LEGAL_DIR

    # Default → books
    else:
        target_dir = BOOKS_DIR

    save_path = os.path.join(target_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path

# ---------------- Clear Knowledge Base ----------------
def clear_data():
    # Clear data folders
    for d in [BOOKS_DIR, LEGAL_DIR, IMAGES_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    # Try to safely clear Chroma (Windows-safe)
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError:
            gc.collect()
            time.sleep(1)
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)

    # Clear chat history
    st.session_state.chat_history = []
    save_chat_history([])

    st.toast("✅ Knowledge Base & History Cleared")
    st.rerun()

# ---------------- Sidebar ----------------
st.sidebar.header("📤 Manage Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.sidebar.button("📥 Index Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one document.")
    else:
        saved_files = []
        for file in uploaded_files:
            path = route_and_save_file(file)
            saved_files.append(path)

        with st.spinner("Indexing documents..."):
            try:
                generate_data_store()
                st.sidebar.success("✅ Documents indexed successfully!")
                st.sidebar.caption("Saved files:")
                for f in saved_files:
                    st.sidebar.caption(f"• {f}")
            except Exception as e:
                st.sidebar.error("❌ Indexing failed")
                st.code(str(e))

st.sidebar.markdown("---")

if st.sidebar.button("🗑️ Clear Knowledge Base", type="primary"):
    clear_data()

# ---------------- Chat ----------------
st.subheader("💬 Chat with your documents")

query = st.text_input("Ask a question")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        try:
            answer, sources = query_rag(query)

            ist = pytz.timezone("Asia/Kolkata")
            timestamp = datetime.now(ist).strftime("%H:%M:%S")

            st.session_state.chat_history.append({
                "time": timestamp,
                "question": query,
                "answer": answer,
                "sources": sources
            })

            save_chat_history(st.session_state.chat_history)

        except Exception as e:
            st.error("❌ Query failed")
            st.code(str(e))

# ---------------- Display Chat History ----------------
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"🕒 **{chat['time']}**")
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Assistant:** {chat['answer']}")
    if chat.get("sources"):
        st.caption(f"Sources: {chat['sources']}")
    st.divider()
