import streamlit as st
import os
import json
import shutil
from datetime import datetime
import pytz

# Direct imports for performance (No more subprocess)
from create_database import generate_data_store
from query_data import query_rag

st.set_page_config(page_title="Zero-Shot RAG Demo", layout="wide")

st.title("📄 Zero-Shot RAG System")
st.caption("Upload documents → Ask questions → Answers grounded in your data")

# ---------------- Session State & Persistence ----------------
CHAT_HISTORY_FILE = "chat_history.json"
DATA_DIR = "data/books"
CHROMA_PATH = "chroma"

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

def clear_data():
    # 1. Clear Data Files
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)
    
    # 2. Clear Vector DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    # 3. Clear Chat History
    st.session_state.chat_history = []
    save_chat_history([])
    
    st.toast("✅ Knowledge Base & History Cleared!")
    st.rerun()

# ---------------- Sidebar ----------------
st.sidebar.header("📤 Manage KnowledgeBase")

# Upload
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / Image",
    type=["pdf", "txt", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.sidebar.button("📥 Index Documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one document.")
    else:
        os.makedirs(DATA_DIR, exist_ok=True)

        for file in uploaded_files:
            with open(os.path.join(DATA_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())

        with st.spinner("Indexing documents..."):
            try:
                # Direct Function Call
                success = generate_data_store()
                if success:
                    st.sidebar.success("Documents indexed successfully!")
            except Exception as e:
                st.sidebar.error("❌ Indexing failed")
                st.code(str(e))

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Knowledge Base", type="primary"):
    clear_data()

# ---------------- Chat Section ----------------
st.subheader("💬 Chat with your documents")

query = st.text_input("Ask a question")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        try:
            # Direct Function Call
            answer, sources = query_rag(query)
            
            # IST Time
            ist = pytz.timezone('Asia/Kolkata')
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
    if chat.get('sources'):
         st.caption(f"Sources: {chat['sources']}")
    st.divider()
