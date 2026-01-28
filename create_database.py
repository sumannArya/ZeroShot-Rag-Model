import os
import shutil
import pytesseract
from pathlib import Path
from PIL import Image

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# OCR CONFIG
# =========================
os.environ["OCR_AGENT"] = "tesseract"
os.environ["TESSERACT_CMD"] = r"S:\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"S:\Tesseract-OCR\tesseract.exe"

# =========================
# PATHS
# =========================
CHROMA_PATH = "chroma"
DATA_PATH = "data"

BOOK_PATH = "books"
LEGAL_PATH = "legal"
IMAGE_PATH = "images"

# =========================
# MAIN
# =========================

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    return True


# =========================
# LOAD DOCUMENTS
# =========================

def load_documents():
    print(f"Loading documents from {DATA_PATH}")
    documents = []

    # ---------------- BOOKS ----------------
    documents += DirectoryLoader(
        os.path.join(DATA_PATH, BOOK_PATH),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    ).load()

    documents += DirectoryLoader(
        os.path.join(DATA_PATH, BOOK_PATH),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()

    documents += DirectoryLoader(
        os.path.join(DATA_PATH, BOOK_PATH),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()

    # ---------------- LEGAL ----------------
    legal_dir = os.path.join(DATA_PATH, LEGAL_PATH)
    if os.path.exists(legal_dir):
        for ext in ("*.txt", "*.md"):
            legal_docs = DirectoryLoader(
                legal_dir,
                glob=f"**/{ext}",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
            ).load()

            for doc in legal_docs:
                doc.metadata["doc_type"] = "legal"

            documents.extend(legal_docs)

        for file in os.listdir(legal_dir):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(legal_dir, file)
                try:
                    pdf_docs = PyPDFLoader(pdf_path).load()
                    for d in pdf_docs:
                        d.metadata["doc_type"] = "legal"
                        d.metadata["source"] = file
                    documents.extend(pdf_docs)
                except Exception:
                    print(f"Skipping unreadable legal PDF: {file}")

    # ---------------- IMAGES (DIRECT OCR) ----------------
    image_dir = os.path.join(DATA_PATH, IMAGE_PATH)
    if os.path.exists(image_dir):
        for img_path in Path(image_dir).rglob("*"):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                try:
                    text = pytesseract.image_to_string(Image.open(img_path))
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": img_path.name,
                                    "doc_type": "image"
                                }
                            )
                        )
                except Exception as e:
                    print(f"Skipping image {img_path.name}: {e}")

    # ---------------- METADATA NORMALIZATION ----------------
    for doc in documents:
        source_path = Path(doc.metadata.get("source", "unknown"))
        doc.metadata["source"] = source_path.name
        doc.metadata.setdefault("doc_type", "book")

    print(f"Loaded {len(documents)} documents total")
    return documents


# =========================
# SPLIT DOCUMENTS
# =========================

def split_text(documents):
    print("Splitting documents...")
    chunks = []

    for doc in documents:
        doc_type = doc.metadata.get("doc_type", "book")

        if doc_type == "book":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=150
            )
        elif doc_type == "legal":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=250
            )
        else:  # image
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50
            )

        doc_chunks = splitter.split_documents([doc])
        for chunk in doc_chunks:
            chunk.metadata.update(doc.metadata)

        chunks.extend(doc_chunks)

    print(f"Created {len(chunks)} chunks")
    return chunks


# =========================
# SAVE TO CHROMA
# =========================

def save_to_chroma(chunks):
    print("Saving to Chroma...")

    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError:
            import gc, time
            gc.collect()
            time.sleep(1)
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)

    valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    if not valid_chunks:
        raise ValueError("No valid chunks to store.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    Chroma.from_documents(
        valid_chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
    )

    print("Chroma DB ready")


if __name__ == "__main__":
    main()
