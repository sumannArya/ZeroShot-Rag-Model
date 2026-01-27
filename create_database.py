import os
import shutil
import pytesseract

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredImageLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔥 HARD-CODE TESSERACT PATH (Windows, S drive)
pytesseract.pytesseract.tesseract_cmd = r"S:\Tesseract-OCR\tesseract.exe"

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    return True


def load_documents():
    print(f"Loading documents from {DATA_PATH}")
    documents = []

    # TXT / MD
    documents += DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()

    documents += DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()

    # PDFs (TEXT-BASED ONLY)
    documents += DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    ).load()

    # Images (OCR handled separately)
    documents += DirectoryLoader(
        DATA_PATH,
        glob="**/*.[pjP][pnPN][gG]",
        loader_cls=UnstructuredImageLoader,
    ).load()

    print(f"Loaded {len(documents)} documents total")
    return documents


def split_text(documents):
    print(" Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=550,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(documents)
    print(f" Created {len(chunks)} chunks")
    return chunks


def save_to_chroma(chunks):
    print(" Saving to Chroma...")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(" Chroma DB ready")


if __name__ == "__main__":
    main()
