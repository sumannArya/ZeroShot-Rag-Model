from langchain_community.document_loaders import UnstructuredPDFLoader
import os
import time

# Helper to time it
start = time.time()

file_path = "data/books/SumanArya's resumeee.pdf"
print(f"Testing UnstructuredPDFLoader on {file_path}...")

try:
    loader = UnstructuredPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"First doc content length: {len(docs[0].page_content)}")
    print(f"Time taken: {time.time() - start:.2f}s")
    print("SUCCESS")
except Exception as e:
    print("FAILED")
    print(e)
