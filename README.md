📄 Zero-Shot RAG System — Chat with Your Documents

A zero-shot Retrieval-Augmented Generation (RAG) system that allows users to upload documents and interact with them through a conversational AI interface.
The system supports PDFs, images, and text files, performs OCR when required, and provides grounded, context-aware answers without fine-tuning any language model.

🚀 Features

📁 Upload up to 5 documents per session

📄 Supports PDF, text files, and images (PNG/JPG)

🔍 OCR fallback for scanned PDFs and images

🧠 Zero-shot RAG (no model training or fine-tuning)

📐 Embedding-based semantic retrieval

💬 Chat interface with contextual, grounded answers

🧩 In-memory vector store for fast similarity search

🌐 demo-ready

📦 Clean, modular architecture

🧠 What is Zero-Shot RAG?

This system uses a pre-trained Large Language Model (LLM) and dynamically retrieves relevant document chunks at query time.
No fine-tuning or training is performed on user documents.

Why this matters:

Faster setup


🔄 Workflow

User uploads up to five documents

Text is extracted from PDFs or images (OCR when required)

Documents are split into overlapping chunks

Each chunk is converted into embeddings

Embeddings are stored in an in-memory vector store

User queries are embedded and matched against stored chunks

Retrieved context is injected into the LLM prompt

The model responds using only retrieved context

Lower cost

Reduced hallucinations

Scales easily to new documents

In-memory cosine similarity search
(Designed for demo and can be swapped with FAISS / Pinecone in production)

🛠️ Tech Stack
Frontend

⚙️ Backend
![Node.js](https://img.shields.io/badge/Node.js-339933?logo=nodedotjs&logoColor=white)
![API](https://img.shields.io/badge/API-Routes-blue)

🤖 AI / NLP
![Gemini](https://img.shields.io/badge/Gemini-LLM-blueviolet)
![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral-AI-orange)


🧠 RAG Concepts
![Embeddings](https://img.shields.io/badge/Embeddings-Semantic_Search-green)
![Zero--Shot](https://img.shields.io/badge/Zero--Shot-RAG-success)

📄 Document Processing
![PDF](https://img.shields.io/badge/PDF-Extraction-red?logo=adobeacrobatreader)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-blue)

📦 Vector Store
![Vector Store](https://img.shields.io/badge/Vector-Store-purple)
![Cosine Similarity](https://img.shields.io/badge/Cosine-Similarity-yellow)




