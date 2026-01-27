import argparse
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context.

Context:
{context}

---

Question: {question}

Answer:
"""

SIMILARITY_THRESHOLD = 0.25  # tune if needed


def query_rag(query_text: str):
    # Embeddings (local, free)
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load Chroma DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search
    results = db.similarity_search_with_relevance_scores(query_text, k=7)

    if not results:
        return "❌ No relevant context found in the documents.", []

    top_score = results[0][1]

    if top_score < SIMILARITY_THRESHOLD:
        return (
            "❌ The answer is not explicitly present in the provided documents "
            f"(similarity score: {top_score:.2f})."
        ), []

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _ in results[:3]]
    )

    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        question=query_text
    )

    # 🔥 OLLAMA LLM (LOCAL)
    llm = ChatOllama(
        model="mistral",
        temperature=0.2,
        timeout=120,
    )

    response = llm.invoke(prompt)
    
    if not response or not response.content or not response.content.strip():
        return "❌ Model returned empty response.", []

    sources = list(
        {doc.metadata.get("source", "unknown") for doc, _ in results}
    )
    
    return response.content, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    answer, sources = query_rag(query_text)
    print(answer)
    print(sources)


if __name__ == "__main__":
    main()
