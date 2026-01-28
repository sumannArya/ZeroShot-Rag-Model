import argparse
import re
from collections import defaultdict, Counter
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from prompts import (
    BOOK_PROMPT,
    LEGAL_PROMPT,
    IMAGE_PROMPT,
    COMPARE_PROMPT
)

load_dotenv()

CHROMA_PATH = "chroma"

# =========================
# QUERY INTENT HELPERS
# =========================

def is_comparison_query(query: str) -> bool:
    return bool(
        re.search(
            r"(compare|comparison|difference|differences|contrast|vs|versus)",
            query,
            re.I
        )
    )


def route_query(query: str) -> str:
    if re.search(r"(image|diagram|photo|screenshot|ocr|chart|table)", query, re.I):
        return "IMAGE"
    if re.search(
        r"(policy|shall|must|agreement|legal|compliance|clause|annex|schedule|exhibit)",
        query,
        re.I,
    ):
        return "LEGAL"
    return "BOOK"


# =========================
# MAIN RAG FUNCTION
# =========================

def query_rag(query_text: str):
    comparison_mode = is_comparison_query(query_text)
    initial_query_type = route_query(query_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    # -------------------------
    # HIGH-RECALL RETRIEVAL
    # -------------------------
    retrieval_k = 60 if comparison_mode else 10

    raw_results = db.similarity_search_with_relevance_scores(
        query_text,
        k=retrieval_k,
    )

    # -------------------------
    # EXPLICIT SOURCE FILTERING
    # -------------------------
    # If users ask for specific files (e.g. "legaldoc1"), semantic search might fail
    # if the filename isn't in the text. We force-fetch likely matches.
    if comparison_mode:
        try:
            # 1. Get all known sources (lightweight enough for this scale)
            # worst case: O(N) but N is small (number of docs)
            all_files = set()
            # Optimization: We don't have a cheap 'list sources' in Chroma 
            # so we only do this if we suspect a filename.
            # But let's cheat and look at what we found in raw_results to seed it, 
            # plus maybe a small query for *everything* metadata if needed?
            # Actually, let's just use what we have, OR:
            # We can run a small query per-token? No, too slow.
            
            # Better approach: We blindly check tokens against "source" metadata filter
            # But we need the EXACT filename for the filter.
            
            # Let's try to deduce filenames from the query tokens.
            tokens = re.findall(r"[\w\.-]+", query_text)
            potential_filenames = [t for t in tokens if len(t) > 3]

            extra_results = []
            for token in potential_filenames:
                # We can't do partial match easily in Chroma.
                # Attempt to find EXACT match for token + extensions
                for ext in ["", ".pdf", ".txt", ".md"]:
                    candidate = token + ext
                    # Search specifically for this source
                    # We use a dummy embedding search (empty string) constrained by metadata
                    # or just a "search" with the query text but forced filter
                    specific_docs = db.similarity_search_with_relevance_scores(
                        query_text,
                        k=2,
                        filter={"source": candidate}
                    )
                    if specific_docs:
                        extra_results.extend(specific_docs)
            
            if extra_results:
                raw_results.extend(extra_results)
        except Exception as e:
            print(f"Explicit source filter error: {e}")

    if not raw_results:
        return "NOT FOUND IN THE DOCUMENT", []

    # -------------------------
    # GROUP BY SOURCE (KEY FIX)
    # -------------------------
    grouped_by_source = defaultdict(list)
    for doc, score in raw_results:
        source = doc.metadata.get("source", "unknown")
        grouped_by_source[source].append((doc, score))

    # Always keep BEST chunk per document
    per_doc_results = [
        max(chunks, key=lambda x: x[1])
        for chunks in grouped_by_source.values()
    ]

    # -------------------------
    # DOMAIN DETECTION (ROBUST)
    # -------------------------
    doc_types = [
        doc.metadata.get("doc_type", initial_query_type.lower())
        for doc, _ in per_doc_results
    ]

    query_type = Counter(doc_types).most_common(1)[0][0].upper()

    # -------------------------
    # DOMAIN SETTINGS
    # -------------------------
    if query_type == "IMAGE":
        threshold = 0.18
        prompt_template = IMAGE_PROMPT
        fallback_answer = "NOT CLEAR FROM IMAGE TEXT"

    elif query_type == "LEGAL":
        threshold = 0.22
        prompt_template = LEGAL_PROMPT
        fallback_answer = "NO APPLICABLE CLAUSE FOUND"

    else:  # BOOK
        threshold = 0.20
        prompt_template = BOOK_PROMPT
        fallback_answer = "NOT FOUND IN THE DOCUMENT"

    # -------------------------
    # COMPARISON OVERRIDE
    # -------------------------
    if comparison_mode:
        threshold = 0.15
        prompt_template = COMPARE_PROMPT
        fallback_answer = "INSUFFICIENT INFORMATION FOR COMPARISON"

    # -------------------------
    # FILTER BY THRESHOLD
    # -------------------------
    # If we are in comparison mode, we might have added "low score" chunks explicitly.
    # We should trust chunks from forced sources.
    
    final_docs = []
    
    for doc, score in per_doc_results:
        # If it's a "high enough" score, keep it
        if score >= threshold:
            final_docs.append((doc, score))
        # Logic for explicit filenames: If the source name is ALMOST in the query, keep it even if low score
        elif comparison_mode:
             src_name = doc.metadata.get("source", "").lower()
             # precise check: if filename is in query, keep it regardless of score
             # (strip extension for checking)
             stem = Path(src_name).stem.lower()
             if stem in query_text.lower() or src_name in query_text.lower():
                 final_docs.append((doc, score))

    if not final_docs:
        return fallback_answer, []

    # -------------------------
    # COMPARISON SAFETY CHECK
    # -------------------------
    sources = {
        doc.metadata.get("source", "unknown")
        for doc, _ in final_docs
    }

    if comparison_mode and len(sources) < 2:
        return "INSUFFICIENT INFORMATION FOR COMPARISON", list(sources)

    # Limit context size (LLM-safe)
    final_docs.sort(key=lambda x: x[1], reverse=True)
    # Ensure we keep at least one form each source if possible
    # (Sorting might push all doc2 to bottom)
    
    # Optimization: Interleave if strictly comparing 2 docs? 
    # For now, just taking top 5 is usually fine if we have the right chunks.
    filtered = final_docs[:5]

    # -------------------------
    # CONTEXT CONSTRUCTION
    # -------------------------
    context_text = "\n\n---\n\n".join(
        f"[SOURCE: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc, _ in filtered
    )

    prompt = prompt_template.format(
        context=context_text,
        question=query_text,
    )

    # -------------------------
    # LLM CALL
    # -------------------------
    llm = ChatOllama(
        model="mistral",
        temperature=0.1,
        timeout=300,
    )

    response = llm.invoke(prompt)

    if not response or not response.content.strip():
        return fallback_answer, list(sources)

    return response.content.strip(), list(sources)


# =========================
# CLI ENTRY
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()

    answer, sources = query_rag(args.query_text)

    print("\nANSWER:\n", answer)
    print("\nSOURCES:\n", sources)


if __name__ == "__main__":
    main()
