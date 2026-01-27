from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def main():
    # Free, local embedding model
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Get embedding for a word
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple' (first 10 values): {vector[:10]}")
    print(f"Vector length: {len(vector)}")

    # Compare embeddings of two words
    word1 = "apple"
    word2 = "iphone"

    vec1 = embedding_function.embed_query(word1)
    vec2 = embedding_function.embed_query(word2)

    similarity = cosine_similarity(vec1, vec2)

    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")


if __name__ == "__main__":
    main()
