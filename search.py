import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "recall_documents.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "data/recall_documents.json not found. Run python prepare_data.py first."
        )

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not documents:
        raise ValueError("No recall documents available.")

    return documents


def build_search_index(documents, model):
    texts = [document["text"] for document in documents]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, texts


def search(query, documents, model, index, top_k=3):
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_vector = np.asarray(query_vector, dtype="float32")

    k = min(top_k, len(documents))
    scores, positions = index.search(query_vector, k)

    print(f"\nQuestion: {query}")
    print("\nMost relevant recall records:")

    for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1):
        document = documents[int(position)]

        print(f"\n--- Result {rank} ---")
        print(f"Similarity score: {float(score):.3f}")
        print(f"Campaign: {document['id']}")
        print(f"Source: {document['source']}")
        print(document["text"])


def main():
    print("Loading recall documents...")
    documents = load_documents()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Building search index...")
    index, _ = build_search_index(documents, model)

    print(f"Ready! Indexed {len(documents)} recall documents.")

    while True:
        query = input("\nAsk a vehicle recall question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        search(query, documents, model, index)


if __name__ == "__main__":
    main()
