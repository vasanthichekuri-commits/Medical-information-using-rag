import pickle
from pathlib import Path

import faiss

from .embeddings import create_embeddings
from .vector_store import DEFAULT_VECTOR_DIR

DISTANCE_THRESHOLD = 0.25


class Retriever:
    def __init__(self, vector_dir: Path = DEFAULT_VECTOR_DIR, threshold: float = DISTANCE_THRESHOLD):
        index_path = vector_dir / "medical.index"
        documents_path = vector_dir / "documents.pkl"
        if not index_path.exists() or not documents_path.exists():
            raise FileNotFoundError("Vector store is missing. Run `python -m src.vector_store` first.")
        self.index = faiss.read_index(str(index_path))
        with documents_path.open("rb") as handle:
            self.documents = pickle.load(handle)
        self.threshold = threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        query_embedding = create_embeddings([query]).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, index_number in zip(scores[0], indices[0]):
            if index_number < 0 or score < self.threshold:
                continue
            document = self.documents[index_number].copy()
            document["score"] = float(score)
            results.append(document)
        return results
