import pickle
from pathlib import Path

import faiss

from .embeddings import create_embeddings
from .prepare_documents import prepare_documents

DEFAULT_VECTOR_DIR = Path(__file__).resolve().parents[1] / "vectorstore"


def build_vector_store(data_dir=None, vector_dir=DEFAULT_VECTOR_DIR) -> int:
    documents = prepare_documents(data_dir) if data_dir else prepare_documents()
    if not documents:
        raise ValueError("No extractable PDF text found in data/medical_documents.")

    embeddings = create_embeddings([doc["text"] for doc in documents]).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    vector_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(vector_dir / "medical.index"))
    with (vector_dir / "documents.pkl").open("wb") as handle:
        pickle.dump(documents, handle)
    return index.ntotal


if __name__ == "__main__":
    print(f"Indexed {build_vector_store()} chunks.")
