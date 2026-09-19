from .chunker import chunk_text
from .document_loader import DEFAULT_DATA_DIR, load_documents


def prepare_documents(data_dir=DEFAULT_DATA_DIR, chunk_size=800, overlap=150) -> list[dict]:
    chunks = []
    for page in load_documents(data_dir):
        for text in chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap):
            chunks.append({"text": text, "source": page["source"], "page": page["page"]})
    return chunks
