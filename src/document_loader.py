from pathlib import Path

from pypdf import PdfReader


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "medical_documents"


def load_documents(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict]:
    """Load extracted text from each PDF page with source metadata."""
    documents = []
    for pdf_file in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_file))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append({"text": text, "source": pdf_file.name, "page": page_number})
    return documents
