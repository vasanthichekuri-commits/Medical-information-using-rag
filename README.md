# Medical Information RAG Assistant

A local retrieval-augmented generation demo built with Python, FAISS, Sentence Transformers, Ollama, PyPDF, and Gradio.

## Setup in VS Code

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install Ollama separately, then download the configured model:

```powershell
ollama pull llama3.2
```

Place trusted, legally usable medical PDFs from organizations such as WHO, CDC, NIH, or government health departments in `data/medical_documents/`. The application only answers from these documents.

Build the local index from the project root:

```powershell
python -m src.vector_store
```

Start Ollama and run the UI:

```powershell
ollama serve
python app.py
```

Open the Gradio URL shown in the terminal, usually `http://127.0.0.1:7860`.

## Tests

```powershell
pytest
```

## Project flow

PDF pages are extracted, split into overlapping chunks, embedded with `all-MiniLM-L6-v2`, and stored in a normalized FAISS index. Retrieval applies a similarity threshold before the selected chunks are passed to Ollama. The UI displays each supporting filename, page, and similarity score.

The threshold is intentionally configurable in `src/retriever.py`; tune it against an evaluation set rather than treating it as universal.
