import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

FALLBACK = "I could not find enough information in the available medical knowledge base to answer this question."


def generate_answer(question: str, retrieved_documents: list[dict], ollama_url: str = OLLAMA_URL, model: str = MODEL) -> str:
    if not retrieved_documents:
        return FALLBACK

    context = "\n\n".join(
        f"Source: {doc['source']}\nPage: {doc['page']}\n{doc['text']}"
        for doc in retrieved_documents
    )
    prompt = f"""You are a Medical Information RAG Assistant.
Answer using ONLY the supplied CONTEXT. Do not invent facts, diagnose, prescribe medication, or recommend personalized treatment.
If the context does not contain enough information, respond exactly: {FALLBACK}
Keep the answer educational and concise. For personal medical concerns, recommend consulting a qualified healthcare professional.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""
    response = requests.post(ollama_url, json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    response.raise_for_status()
    return response.json()["response"].strip()
