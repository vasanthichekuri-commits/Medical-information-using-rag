from .generator import generate_answer
from .retriever import Retriever


def answer_question(question: str, retriever: Retriever | None = None) -> tuple[str, list[dict]]:
    retriever = retriever or Retriever()
    documents = retriever.retrieve(question)
    answer = generate_answer(question, documents)
    sources = []
    for document in documents:
        source = {"source": document["source"], "page": document["page"], "score": document["score"]}
        if source not in sources:
            sources.append(source)
    return answer, sources
