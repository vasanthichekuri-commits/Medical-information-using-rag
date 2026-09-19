import gradio as gr

from src.rag_pipeline import answer_question


def chat(question: str):
    if not question or not question.strip():
        return "Please enter a medical information question.", ""
    try:
        answer, sources = answer_question(question)
    except FileNotFoundError as error:
        return str(error), ""
    source_text = "\n".join(
        f"{index}. {source['source']} - Page {source['page']} - Similarity {source['score']:.2f}"
        for index, source in enumerate(sources, start=1)
    )
    return answer, source_text or "No supporting sources found."


with gr.Blocks(title="Medical Information RAG Assistant") as demo:
    gr.Markdown("# Medical Information RAG Assistant\nAsk general health-information questions answered from the indexed medical documents.")
    gr.Markdown("**Medical safety notice:** Educational information only. This system does not diagnose conditions or provide personalized treatment. Consult a qualified healthcare professional for personal concerns.")
    question = gr.Textbox(label="Question", placeholder="What are the complications of diabetes?", lines=3)
    ask_button = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")
    sources = gr.Textbox(label="Sources", lines=5, interactive=False)
    ask_button.click(chat, inputs=question, outputs=[answer, sources])
    question.submit(chat, inputs=question, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
