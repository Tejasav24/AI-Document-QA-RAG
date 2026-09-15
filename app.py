from __future__ import annotations

import gradio as gr

from app.services import RAGService


def create_app(rag_service: RAGService) -> gr.Blocks:
    """Create the Gradio interface for the RAG assistant."""

    def ask_question(query: str):
        if not query or not query.strip():
            return "Please enter a question.", "No sources."

        try:
            result = rag_service.ask(
                query=query,
                top_k=3,
            )

            answer = result["answer"]

            if result["sources"]:
                source_lines = []

                for source in result["sources"]:
                    source_lines.append(
                        f"- {source['document_name']} "
                        f"| Page {source['page_number']}"
                    )

                sources = "\n".join(source_lines)
            else:
                sources = "No relevant sources found."

            return answer, sources

        except Exception as exc:
            return (
                f"Error: {exc}",
                "Unable to retrieve sources.",
            )

    with gr.Blocks(
        title="AI Document Q&A",
    ) as demo:

        gr.Markdown(
            "# 📚 AI Document Q&A"
        )

        gr.Markdown(
            "Ask questions about your indexed documents. "
            "Answers are grounded in retrieved document context."
        )

        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about the document...",
                    lines=3,
                )

                ask_button = gr.Button(
                    "Ask Question",
                    variant="primary",
                )

            with gr.Column(scale=3):
                answer = gr.Textbox(
                    label="Answer",
                    lines=6,
                    interactive=False,
                )

                sources = gr.Textbox(
                    label="Sources",
                    lines=4,
                    interactive=False,
                )

        ask_button.click(
            fn=ask_question,
            inputs=question,
            outputs=[answer, sources],
        )

        question.submit(
            fn=ask_question,
            inputs=question,
            outputs=[answer, sources],
        )

    return demo


def launch_app(rag_service: RAGService) -> None:
    """Launch the Gradio application."""

    demo = create_app(rag_service)
    demo.launch()


__all__ = [
    "create_app",
    "launch_app",
]
