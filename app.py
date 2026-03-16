import gradio as gr
from rag_pipeline import get_answer

def ask_question(question):
    return get_answer(question)

iface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="AI-Powered Document QA System",
    description="Ask questions from your document"
)

iface.launch()