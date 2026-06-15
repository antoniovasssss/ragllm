import gradio as gr
from dotenv import load_dotenv
from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(docs):
    """Format retrieved context for display."""
    result = ""
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        result += f"📄 Source: {source}\n\n{doc.page_content}\n\n{'-'*50}\n\n"
    return result


def chat(message, history):
    """Handle user input and return assistant response."""
    history = history or []  # ensure it's always a list

    # Pass history into answer_question
    answer, docs = answer_question(message, history)

    # Update history after getting the answer
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, format_context(docs)




def main():
    with gr.Blocks(title="Insurellm Expert Assistant") as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="Conversation", height=500)
                message = gr.Textbox(placeholder="Ask a question...", show_label=False)
            with gr.Column():
                context_box = gr.Textbox(label="Retrieved Context", lines=25)

        message.submit(chat, inputs=[message, chatbot], outputs=[message, chatbot, context_box])
    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
