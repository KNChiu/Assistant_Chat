import os
import gradio as gr

from utils.user_interface import UserInterface
from utils.llm_generate import LLMGenerate
from utils.google_search import SearchQA

from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = os.getenv("model")


with gr.Blocks() as demo:
    with gr.Row():
        chatbot = gr.Chatbot(height=350)
        history = gr.State(value=[])

    with gr.Row():
        with gr.Accordion("Setting", open=False) as accordion:
            with gr.Row():
                with gr.Column(scale=6):
                    system_prompt_input = gr.Textbox(
                        placeholder="輸入 system prompt", value="你是一位專業的AI助手，依據用戶的問題進行思考並選擇合適的工具，使用繁體中文回應", show_label=False
                    )
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                        value="gpt-4o-mini",
                        show_label=False,
                    )

    with gr.Row():
        message = gr.Textbox(placeholder="輸入對話", show_label=False)

    with gr.Row():
        submit_button = gr.Button("Send (0)")
        clear_button = gr.Button("Clear")

    num_tokens = 0
    
    userinterface = UserInterface()
    llmgenerate = LLMGenerate(model)

    def task_choice(system_prompt_input, chatbot):
        chatbot, submit_button = llmgenerate.chat_tool(system_prompt_input, chatbot)

        return chatbot, submit_button

    def change_model(model_dropdown):
        global model, searchqa, llmgenerate
        model = model_dropdown
        searchqa = SearchQA(model)
        llmgenerate = LLMGenerate(model)


    model_dropdown.change(
        fn=change_model,
        inputs=model_dropdown,
    )

    submit_button.click(
        fn=userinterface.user_set, inputs=[message, chatbot], outputs=[message, chatbot]
    ).then(
        fn=task_choice,
        inputs=[system_prompt_input, chatbot],
        outputs=[chatbot, submit_button],
    )

    message.submit(
        fn=userinterface.user_set, inputs=[message, chatbot], outputs=[message, chatbot]
    ).then(
        fn=task_choice,
        inputs=[system_prompt_input, chatbot],
        outputs=[chatbot, submit_button],
    )

    clear_button.click(fn=userinterface.clear_chat, outputs=[chatbot, submit_button])

if __name__ == "__main__":
    demo.launch()
