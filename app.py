
import os
import gradio as gr

from utils.user_interface import UserInterface
from utils.llm_generate import LLMGenerate

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model = os.getenv("model")


with gr.Blocks() as demo: 
    with gr.Row():
        chatbot = gr.Chatbot(height=300)
        history = gr.State(value=[])

    with gr.Row(): 
        system_prompt_radio = gr.Radio(
                ["Code Pilot", "專業知識問答", "文章重點總結", "中英對翻", "無"], show_label=False
        )

    with gr.Row():
        with gr.Accordion("System Prompt", open=False) as accordion:
            system_prompt_input = gr.Textbox(
                placeholder="輸入 system prompt", 
                value="",
                show_label=False
            )

    with gr.Row():
        message = gr.Textbox(placeholder="輸入對話", show_label=False)

    with gr.Row():
        submit_button = gr.Button("Send (0)")
        clear_button  = gr.Button("Clear")
    
    num_tokens = 0

    userinterface = UserInterface()
    llmgenerate = LLMGenerate(model)

    system_prompt_radio.change(
        fn=userinterface.change_prompt, 
        inputs=system_prompt_radio, 
        outputs=system_prompt_input
    )

    submit_button.click(
        fn=userinterface.user_set,
        inputs=[message, chatbot],
        outputs=[message, chatbot] 
    ).then(
        fn=llmgenerate.chat_QA,
        inputs=[system_prompt_input, chatbot],
        outputs=[chatbot, submit_button]
    )
    
    message.submit(
        fn=userinterface.user_set,
        inputs=[message, chatbot],
        outputs=[message, chatbot] 
    ).then(
        fn=llmgenerate.chat_QA,
        inputs=[system_prompt_input, chatbot],
        outputs=[chatbot, submit_button]
    )
        
    clear_button.click(
        fn=userinterface.clear_chat,
        outputs=[chatbot, submit_button] 
    )

if __name__ == '__main__':
    demo.launch()