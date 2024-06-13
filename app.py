
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
                system_prompt_input = gr.Textbox(
                    placeholder="輸入 system prompt", 
                    value="",
                    show_label=False
                )

            with gr.Row():
                with gr.Column(scale=6):
                    system_prompt_radio = gr.Radio(
                        choices=["Code", "問答", "總結", "翻譯", "搜尋", "無"],
                        show_label=False
                    )
                    
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=['gpt-3.5-turbo', 'gpt-4o'],
                        value='gpt-3.5-turbo',
                        show_label=False
                    )

    with gr.Row():
        message = gr.Textbox(placeholder="輸入對話", show_label=False)

    with gr.Row():
        submit_button = gr.Button("Send (0)")
        clear_button  = gr.Button("Clear")
    
    num_tokens = 0

    userinterface = UserInterface()
    
    searchqa = SearchQA(model)
    llmgenerate = LLMGenerate(model)

    def task_choice(system_prompt_radio, system_prompt_input, chatbot):
        if system_prompt_radio == "搜尋":
            chatbot[-1][1], _ = searchqa.search_QA(chatbot[-1][0])
            submit_button = "Send (search)"
        else:
            chatbot, submit_button = llmgenerate.chat_QA(system_prompt_input, chatbot)

        return chatbot, submit_button
    
    def change_model(model_dropdown):
        global model, searchqa, llmgenerate
        model = model_dropdown
        searchqa = SearchQA(model)
        llmgenerate = LLMGenerate(model)


    system_prompt_radio.change(
        fn=userinterface.change_prompt, 
        inputs=system_prompt_radio, 
        outputs=system_prompt_input
    )


    model_dropdown.change(
        fn=change_model, 
        inputs=model_dropdown,
    )

    submit_button.click(
        fn=userinterface.user_set,
        inputs=[message, chatbot],
        outputs=[message, chatbot] 
    ).then(
        fn=task_choice,
        inputs=[system_prompt_radio, system_prompt_input, chatbot],
        outputs=[chatbot, submit_button]
    )
    
    message.submit(
        fn=userinterface.user_set,
        inputs=[message, chatbot],
        outputs=[message, chatbot] 
    ).then(
        fn=task_choice,
        inputs=[system_prompt_radio, system_prompt_input, chatbot],
        outputs=[chatbot, submit_button]
    )
        
    clear_button.click(
        fn=userinterface.clear_chat,
        outputs=[chatbot, submit_button] 
    )

if __name__ == '__main__':
    demo.launch()