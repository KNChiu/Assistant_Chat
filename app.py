
import os
import tiktoken
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model = os.getenv("model")

llm = ChatOpenAI(temperature=1.0, model=model)
encoding = tiktoken.encoding_for_model(model)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def predict(system_prompt, message, history):
    history_langchain_format = [SystemMessage(content=system_prompt)]
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm.invoke(history_langchain_format)

    return gpt_response.content

with gr.Blocks() as demo: 
    def userSet(message, history):
        return "", history + [[message, ""]]
    
    def chat(system_prompt, history):
        history[-1][1] = predict(system_prompt, history[-1][0], history[:-1])
        num_tokens = num_tokens_from_string(system_prompt + str(history))

        return history, f"Send ({str(num_tokens)})"
    
    def clear_chat():
        return "", "Send (0)"
    
    with gr.Row():
        chatbot = gr.Chatbot(height=300)
        history = gr.State(value=[])

    with gr.Row():  
        with gr.Accordion("System Prompt", open=False) as accordion:
            system_prompt_input = gr.Textbox(
                placeholder="輸入 system prompt", 
                value="你是一個生活在台灣的資深軟體工程師，使用 python 為主的程式語言，請根據提問生成合適的程式碼，並條列說明功能",
                show_label=False
            )

    with gr.Row():
        message = gr.Textbox(placeholder="輸入對話", show_label=False)

    with gr.Row():
        submit_button = gr.Button("Send (0)")
        clear_button  = gr.Button("Clear")
    
    num_tokens = 0

    submit_button.click(
        fn=userSet,
        inputs=[message, chatbot],
        outputs=[message, chatbot] 
    ).then(
        fn=chat,
        inputs=[system_prompt_input, chatbot],
        outputs=[chatbot, submit_button]
    )
        
    clear_button.click(
        fn=clear_chat,
        outputs=[chatbot, submit_button] 
    )

if __name__ == '__main__':
    demo.launch()