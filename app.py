
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

def change_prompt(choice: str) -> str:
    """
    Return different prompt messages based on the user's choice

    Parameters:
    choice (str): The user's choice

    Returns:
    str: The corresponding prompt message for the choice, respond in Traditional Chinese by default
    """
    if choice == "Code Pilot":
        return "你是一個生活在台灣的資深軟體工程師，使用 python 為主的程式語言，請根據提問生成合適的程式碼，並使用繁體中文條列說明功能"
    elif choice == "專業知識問答":
        return "你是一位人工智慧領域的專家，請專業並有邏輯的使用繁體中文回答問題"
    elif choice == "文章重點總結":
        return "你是一位重點統整的專家，請依據輸入的內容統整成簡短且有意義的文字，使用繁體中文回答"
    elif choice == "中英對翻":
        return "依據輸入的文字，判斷是英文還是中文，如果輸入是英文翻譯成通順的中文，如果輸入是中文則翻譯成通順的英文"
    else:
        return "使用繁體中文回應以下問題"

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

def userSet(message, history):
        return "", history + [[message, ""]]
    
def chat(system_prompt, history):
    history[-1][1] = predict(system_prompt, history[-1][0], history[:-1])
    num_tokens = num_tokens_from_string(system_prompt + str(history))

    return history, f"Send ({str(num_tokens)})"

def clear_chat():
    return "", "Send (0)"

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

    system_prompt_radio.change(
        fn=change_prompt, 
        inputs=system_prompt_radio, 
        outputs=system_prompt_input
    )

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