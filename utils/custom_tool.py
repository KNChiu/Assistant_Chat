import os
from typing import Type
from langchain.tools import BaseTool
from langchain.agents import (
    create_tool_calling_agent,
    AgentExecutor,
    load_tools,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain import hub


class TranslateInput(BaseModel):
    targetLang: str = Field(description="目標語言使用 ISO-639 標準")
    text: str = Field(description="需要翻譯的文字。它可以是任何語言")

class TranslateTool(BaseTool):
    '''依據輸入的文字找出要翻譯的目標語言與預翻譯的文字'''

    name = "translate"
    description = "依據輸入的文字找出要翻譯的目標語言與預翻譯的文字"
    args_schema: Type[BaseModel] = TranslateInput

    def _run(
        self,
        targetLang: str = "zh-TW",
        text: str = None,
    ):
        print(f"Tool: TranslateTool | {targetLang}, {text}")

        if targetLang and text:
            return f"""將以下文字片段{text}翻譯成通順的{targetLang}"""
        else:
            return f"""無法翻譯{text}成{targetLang}請重試"""

class CopilotInput(BaseModel):
    language: str = Field(description="使用的程式語言")
    code: str = Field(description="需求或是需要修改的程式碼片段")

class CopilotTool(BaseTool):
    '''依據一段需求撰寫程式，或是修改一段程式碼'''

    name = "Copilot"
    description = "依據一段需求撰寫程式，或是修改一段程式碼"
    args_schema: Type[BaseModel] = CopilotInput

    def _run(
        self,
        language: str = "python",
        code: str = None,
    ):
        print(f"Tool: CopilotTool | {language}")

        return f"""你是一位資深軟體工程師，請依據{code}生成或是修改成合適的{language}程式碼，並使用繁體中文條列說明功能"""

class ProfessionalInput(BaseModel):
    question: str = Field(description="問題")

class ProfessionalTool(BaseTool):
    '''人工智慧相關領域的專業問答'''

    name = "Professional"
    description = "人工智慧相關領域的專業問答"
    args_schema: Type[BaseModel] = ProfessionalInput

    def _run(
        self,
        question: str,
    ):
        print(f"Tool: ProfessionalTool | {question}")

        return f"""你是一位人工智慧領域的專家，請專業並有邏輯的使用繁體中文回答，問題: {question}"""


def get_custom_tool(llm):
    tools = load_tools(["google-search"], llm=llm)
    tools += [TranslateTool(), CopilotTool(), ProfessionalTool()]

    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    return agent_executor


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    # os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
    # os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    agent_executor = get_custom_tool(llm)

    chat_return = agent_executor.invoke({"input": "翻譯How are you?"})
    print(chat_return["output"])
    print("===========================================")

    chat_return = agent_executor.invoke({"input": "幫我查一下台南今天的天氣如何"})
    print(chat_return["output"])
    print("===========================================")


    chat_return = agent_executor.invoke({"input": "幫我用python生成一個自動累加的程式碼"})
    print(chat_return["output"])
    print("===========================================")

    chat_return = agent_executor.invoke({"input": "說明人工智慧中Translation self-attention的架構"})
    print(chat_return["output"])
    print("===========================================")

