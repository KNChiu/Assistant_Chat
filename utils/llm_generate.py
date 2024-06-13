import tiktoken
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

class LLMGenerate():
    def __init__(self, model) -> None:
        self.llm = ChatOpenAI(temperature=1.0, model=model)
        self.encoding = tiktoken.encoding_for_model(model)
        
    def _tokens_calculation(self, string: str) -> int:
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def _predict(self, system_prompt:str , message:str , history:str) -> str:
        history_langchain_format = [SystemMessage(content=system_prompt)]
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = self.llm.invoke(history_langchain_format)

        return gpt_response.content

    def chat_QA(self, system_prompt: str, history: list):
        history[-1][1] = self._predict(system_prompt, history[-1][0], history[:-1])
        num_tokens = self._tokens_calculation(system_prompt + str(history))

        return history, f"Send ({str(num_tokens)})"
    
if __name__ == "__main__":
    model = 'gpt-3.5-turbo'
    system_prompt = "你是一位AI助手，解決人們的問題"
    history = [["台灣最高的山是?", ""]]

    llmgenerate = LLMGenerate(model)

    llmgenerate.chat_QA(system_prompt, history)
