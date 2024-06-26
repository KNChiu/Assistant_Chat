import tiktoken
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from utils.custom_tool import get_custom_tool


class LLMGenerate:
    def __init__(self, model) -> None:
        self.llm = ChatOpenAI(temperature=1.0, model=model)
        self.encoding = tiktoken.encoding_for_model(model)

    def _tokens_calculation(self, string: str) -> int:
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def _make_history(self, system_prompt: str, history: str) -> str:
        history_langchain_format = [SystemMessage(content=system_prompt)]
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))

        return history_langchain_format

    def chat_QA(self, system_prompt: str, history: list) -> tuple:
        history_langchain_format = self._make_history(system_prompt, history[:-1])
        history_langchain_format.append(HumanMessage(content=history[-1][0]))

        ### Generate
        history[-1][1] = self.llm.invoke(history_langchain_format).content

        num_tokens = self._tokens_calculation(system_prompt + str(history))

        return history, f"Send ({str(num_tokens)})"

    def chat_tool(self, system_prompt: str, history: list):
        history_langchain_format = self._make_history(system_prompt, history[:-1])

        ### Generate
        agent_executor = get_custom_tool(self.llm)
        history[-1][1] = agent_executor.invoke(
            {
                "input": history[-1][0],
                "chat_history": history_langchain_format,
            }
        ).get("output")

        num_tokens = self._tokens_calculation(system_prompt + str(history))

        return history, f"Send ({str(num_tokens)})"


if __name__ == "__main__":
    model = "gpt-3.5-turbo"
    system_prompt = "你是一位AI助手，解決人們的問題"
    history = [["台灣最高的山是?", ""]]

    llmgenerate = LLMGenerate(model)

    llmgenerate.chat_QA(system_prompt, history)
