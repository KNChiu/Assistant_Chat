import os
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = os.getenv("model")

class SearchQA:
    def __init__(self, model):
        self.vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")
        self.llm = ChatOpenAI(temperature=1.0, model=model)
        self.search = GoogleSearchAPIWrapper()

        self.memory = ConversationSummaryBufferMemory(llm=self.llm, input_key='question', output_key='answer', return_messages=True)
        self.web_research_retriever = WebResearchRetriever.from_llm(vectorstore=self.vectorstore, llm=self.llm, search=self.search)

    def search_QA(self, user_input):
        template = """輸入為長篇文件中提取的部分和一個問題，回答一個最終答案並附上參考資料 ("參考資料"). 
如果你不知道答案，就坦率說不知道，不要隨意瞎猜。
在你的回答中，一定要包括("參考資料")部分。

問題: {question}
=========
{summaries}
=========
最終答案 : 
參考資料: """  
        prompt = PromptTemplate(template=template, input_variables=["summaries", "question"])

        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=self.llm, 
                    chain_type="stuff",
                    retriever=self.web_research_retriever,
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True,
                    verbose=True
                )
        result = qa_chain.invoke({"question": user_input})
        return result["answer"], result["sources"]

if __name__ == "__main__":
    search_qa = SearchQA(model)
    user_input = "2024年台灣的總統是誰"
    answer, sources = search_qa.search_QA(user_input)
    print(answer)
    print(sources)