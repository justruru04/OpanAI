from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.7, model='gpt-4o-mini')

classification_prompt = ChatPromptTemplate.from_template(
    "請判斷以下使用者提問屬於哪種分類？只回答單字：'技術'、'客服'或'其他'。\n提問：{question}"
)
classification_chain = classification_prompt | llm | StrOutputParser()

tech_chain = ChatPromptTemplate.from_template('你是一位資深工程師，請專業解答技術問題：\n{question}') | llm | StrOutputParser()
service_chain = ChatPromptTemplate.from_template('你是一位親切的客服專員，請禮貌回應客服問題：\n{question}') | llm | StrOutputParser()
general_chain = ChatPromptTemplate.from_template('請簡短回答以下問題：\n{question}') | llm | StrOutputParser()

branch = RunnableBranch(
    (lambda x: '技術' in x['topic'], tech_chain),
    (lambda x: '客服' in x['topic'], service_chain),
    general_chain
)

full_chain = (
    {
        'topic': classification_chain,
        'question': RunnablePassthrough()
    }
    | branch
)

# full_chain = (
#     RunnablePassthrough.assign(topic = classification_chain) | branch
# )

print(full_chain.invoke({'question': '請問 python 的裝飾器 @ 是甚麼原理？'}))