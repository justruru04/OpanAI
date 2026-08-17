"""
最基本的 LCEL 語法練習
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.7, model='gpt-4o-mini')

prompt = ChatPromptTemplate.from_template(
    '你是一位專業的行銷專家。請為產品：{product_name} 撰寫 3 句吸引人的行銷標語。'
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser      # LCEL

result = chain.invoke({'product_name': '無線降噪藍芽耳機'})

print(result)