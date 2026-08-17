"""
利用 Sequential Process 將文本翻譯成中文，並寫成社群貼文
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.7, model='gpt-4o-mini')

translate_prompt = ChatPromptTemplate.from_template('請將以下英文翻譯成在地的繁體中文：\n{text}')
translate_chain = translate_prompt | llm | StrOutputParser()

post_prompt = ChatPromptTemplate.from_template('請根據以下內容，寫出一篇充滿吸引力的 IG 社群貼文：\n{content}')
post_chain = post_prompt | llm | StrOutputParser()

overall_chain = (
    {'content': translate_chain} | post_chain
)

result = overall_chain.invoke({'text': 'Our new wireless noise-cancelling headphones are officially launched today!'})
print(result)