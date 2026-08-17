"""
利用 RunnableWithMessageHistory 進行記憶管理
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.7, model='gpt-4o-mini')
parser = StrOutputParser()

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",          
    token_counter=llm,        
    include_system=True,      
    allow_partial=False,      
    start_on="human",         
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一位專業且親切的 AI 助手。'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])

chain_with_trimming = (
    RunnablePassthrough.assign(chat_history=lambda x: trimmer.invoke(x["chat_history"]))
    | prompt
    | llm
    | parser
)

session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根據 session_id 取得或建立該使用者的歷史對話"""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    runnable = chain_with_trimming,
    get_session_history = get_session_history,
    input_messages_key = "input",
    history_messages_key = "chat_history",
)

# 執行與測試
config_user_a = {'configurable': {'session_id': 'user_101'}}

res1 = chain_with_memory.invoke(
    {'input': '你好，我是 Alex，我住在台灣。'},
    config = config_user_a
)
print('AI 對 user A 的回覆：', res1)

res2 = chain_with_memory.invoke(
    {'input': '請問我剛剛說我住哪裡？'},
    config=config_user_a
)
print('AI 記憶測試回覆：', res2)