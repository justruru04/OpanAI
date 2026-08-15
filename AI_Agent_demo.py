from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

@tool
def calculate_expression(expression: str) -> str:
    """精確計算數學運算式的工具。當需要做任何四則運算、次方、開根號等數學計算時，必須使用此工具。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"計算出錯：{e}"

@tool
def get_word_length(word: str) -> int:
    """計算給定字串的長度/字數。"""
    return len(word)

tools = [calculate_expression, get_word_length]
system_prompt = "你是一位聰明且善於利用外部工具解決問題的 AI 助手。"

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt
)

question = "請幫我計算 12345 乘以 6789 等於多少？並告訴我結果總共有幾位數？"
result = agent.invoke({"messages": [("user", question)]})

final_answer = result["messages"][-1].content
print("\n最終回答：\n", final_answer)