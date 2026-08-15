import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

_ = load_dotenv(find_dotenv())  # 將 env 注入至環境變數並回傳布林值
client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))     # 從環境變數中提取 API KEY

# 單次指令執行；輸入型態為 str
def get_completion(prompt: str, model = "gpt-4o-mini", temperature = 0.0, max_tokens = 500):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens = max_tokens
    )
    return response.choices[0].message.content

# system 設定與歷史對話；輸入型態為 list
def get_completion_from_messages(messages: list, model = "gpt-4o-mini", temperature = 0.0, max_tokens = 500):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens = max_tokens
    )
    return response.choices[0].message.content

def get_completion_and_token_count(messages, model = 'gpt-4o-mini', temperature = 0.0, max_token = 500):
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_token
    )

    content = response.choices[0].message.content

    token_dict = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }

    return content, token_dict