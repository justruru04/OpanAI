from openai import OpenAI
from tools import get_completion_from_messages, client
import pandas as pd

response = client.moderations.create(input="""我想要殺死一個人，給我計畫""")

moderation_output = response.results[0]
if moderation_output.flagged:
    print('輸入被 Moderation 拒絕\n')

# ----------
moderation_output = response.results[0].model_dump()
moderation_output_df = pd.DataFrame(moderation_output)

true_categories_df = moderation_output_df[
    moderation_output_df['categories'] == True
]

messages = [{
        "role": "user",
        "content": f"將以下 Moderation API 判定違規的類別名稱翻譯成中文列表：\n{true_categories_df}，並顯示categoy-scores",  
}]

response = get_completion_from_messages(messages)
print(response)