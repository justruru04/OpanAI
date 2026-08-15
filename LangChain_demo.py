from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

# 1. 使用 Pydantic 定義資料結構與型別
class ReviewExtraction(BaseModel):
    gift: Optional[str] = Field(
        default=None, 
        description="這份禮物是買給誰的？若未提及請回傳 None"
    )
    delivery_days: int = Field(
        default=-1, 
        description="商品花了幾天送達，請只回傳數字（整數），若未提及請回傳 -1"
    )

# 2. 初始化模型並綁定結構化輸出
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
structured_llm = llm.with_structured_output(ReviewExtraction)

# 3. 簡化 Prompt（不再需要手動注入 format_instructions）
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "請從客戶評價中，精準提取指定的資訊。"),
    ("human", "客戶評價：\n<review>\n{review_text}\n</review>")
])

# 4. 使用 LCEL 建立處理鏈
chain = prompt_template | structured_llm

# 5. 執行
customer_review = "這款玩偶真的太可愛了！我買來送給女朋友當生日禮物，下單後大概 3 天就收到了，非常迅速！"
result: ReviewExtraction = chain.invoke({"review_text": customer_review})

# 輸出結果
print(result.model_dump())
print(f"禮物對象：{result.gift}")
print(f"物流天數：{result.delivery_days}")