"""
取得向量資料庫，並利用 MMR 與 Matadata Filter 進行檢索
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    collection_name = 'pokemon_mimikyu',
    embedding_function = embeddings,
    persist_directory = './chroma_db'
)

# MMR
retriever_mmr = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {'k': 4, 'fetch_k': 8, 'lambda_mult': 0.5}
)


if __name__ == "__main__":
    query = "Mimikyu 為甚麼晚上才會出現？"
    results = retriever_mmr.invoke(query)
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "未知來源")
        print(f"\n--- 結果 {i+1}（來源：{source}）---")
        print(doc.page_content)



# Metadata Filter
# filter_results = vectorstore.similarity_search(
#     "query",
#     k = 1,
#     filter = {"key": "value"}
# )
# print("檢索結果：", filter_results[0].page_content)