import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chunking import get_split_documents
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

split_docs = get_split_documents("data/Mimikyu.txt")
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vectorstore = Chroma.from_documents(
    documents = split_docs,
    embedding = embeddings,
    persist_directory = './chroma_db',
    collection_name = 'pokemon_mimikyu'
)

print(f"\n文件向量化已完成，{len(split_docs)} 個 chunks 已儲存至 ChromaDB")

# 方法 1：基礎相似度檢索
query = "Mimikyu 為甚麼晚上才會出現？"
similar_docs = vectorstore.similarity_search(query, k=2) # k=2 代表取最相關的前 2 筆

print(f"針對問題『{query}』找到的相關文件：")
for i, doc1 in enumerate(similar_docs):
    print(f"\n結果 {i+1} 來源: {doc1.metadata['source']}")
    print(f"內文: {doc1.page_content}")


# 方法 2 ：分數越小通常代表越相似
results_with_scores = vectorstore.similarity_search_with_score(query, k=1)
doc2, score = results_with_scores[0]
print(f"\n最相似段落內容：{doc2.page_content}")
print(f"\n最相似段落分數：{score:.4f}")