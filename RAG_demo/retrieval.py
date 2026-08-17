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
    search_kwargs = {'k': 2, 'fetch_k': 4, 'lambda_mult': 0.5}
)
mmr_results = retriever_mmr.invoke("Mimikyu 為甚麼晚上才會出現？")


# Metadata Filter
# filter_results = vectorstore.similarity_search(
#     "query",
#     k = 1,
#     filter = {"key": "value"}
# )
# print("檢索結果：", filter_results[0].page_content)