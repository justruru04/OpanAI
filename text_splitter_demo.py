from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data/Mimikyu.txt", encoding="utf-8")
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 10,
    separators = ["\n\n", "\n", " ", ""]
)

split_docs = text_splitter.split_documents(raw_docs)

print(f"總共有 {len(split_docs)} 個 chunks")