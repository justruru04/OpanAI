"""
將 Document 物件 splitting 成 chunks
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_split_documents(file_path: str = "data/Mimikyu.txt", encoding = 'utf-8') -> list[Document]:
    loader = TextLoader(file_path, encoding = encoding)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 50,
        chunk_overlap = 10,
        separators = ["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(raw_docs)
    return split_docs

chunks = get_split_documents()

print(f"總共有 {len(chunks)} 個 chunks")
print("第一個 chunk 內容：", chunks[0].page_content)
print("第一個 chunk metadata：", chunks[0].metadata)