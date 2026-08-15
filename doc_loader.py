from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader,TextLoader

# ----------
pdf_loader = PyPDFLoader('data/aerodynamic_ch3.pdf')
pdf_docs = pdf_loader.load()

print(f"PDF 總共載入了 {len(pdf_docs)} 頁")
print("第 1 頁內文預覽：\n", pdf_docs[0].page_content[:200])
print("第 1 頁 Metadata：", pdf_docs[0].metadata)

# ----------
web_loader = WebBaseLoader("https://wiki.52poke.com/zh-hant/%E8%B0%9C%E6%8B%9F%E4%B8%98")
web_docs = web_loader.load()

print(f"\n網頁文字長度：{len(web_docs[0].page_content)} 字")
print("網頁 Metadata：", web_docs[0].metadata)

# ----------
txt_loader = TextLoader("data/Mimikyu.txt", encoding = "utf-8")
txt_docs = txt_loader.load()
print(f"\n文字來源：{txt_docs[0].metadata['source']}")