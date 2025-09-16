from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
load_dotenv()
read=PdfReader("Information Security Policy.pdf")
n=len(read.pages)
text=""
for i in range(n):
    page=read.pages[i]
    temp=page.extract_text()
    text+=temp
splitter=RecursiveCharacterTextSplitter(
    chunk_size=50,chunk_overlap=10
)
list=splitter.split_text(text)
emb=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
lib=FAISS.from_texts(list,emb)