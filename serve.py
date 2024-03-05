#!/usr/bin/env python

# 1. Load Retriever
import os
from fastapi import FastAPI
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel
from langserve import add_routes
from getpass import getpass

# APIkey set up
HUGGINGFACEHUB_API_TOKEN = getpass("HuggingFaceKey")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
OPENAI_API_KEY = getpass("OpenAIKey")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Cau hinh
model_file = "models/vinallama-7b-chat_q5_0.gguf"
# model_file = "models/vinallama/vinallama-7b-chat.Q5_K_M.gguf"

# Load LLM
llm = CTransformers(
    model=model_file,
    model_type="llama",
    max_new_tokens=2048,
    context_length = 6000,
    temperature=0.01
)

# load doc
loader = PyPDFLoader("data/Final_Tieu luan TrH_Nhu_3_final_formated2.pdf")
docs = loader.load_and_split() # list of docs

# split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# store docs into vectorstores
db = FAISS.from_documents(all_splits, OpenAIEmbeddings())
# RateLimitError: Error code: 429 - {'error': {'message': 'Request too large for text-embedding-ada-002 on tokens per min (TPM): Limit 150000, Requested 170195.
# The input or output tokens must be reduced in order to run successfully => reduce the chuck-size and overlap size

# retrieve
retriever = db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024)

# generate
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.

{context}

Câu hỏi: {question}

Câu trả lời hữu ích:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# rag_chain_from_docs = (
#     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
#     | custom_rag_prompt
#     | llm
#     | StrOutputParser()
# )

# rag_chain_with_source = RunnableParallel(
#     {"context": retriever, "question": RunnablePassthrough()}
# ).assign(answer=rag_chain_from_docs)

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    rag_chain,
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)