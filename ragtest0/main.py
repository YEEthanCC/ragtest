import os
import json
from dotenv import load_dotenv
import getpass
from langchain_unstructured import UnstructuredLoader
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma



file_path = "input/ROM-2820.txt"
db = None
load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("AZURE_OPENAI_ENDPOINT")
_set_env("AZURE_OPENAI_API_KEY")

embed_model = AzureOpenAIEmbeddings(model="text-embedding-3-small")

if len(os.listdir('knowledge-base/txt')) == 0:
    docs = UnstructuredLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=30,
        add_start_index=True)

    all_splits = text_splitter.split_documents(docs)
    filtered_splits = filter_complex_metadata(all_splits)

    print(f"Split the file into {len(all_splits)} sub-documents.")

    # vector storage
    db = Chroma.from_documents(
        documents=filtered_splits, 
        embedding=embed_model,
        persist_directory="./knowledge-base/txt"
    )
else:
    db = Chroma(persist_directory="./knowledge-base/txt", embedding_function=embed_model)

llm = AzureChatOpenAI(
    api_version="2023-07-01-preview",
    azure_deployment="gpt-4o",
)

while True:
    query = input("Query: ")
    res = db.as_retriever().invoke(query)
    if len(res) != 0:
        print(db.as_retriever().invoke(query)[0].page_content)
    else:
        print("no content is found")