from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os
import getpass
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, Annotated
import operator


load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("AZURE_OPENAI_ENDPOINT")
_set_env("AZURE_OPENAI_API_KEY")

llm = AzureChatOpenAI(
    api_version="2023-07-01-preview",
    azure_deployment="gpt-4o",
)

prompt = ChatPromptTemplate([
    ("system", "You are an intelligent, friendly, and helpful AI chatbot. You will respond to user questions regarding the company's product. You will be provided with the context related to questions. If the information is not provided just respond, I'm sorry, I do not have information related to your question."), 
    MessagesPlaceholder("history"), 
    ("system" "{context}"),
    ("user", "{msg}")
])

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]