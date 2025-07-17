from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os
import getpass
from langgraph.graph import StateGraph, END, START
from .utils import AgentState, get_input, call_model, should_continue, tool_node, validate_search, get_product_name


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["azure_openai"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("get_input", get_input)
workflow.add_node("action", tool_node)
workflow.add_node("agent", call_model)
workflow.add_node("get_product_name", get_product_name)
workflow.add_edge(START, "action")
workflow.add_edge("action", "get_product_name")
workflow.add_edge("get_product_name", "agent")
workflow.add_edge("agent", END)

graph = workflow.compile()