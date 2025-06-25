from .state import AgentState, llm, prompt, product_urls
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END

from pathlib import Path
from pprint import pprint

import pandas as pd

import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
import re

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from typing import Union, List, Dict, Any

PROJECT_DIRECTORY = "ragtest6"
COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"
graphrag_config = load_config(Path(PROJECT_DIRECTORY))
entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")

def recursively_convert(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, list):
        return [recursively_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: recursively_convert(value) for key, value in obj.items()}
    return obj

def process_context_data(context_data: Union[str, List[pd.DataFrame], Dict, pd.DataFrame]) -> Any:
    if isinstance(context_data, str):
        return context_data
    if isinstance(context_data, pd.DataFrame):
        return context_data.to_dict(orient="records")
    if isinstance(context_data, (list, dict)):
        return recursively_convert(context_data)
    return None


def get_input(state: "AgentState") -> str:
    msg = input("Input: ")
    return {"messages": [HumanMessage(msg)]}

def should_continue(state: "AgentState") -> str:
    if state['messages'][-1].content != "quit":
        return "action"
    else:
        return END

async def tool_node(state: "AgentState") -> str:
    try:
        response, context = await api.local_search(
                                config=graphrag_config,
                                entities=entities,
                                communities=communities,
                                community_reports=community_reports,
                                text_units=text_units,
                                relationships=relationships,
                                covariates=None,
                                community_level=COMMUNITY_LEVEL,                                
                                response_type=RESPONSE_TYPE,
                                query=state['messages'][-1].content,
                            )
        # response = re.sub(r'\[Data: [^\]]*\]', '', response)
        for p in re.findall(r'\[Product: ([^\]]+)\]', response):
            if p in product_urls:
                response = response.replace(f"[Product: {p}]", f"<a href='{product_urls[p]}'>&#128279;</a>")
            else:
                response = response.replace(f"[Product: {p}]", "")
        return {'messages': [process_context_data(context), AIMessage(content=response)]}
    except Exception as e:
        print(f"Error: {e}")
        return {'messages': [{}, AIMessage(content=f"Error: {e}")]}

async def rag(state: "AgentState") -> str:
    PROJECT_DIRECTORY = "ragtest6"
    graphrag_config = load_config(Path(PROJECT_DIRECTORY))
    text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")

    response, context = await api.basic_search(
        config=graphrag_config,
        text_units=text_units,
        query=state['messages'][-1].content,
    )
    print(context)
    return {'messages': [context, AIMessage(content=response)]}

def call_model(state: "AgentState") -> str:
    # history = state['messages'][:-1]
    # context = state['messages'][-2]
    # message = prompt.invoke({'msg': state['messages'][-1].content, 'history': history, 'context': context})
    # response = llm.invoke(message).content
    # return {'messages': [AIMessage(content=response)]}
    return 

