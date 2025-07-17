from .state import AgentState, llm, prompt, product_urls
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, END
import json
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
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent

from difflib import get_close_matches


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
    if "pass" in state['messages'][-1].content:
        return "agent"
    else:
        return "action"

async def local_search(query: str) -> str:
    """
    Search product information of the company based on query
    Args:
        query (str): The search query string.
    """
    print(f"query: {query}")
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
                                query=query
                            )
        return response
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

async def tool_node(state: "AgentState") -> str:
    try:
        print(f"query: {state['messages'][-1].content}")
        response = await local_search(state['messages'][-1].content)
        # for p in re.findall(r'\[Product: ([^\]]+)\]', response):
        #     if p in product_urls:
        #         response = response.replace(f"[Product: {p}]", f"<a href='{product_urls[p]}'>&#128279;</a>")
        #     else:
        #         response = response.replace(f"[Product: {p}]", "")
        return {'messages': [AIMessage(content=response)]}
    except Exception as e:
        print(f"Error: {e}")
        return {'messages': [AIMessage(content=f"Error: {e}")]}

async def validate_search(state: "AgentState"):

    try:
        judge_prompt = ChatPromptTemplate(
            [
SystemMessage(content="""You are a validation assistant. Your job is to check if product information is complete.

INSTRUCTIONS:
1. Review the search tool response that includes different product description in response to {query}: {response}
2. Idenify the products mentioned in the response without clear technical specifications indicated in the user's query
3. Respond with the product name and the specific missing specifications

RESPONSE RULES:
- If ALL products mentioned in the response contain the required specifications, respond with EXACTLY: pass
- If ANY product lacks specifications or hasn't been searched, respond with: [Product Name]: [specific missing specs]

IMPORTANT: 
- Only respond with "pass" (lowercase, no punctuation) when everything is complete
- Do not use phrases like "已包含所有完整技術規格" or explanatory text
- Be precise: either "pass" or list what's missing

Examples:
✓ Good: pass
✓ Good: GPU Server Model X: VRAM capacity
✗ Bad: 已包含所有完整技術規格，無需補充
✗ Bad: All specifications are complete
"""),
            ]
        )
        print(f"query: {state['messages'][0].content}")
        print(f"response: {state['messages'][1].content}")
        res = llm.invoke(judge_prompt.invoke({'query': state['messages'][0].content, 'response': state['messages'][1].content})).content
        print(f"Validation result: {res}")
        if "pass" in res:
            return 
        else:
            return {'messages': [AIMessage(content=await local_search(res))]}
    except Exception as e:
        print(f"Error invoking model: {e}")
        raise HTTPException(status_code=500, detail="Model invocation failed")


def get_product_name(state: "AgentState") -> str:
    try:
        judge_prompt = ChatPromptTemplate(
            [
                ("system", """{text}\nYou are a helpful assistant that identifies products from text above.

        INSTRUCTIONS:
        1. Review the text and identify the names of products mentioned.
        2. Respond with product names as a raw string, separated by commas.

        RESPONSE RULES:
        - If NO product is mentioned, respond with: 'None'
        - If ANY product is found, respond with product names as a raw string, separated by commas: product1,product2,product3

        IMPORTANT:
        - Only respond with "None" when no product is found.
        - Do not include any other text or explanation, just the product names.""")
            ]
        )
        match_prompt = ChatPromptTemplate(
            [
                ("system", """You are a helpful assistant that matches a list of products: {products} with a list of correct product names: {product_names}.

        INSTRUCTIONS:
        - Matches should be based on the closest name match.
        - If a product name is not found, return 'No match found'.
        - Respond in pure json format with the following structure:[{{"product1": "product_name1"}}, {{"product2": "product_name2"}}, ...]
        - No additional text or explanation should be included in the response.""")
            ]
        )
        res = llm.invoke(judge_prompt.invoke({'text': state['messages'][-1].content})).content
        products = res.split(',')
        trial_count = 0
        product_names = []
        while trial_count < 3:
            try:
                product_names = json.loads(llm.invoke(match_prompt.invoke({'products': products, 'product_names': product_urls.keys()})).content)
                break
            except Exception as e:
                print(f"Error matching product names: {e}")
            trial_count+=1
        print(f"Product names: {product_names}")
        text = state['messages'][-1].content
        for p in product_names:
            try: 
                name = list(p.values())[0]
                url = product_urls[name]
                print(f"name: {name}, url: {url}")
                text = text.replace(list(p.keys())[0], f"<a href='{url}'>{name} &#128279;</a>", 1)
            except Exception as e:
                continue
        return {'messages': [AIMessage(content=text)]}
    except Exception as e:
        print(f"Error invoking model: {e}")
        raise HTTPException(status_code=500, detail="Model invocation failed")

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


