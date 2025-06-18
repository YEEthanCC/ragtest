from .state import AgentState, llm, prompt
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

def get_input(state: "AgentState") -> str:
    msg = input("Input: ")
    return {"messages": [HumanMessage(msg)]}

def should_continue(state: "AgentState") -> str:
    if state['messages'][-1].content != "quit":
        return "action"
    else:
        return END

async def tool_node(state: "AgentState") -> str:
    PROJECT_DIRECTORY = "ragtest4"
    graphrag_config = load_config(Path(PROJECT_DIRECTORY))
    entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
    communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
    community_reports = pd.read_parquet(
        f"{PROJECT_DIRECTORY}/output/community_reports.parquet"
    )

    response, context = await api.global_search(
        config=graphrag_config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=2,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=state['messages'][-1].content,
    )
    df = context['reports']
    new_response = ""
    pattern = r"\[Data: Reports \((.*?)\)\]"
    for line in response.splitlines():
        match = re.search(pattern, line)
        if match:
            for id in match.group(1).split(','):
                try:
                    content = df[df['id'] == id.strip()]['content'].iloc[0]
                    line+=(f"<a>{id}</a>")
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        line+="\n"        
        new_response+=line
    context['reports'].to_csv('result_context.csv', index=False)
    return {'messages': [context['reports'].to_json(orient="records"), AIMessage(content=new_response)]}

async def rag(state: "AgentState") -> str:
    PROJECT_DIRECTORY = "ragtest4"
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

