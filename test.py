from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import os
import getpass
from langchain_core.tools import tool
import pandas as pd
from graphrag.config.load_config import load_config
from pathlib import Path
import graphrag.api as api
from langgraph.prebuilt import create_react_agent
from langchain import hub
import asyncio
from langgraph_supervisor import create_supervisor
from langchain_core.messages import convert_to_messages


load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("AZURE_OPENAI_ENDPOINT")
_set_env("AZURE_OPENAI_API_KEY")

PROJECT_DIRECTORY = "ragtest6"
COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"
graphrag_config = load_config(Path(PROJECT_DIRECTORY))
entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")


llm = AzureChatOpenAI(
    api_version="2023-07-01-preview",
    azure_deployment="gpt-4o",
)

@tool
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

agent = create_react_agent(
    model =llm, 
    tools=[local_search],
    name="agent",
    prompt="""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {content}
Thought:{agent_scratchpad}
"""
)

supervisor = create_supervisor(
    model=llm,
    agents=[agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- agent. Assign user's question and searching product information tasks to this agent\n"
        "You will assess if there are uncertainties from the agent's reponse regarding product information, clarify it by letting the agent to search the specific product\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

async def main():
    async for chunk in agent.astream(
        {
            "messages": {"role": "user", "content": "請提供我VRAM 大於70GB的GPU伺服器"}
        },
    ):
        pretty_print_messages(chunk)
    # async for chunk in supervisor.astream(
    #     {
    #         "messages": {"role": "user", "content": "請提供我VRAM 大於70GB的GPU伺服器"}
    #     },
    # ):
    #     pretty_print_messages(chunk, last_message=True)
    # print(f"Respones: {chunk["supervisor"]["messages"][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())