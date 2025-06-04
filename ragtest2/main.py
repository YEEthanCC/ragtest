import os

import pandas as pd
import tiktoken

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from dotenv import load_dotenv
import asyncio

from pathlib import Path
import graphrag.api as api
from graphrag.config.load_config import load_config

load_dotenv()

async def main():
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    llm_model = "gpt-4o"

    config = LanguageModelConfig(
        api_key=api_key,
        api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
        type=ModelType.AzureOpenAIChat,
        api_version="2023-07-01-preview",
        model=llm_model,
        deployment_name=llm_model,
        max_retries=20,
    )
    model = ModelManager().get_or_create_chat_model(
        name="global_search",
        model_type=ModelType.AzureOpenAIChat,
        config=config,
    )

    token_encoder = tiktoken.encoding_for_model(llm_model)
    graphrag_config = load_config(Path("."))

    COMMUNITY_LEVEL = 2

    entity_df = pd.read_parquet(f"output/entities.parquet")
    community_df = pd.read_parquet(f"output/communities.parquet")
    report_df = pd.read_parquet(
        f"output/community_reports.parquet"
    )

    communities = read_indexer_communities(community_df, report_df)
    reports = read_indexer_reports(report_df, community_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
    report_df.head()

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )

    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        model=model,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=True,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    result = await search_engine.search("What camera input does ROM-2820 support?")

    print(result.response)

    print(result.context_data["reports"])

    # inspect number of LLM calls and tokens
    print(
        f"LLM calls: {result.llm_calls}. Prompt tokens: {result.prompt_tokens}. Output tokens: {result.output_tokens}."
    )
    

if __name__ == "__main__":
    asyncio.run(main())

