from pathlib import Path
from pprint import pprint

import pandas as pd

import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.index.typing.pipeline_run_result import PipelineRunResult

import asyncio

async def main():
    PROJECT_DIRECTORY = "."
    graphrag_config = load_config(Path(PROJECT_DIRECTORY))
    entities = pd.read_parquet(f"output/entities.parquet")
    communities = pd.read_parquet(f"output/communities.parquet")
    community_reports = pd.read_parquet(
        f"output/community_reports.parquet"
    )

    response, context = await api.global_search(
        config=graphrag_config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=2,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query="Provide me products that can operate under high temperature",
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())