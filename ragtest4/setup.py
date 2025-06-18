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

    index_result: list[PipelineRunResult] = await api.build_index(config=graphrag_config)

    # index_result is a list of workflows that make up the indexing pipeline that was run
    for workflow_result in index_result:
        status = f"error\n{workflow_result.errors}" if workflow_result.errors else "success"
        print(f"Workflow Name: {workflow_result.workflow}\tStatus: {status}")

if __name__ == "__main__":
    asyncio.run(main())