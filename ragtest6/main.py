import os
from pathlib import Path

import pandas as pd
import tiktoken

from graphrag.config.enums import ModelType
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.structured_search.drift_search.drift_context import (
    DRIFTSearchContextBuilder,
)
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from dotenv import load_dotenv

load_dotenv()

import asyncio

async def main():
    COMMUNITY_LEVEL = 2
    chat_config = LanguageModelConfig(
        api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
        api_base=os.environ.get('AZURE_OPENAI_ENDPOINT'),
        type=ModelType.AzureOpenAIEmbedding,
        model='gpt-4o',
        max_retries=20,
    )    
    chat_model = ModelManager().get_or_create_chat_model(
        name="local_search",
        model_type=ModelType.AzureOpenAIEmbedding,
        config=chat_config,
    )
    embedding_config = LanguageModelConfig(
        api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
        api_base=os.environ.get('AZURE_OPENAI_ENDPOINT'),
        type=ModelType.OpenAIEmbedding,
        model='gpt-4o',
        max_retries=20,
    )
    text_embedder = ModelManager().get_or_create_embedding_model(
        name="local_search_embedding",
        model_type=ModelType.AzureOpenAIEmbedding,
        config=embedding_config,
    )
    drift_params = DRIFTSearchConfig(
        temperature=0,
        max_tokens=12_000,
        primer_folds=1,
        drift_k_followups=3,
        n_depth=3,
        n=1,
    )
    entity_df = pd.read_parquet(f"output/entities.parquet")
    community_df = pd.read_parquet(f"output/communities.parquet")
    entities = read_indexer_entities(entity_df, community_df, COMMUNITY_LEVEL)
    relationship_df = pd.read_parquet(f"output/relationships.parquet")
    relationships = read_indexer_relationships(relationship_df)
    def read_community_reports(
        input_dir: str,
        community_report_table: str = "community_reports",
    ):
        """Embeds the full content of the community reports and saves the DataFrame with embeddings to the output path."""
        input_path = Path(input_dir) / f"{community_report_table}.parquet"
        return pd.read_parquet(input_path)
    report_df = read_community_reports('output')
    reports = read_indexer_reports(
        report_df,
        community_df,
        COMMUNITY_LEVEL,
        content_embedding_col="full_content_embeddings",
    )
    description_embedding_store = LanceDBVectorStore(
        collection_name="default-entity-description",
    )
    description_embedding_store.connect(db_uri='output/lancedb')
    text_unit_df = pd.read_parquet(f"output/text_units.parquet")
    text_units = read_indexer_text_units(text_unit_df)
    token_encoder = tiktoken.encoding_for_model('gpt-4o')

    context_builder = DRIFTSearchContextBuilder(
        model=chat_model,
        text_embedder=text_embedder,
        entities=entities,
        relationships=relationships,
        reports=reports,
        entity_text_embeddings=description_embedding_store,
        text_units=text_units,
        token_encoder=token_encoder,
        config=drift_params,
    )

    search = DRIFTSearch(
        model=chat_model, context_builder=context_builder, token_encoder=token_encoder
    )
    resp = await search.search("請提供我VRAM 70GB的GPU伺服器")
    print(resp.response)

if __name__ == "__main__":
    asyncio.run(main())