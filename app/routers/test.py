from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
import graphrag.api as api
from graphrag.config.load_config import load_config
import json
from typing import Union, List, Dict, Any
import pandas as pd
from graphrag.query.structured_search.base import SearchResult

def convert_response_to_string(response: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Convert a response that can be a string, dictionary, or list of dictionaries to a string.
    """
    if isinstance(response, (dict, list)):
        return json.dumps(response)
    elif isinstance(response, str):
        return response
    else:
        return str(response)

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

def serialize_search_result(search_result: SearchResult) -> Dict[str, Any]:
    return {
        "response": search_result.response,
        "context_data": process_context_data(search_result.context_data),
        "context_text": search_result.context_text,
        "completion_time": search_result.completion_time,
        "llm_calls": search_result.llm_calls,
        "prompt_tokens": search_result.prompt_tokens
    }

test_router = APIRouter()
PROJECT_DIRECTORY = "ragtest4"
COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"
graphrag_config = load_config(Path(PROJECT_DIRECTORY))
entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")

@test_router.get("/status")
async def status():
    print("Status check requested")
    return JSONResponse(content={"status": "Server is up and running"})

@test_router.get("/search/global")
async def global_search(query: str = Query(..., description="Global Search")):
    try:
        response, context = await api.global_search(
                                config=graphrag_config,
                                entities=entities,
                                communities=communities,
                                community_reports=community_reports,                                
                                community_level=COMMUNITY_LEVEL,
                                dynamic_community_selection=False,
                                response_type=RESPONSE_TYPE,
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@test_router.get("/search/local")
async def local_search(query: str = Query(..., description="Local Search")):
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
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }        
        return JSONResponse(content=response_dict)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@test_router.get("/search/local")
async def local_search(query: str = Query(..., description="Local Search")):
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
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }        
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@test_router.get("/search/drift")
async def drift_search(query: str = Query(..., description="DRIFT Search")):
    try:
        response, context = await api.drift_search(
                                config=graphrag_config,
                                entities=entities,
                                communities=communities,
                                community_reports=community_reports,
                                text_units=text_units,
                                relationships=relationships,
                                community_level=COMMUNITY_LEVEL,                                
                                response_type=RESPONSE_TYPE,
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@test_router.get("/search/basic")
async def basic_search(query: str = Query(..., description="Basic Search")):
    try:
        response, context = await api.basic_search(
                                config=graphrag_config,
                                text_units=text_units,                                
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))