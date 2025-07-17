from fastapi import APIRouter
from pydantic import BaseModel
from app.agents import graph, rag_graph
from langchain_core.messages import HumanMessage
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional


router = APIRouter()

class Message(BaseModel):
    message: str

@router.post("/chat/")
async def connect(msg: Message):
    async for event in graph.astream({"messages": [HumanMessage(msg.message)]}, stream_mode='values'):
        print(event["messages"][-1].content)
    return {"response": event["messages"][-1].content}

@router.post("/rag/")
async def rag(msg: Message):
    print(f"Message received: {msg}")
    async for event in rag_graph.astream({"messages": [HumanMessage(msg.message)]}, stream_mode='values'):
        print(event["messages"][-1].content)
    return {"response": event["messages"][-1].content}


class Request(BaseModel):
    model: str
    prompt: Optional[str] = None
    messages: Optional[list] = None

@router.post("/api/chat")
async def chat(req: Request):
    start_time = datetime
    if not req.messages:
        print("model in request")
        return {
            "model": "llama", 
            "created_at": datetime.now().time(),
            "response": "",
            "done": True
        }
    else:
        async for event in graph.astream({"messages": [HumanMessage(req.messages[-1]['content'])]}, stream_mode='values'):
            print(event["messages"][-1].content)
        return {
            "model": "llama",
            "created_at": "2023-12-12T14:13:43.416799Z",
            "message": {
                "role": "assistant",
                "content": event["messages"][-1].content
            },
            "done": True,
            "total_duration": 5191566416,
            "load_duration": 2154458,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 383809000,
            "eval_count": 298,
            "eval_duration": 4799921000
        }
