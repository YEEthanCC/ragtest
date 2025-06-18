from fastapi import APIRouter
from pydantic import BaseModel
from app.agents import graph, rag_graph
from langchain_core.messages import HumanMessage
from fastapi.responses import JSONResponse


router = APIRouter()

class Message(BaseModel):
    message: str

@router.post("/chat/")
async def connect(msg: Message):
    print(f"Message received: {msg}")
    response = []
    async for event in graph.astream({"messages": [HumanMessage(msg.message)]}, stream_mode='values'):
        response.append(event["messages"][-1].content)
    return {"response": event["messages"][-1].content, "context": event["messages"][-2]}

@router.post("/rag/")
async def rag(msg: Message):
    print(f"Message received: {msg}")
    response = []
    async for event in rag_graph.astream({"messages": [HumanMessage(msg.message)]}, stream_mode='values'):
        response.append(event["messages"][-1].content)
    return {"response": event["messages"][-1].content, "context": event["messages"][-2]}
