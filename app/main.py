from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.routers import router
from app.routers import test_router

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
)

app.include_router(router)
app.include_router(test_router)