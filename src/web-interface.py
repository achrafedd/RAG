from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.__main__ import RAG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str
    k: int = 5

@app.post('/')
def search(q: Query):
    rag = RAG()
    answer = rag.answer(q.message, k=q.k)
    return {'message': answer.answer}
