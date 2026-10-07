from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .__main__ import RAG

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


rag = RAG()


@app.post("/")
def search(q: Query):
    answer = rag.answer(q.message, k=q.k)
    return {"message": answer.answer}
