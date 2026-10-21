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
    """Request payload used by the HTTP API to query the RAG system.

    Attributes:
        message: The user question to answer.
        k: Number of retrieved sources to provide to the model.
    """

    message: str
    k: int = 5


rag = RAG()


@app.post("/")
def search(q: Query) -> dict:
    """Answer a question through the local HTTP API.

    Args:
        q: Incoming request with the query message and result count.

    Returns:
        A dictionary containing the generated answer text.
    """
    answer = rag.answer(q.message, k=q.k)
    return {"message": answer.answer}
