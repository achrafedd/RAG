from typing import Any
from chromadb.api.types import OneOrMany
from chromadb.types import Metadata
from sentence_transformers import SentenceTransformer
from pathlib import Path
import chromadb


class Semantic:
    def __init__(self) -> None:
        self.path: Path = Path("data/")
        self.processed: Path = self.path / "processed/chromab"
        self.client = chromadb.PersistentClient(self.processed)
        self.collection = self.client.get_or_create_collection(
            'collection',
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 100,
                "hnsw:construction_ef": 200
            })
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def index(
        self, documents: list[str], chunks: list[dict[str, Any]]
    ) -> None:
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise ValueError("can't get embeddings dimensions")
        embeddings = self.model.encode(documents, show_progress_bar=True)
        ids = [chunk['id'] for chunk in chunks]
        metadatas: OneOrMany[Metadata] = [{'file_path': chunk['file_path'],
            'first_character_index': chunk['first_character_index'],
            'last_character_index': chunk['last_character_index']}
        for chunk in chunks]
        max_bs = self.client.get_max_batch_size()
        for start in range(0, len(chunks), max_bs):
            end = start + max_bs
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                documents=documents[start:end]
            )

    def search(self, query: str, k: int) -> list[tuple[str, dict[str, Any]]]:
        query_emd = self.model.encode([query])
        search_results = self.collection.query(
            query_embeddings=query_emd,
            n_results=k
        )
        semantic_results = []
        for idx in range(len(search_results['ids'][0])):
            metadatas = search_results['metadatas'][0][idx]
            semantic_results.append(
                (
                    search_results['ids'][0][idx],
                    {
                        "file_path": metadatas["file_path"],
                        "first_character_index": metadatas[
                            "first_character_index"
                        ],
                        "last_character_index": metadatas[
                            "last_character_index"
                        ],
                    },
                )
            )
        return semantic_results
