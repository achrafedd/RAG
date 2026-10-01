from typing import Any
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import json


class Semantic:
    def __init__(self) -> None:
        self.path: Path = Path("data/")
        self.processed: Path = self.path / "processed/faiss"
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
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(self.processed))
        with open("chunks.json", "w+") as file:
            json.dump(chunks, file, indent=2)

    def search(self, query: str, k: int) -> list[tuple[str, dict[str, Any]]]:
        results = faiss.read_index(str(self.processed))
        query_emd = self.model.encode([query])
        search_results = results.search(query_emd, k)
        with open("chunks.json") as file:
            chunks = json.load(file)
        semantic_results = []
        for idx in search_results[1][0]:
            metadatas = chunks[idx]
            semantic_results.append(
                (
                    metadatas["id"],
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
