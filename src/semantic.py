from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Semantic:
    """Semantic retrieval layer built on sentence embeddings and ChromaDB."""

    def __init__(self) -> None:
        """Initialize the persistent vector database and
        sentence embedding model."""
        self.path: Path = Path("data/")
        self.processed: Path = self.path / "processed/chromab"
        self.cache: Path = Path("cache/emb.npy")
        self.client = chromadb.PersistentClient(self.processed)
        self.collection = self.client.get_or_create_collection(
            "collection",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 100,
                "hnsw:construction_ef": 200,
            },
        )
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def index(
        self, documents: list[str], chunks: list[dict[str, Any]], changed: bool
    ) -> None:
        """Create or update the semantic vector index for all chunk documents.

        Args:
            documents: Text content of each chunk.
            chunks: Chunk metadata including IDs and file locations.
            changed: Flag indicating whether the corpus layout changed.
        """
        embeddings = self.model.encode(documents, show_progress_bar=True)
        ids = [chunk["id"] for chunk in chunks]
        metadatas: Any = [
            {
                "file_path": chunk["file_path"],
                "first_character_index": chunk["first_character_index"],
                "last_character_index": chunk["last_character_index"],
            }
            for chunk in chunks
        ]
        max_bs = self.client.get_max_batch_size()
        for start in tqdm(range(0, len(chunks), max_bs), "Saving Vetors: "):
            end = start + max_bs
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                documents=documents[start:end],
            )

    def search(self, query: str, k: int) -> list[tuple[str, dict[str, Any]]]:
        """Find the top-k semantically similar chunks for a query.

        Args:
            query: User query to embed and search for.
            k: Number of nearest neighbors to retrieve.

        Returns:
            A list of tuples containing chunk IDs and metadata for
            semantic matches.
        """
        query_emd = self.model.encode([query])
        search_results = self.collection.query(
            query_embeddings=query_emd, n_results=k
        )
        semantic_results = []
        for idx in range(len(search_results["ids"][0])):
            res = search_results["metadatas"]
            if res is None:
                continue
            metadatas = res[0][idx]
            semantic_results.append(
                (
                    search_results["ids"][0][idx],
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
