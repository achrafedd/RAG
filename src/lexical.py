from pathlib import Path
from typing import Any
import bm25s
import json


class Lexical:
    """Lexical retrieval component based on the BM25 ranking algorithm."""

    def __init__(self) -> None:
        """Initialize the BM25 index path and retriever configuration."""
        self.path: Path = Path("data/")
        self.processed: Path = self.path / "processed/bm25s_index"
        self.retriever: bm25s.BM25 = bm25s.BM25()
        self.cache: Path = Path("cache/bm25.pkl")

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """Build a BM25 index from the provided chunk list.

        Args:
            chunks: Chunk metadata and content to index.
        """
        documents = [chunk["content"] for chunk in chunks]
        documents_tokens = bm25s.tokenize(documents)
        retriever = bm25s.BM25(corpus=chunks)
        retriever.index(documents_tokens)
        retriever.save(self.processed)
        with open(self.processed / "corpus.json", "w+") as file:
            json.dump(chunks, file, indent=4)

    def reindex(self, chunks: list[dict[str, Any]]) -> None:
        """Refresh the index for changed chunks while
        preserving unchanged ones.

        Args:
            chunks: Newly generated chunks that should
            replace outdated entries.
        """
        with open(self.processed / "corpus.json") as file:
            corpus = json.load(file)
        changed_files = set([chunk["file_path"] for chunk in chunks])
        new_corpus = list(
            filter(
                lambda chunk: chunk["file_path"] not in changed_files, corpus
            )
        )
        new_corpus += chunks
        documents = [chunk["content"] for chunk in new_corpus]
        documents_tokens = bm25s.tokenize(documents)
        retriever = bm25s.BM25(corpus=new_corpus)
        retriever.index(documents_tokens)
        retriever.save(self.processed)
        with open(self.processed / "corpus.json", "w+") as file:
            json.dump(chunks, file, indent=4)

    def search(self, query: str, k: int) -> list[tuple[str, dict[str, Any]]]:
        """Search the BM25 index for the top-k relevant chunks.

        Args:
            query: User query to rank against the indexed corpus.
            k: Maximum number of results to return.

        Returns:
            A list of tuples containing document IDs and
            metadata for top-ranked chunks.
        """
        retriever = self.retriever.load(
            str(self.processed), load_corpus=True, mmap=True
        )
        query_tokens = bm25s.tokenize([query])
        results, _ = retriever.retrieve(query_tokens, k=k)
        lexical_results = []
        for metadatas in results[0]:
            lexical_results.append(
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
        return lexical_results
