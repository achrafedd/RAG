from pathlib import Path
from typing import Any
import bm25s


class Lexical:
    def __init__(self) -> None:
        self.path: Path = Path("data/")
        self.processed: Path = self.path / "processed/bm25s_index"
        self.retriever: bm25s.BM25 = bm25s.BM25()

    def index(
        self, documents: list[str], chunks: list[dict[str, Any]]
    ) -> None:
        documents_tokens = bm25s.tokenize(documents)
        retriever = bm25s.BM25(corpus=chunks)
        retriever.index(documents_tokens)
        retriever.save(self.processed)

    def search(self, query: str, k: int) -> list[tuple[str, dict[str, Any]]]:
        retriever = self.retriever.load(str(self.processed), load_corpus=True)
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
