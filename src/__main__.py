import hashlib
import json
import uuid
from pathlib import Path

import fire
from tqdm import tqdm

from .caching import load_cache_query, save_cache_query
from .chunking import Chunker
from .evaluator import evaluate
from .lexical import Lexical
from .LLM import LLM
from .Models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .rrf import reciprocal_rank_fusion
from .semantic import Semantic


class RAG(object):
    """Main orchestrator for the repository RAG pipeline.

    The class coordinates indexing, retrieval, answer generation,
    and evaluation
    for codebase questions.
    """

    def __init__(self) -> None:
        """Initialize the project paths and retrieval components."""
        self.path: Path = Path("data/")
        self.data: Path = self.path / "raw/vllm-0.10.1"
        self.output: Path = self.path / "output/"
        self.chunker: Chunker = Chunker()
        self.model: LLM = LLM()
        self.lexical: Lexical = Lexical()
        self.semantic: Semantic = Semantic()
        self.is_direct: bool = True
        self.cache_query: Path = Path("cache/querys.pkl")
        self.cache_meta: Path = Path("cache/meta.json")
        self.changed: bool = False

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index the repository corpus and persist the processed
        retrieval data.

        Args:
            max_chunk_size: Maximum length of an indexed chunk in characters.
        """
        state, chunks = self.chunker.chunk(max_chunk_size)
        if state == "exit":
            return
        documents: list[str] = [doc["content"] for doc in chunks]
        h = hashlib.sha256("".join(documents).encode())
        meta = {"max_chunk_size": max_chunk_size, "corpus": h.hexdigest()}
        if self.cache_meta.exists():
            with open(self.cache_meta) as file:
                loaded_meta = json.load(file)
            if meta != loaded_meta:
                self.changed = True
        if state == "build":
            self.lexical.index(chunks)
        elif state == "sync":
            self.lexical.reindex(chunks)
        self.semantic.index(documents, chunks, self.changed)

        print(
            f"Ingestion complete! Indexed {len(chunks)} "
            "chunks under data/processed"
        )

    def search(self, query: str, k: int = 1) -> MinimalSearchResults | None:
        """Retrieve the top-k source locations for a single
        natural-language query.

        Args:
            query: The question to search for.
            k: Maximum number of sources to return.

        Returns:
            A `MinimalSearchResults` object when running programmatically.
            If the method is used in direct CLI mode,
            it prints results instead.
        """
        querys = {}
        if self.cache_query.exists() and not self.changed:
            querys = load_cache_query(self.cache_query)
            if query in querys:
                res = MinimalSearchResults(**querys[query])
                if self.is_direct:
                    for source in res.retrieved_sources:
                        print(
                            f"{source.file_path}"
                            f"[{source.first_character_index}:"
                            f"{source.last_character_index}]"
                        )
                    return None
                else:
                    return res
        semantic_results = self.semantic.search(query, k * 2)
        lexical_results = self.lexical.search(query, k * 2)
        combined_results = reciprocal_rank_fusion(
            [semantic_results, lexical_results]
        )
        if k == 0:
            exit()
        retrieved_sources: list[MinimalSource] = []
        for metadatas in combined_results[:k]:
            retrieved_sources.append(
                MinimalSource(
                    file_path=metadatas["file_path"],
                    first_character_index=metadatas["first_character_index"],
                    last_character_index=metadatas["last_character_index"],
                )
            )

        q_id = uuid.uuid4()
        search_result = MinimalSearchResults(
            question_id=str(q_id),
            question=query,
            retrieved_sources=retrieved_sources,
        )
        querys[query] = search_result.model_dump()
        save_cache_query(self.cache_query, querys)
        if self.is_direct:
            for source in retrieved_sources:
                print(
                    f"{source.file_path}"
                    f"[{source.first_character_index}:"
                    f"{source.last_character_index}]"
                )
            return None
        else:
            return search_result

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        """Run retrieval over a dataset and persist the resulting JSON output.

        Args:
            dataset_path: Path to the dataset file containing questions.
            k: Number of sources to retrieve per question.
            save_directory: Directory where the search results JSON
            should be saved.
        """
        self.is_direct = False
        search_result: list[MinimalSearchResults] = []
        with open(dataset_path) as file:
            dataset: RagDataset = RagDataset(**json.load(file))

            for question in tqdm(dataset.rag_questions, "Searching: "):
                res = self.search(question.question, k)
                if res is None:
                    return
                res.question_id = question.question_id
                search_result.append(res)
            student_search = StudentSearchResults(
                search_results=search_result, k=k
            )

        full_path = Path(f"{save_directory}/{dataset_path.split('/')[-1]}")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w+") as file:
            json.dump(student_search.model_dump(), file, indent=4)
        print(f"The search dataset is saved on {str(full_path)}")

    def answer(self, query: str, k: int = 1) -> MinimalAnswer:
        """Generate a grounded answer for a question using retrieved snippets.

        Args:
            query: The question text to answer.
            k: Number of source snippets to include as context.

        Returns:
            A `MinimalAnswer` object containing the retrieved sources
            and final answer.

        Raises:
            ValueError: If the model does not return any content.
        """
        if k == 0:
            exit()
        self.is_direct = False
        search_result: MinimalSearchResults | None = self.search(query, k)
        augmented_query = (
            "system:\n"
            "You are a helpful assistant. Answer the question "
            "using ONLY the context below.\n"
            "If the answer isn't in the context, say you don't know.\n"
            "user:\n"
            "Context:\n\n"
        )
        if search_result is None:
            return
        for idx, result in enumerate(search_result.retrieved_sources, start=1):
            with open(result.file_path, errors="replace") as file:
                file.seek(result.first_character_index)
                data = file.read(
                    result.last_character_index - result.first_character_index
                )
                augmented_query += (
                    f"[ {idx} ] {data}.\n(source:"
                    f" {result.file_path} {result.first_character_index}"
                    f" {result.last_character_index}).\n\n"
                )

        augmented_query += f"Question: {query}\n\n"
        augmented_query += "Answer: "
        content = self.model.generate(augmented_query)
        if content is None:
            raise ValueError("Ok bon8 a khoya!")
        return MinimalAnswer(answer=content, **search_result.model_dump())

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for every question in a retrieval JSON result file.

        Args:
            student_search_results_path: Path to the saved retrieval dataset.
            save_directory: Directory where the answer dataset
            JSON should be written.
        """
        answers: list[MinimalAnswer] = []
        with open(student_search_results_path) as file:
            search_results = StudentSearchResults(**json.load(file))
            for res in search_results.search_results:
                answers.append(self.answer(res.question, search_results.k))

        search_results_and_answer = StudentSearchResultsAndAnswer(
            search_results=answers, k=search_results.k
        )

        path = Path(save_directory) / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w+") as file:
            json.dump(search_results_and_answer.model_dump(), file, indent=4)

        print(f"Saved Student_search_results_and_answer to {path}")

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Compute recall@k for the current retrieval output against a dataset.

        Args:
            student_search_results_path: Path to the
            student-generated search file.
            dataset_path: Path to the ground-truth dataset.
        """
        k_values = [1, 3, 5, 10]

        for k in k_values:
            evaluate(student_search_results_path, dataset_path, k)


if __name__ == "__main__":
    fire.Fire(RAG)
