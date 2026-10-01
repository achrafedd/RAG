import json
from pathlib import Path
from typing import Any
import uuid
from rich import print


from lexical import Lexical
from rrf import reciprocal_rank_fusion
from semantic import Semantic

from tqdm import tqdm

from langchain_core.documents.base import Document
from rich.progress import Progress
from LLM import Lmachine
import fire
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from Models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    RagDataset,
    StudentSearchResultsAndAnswer,
)


class RAG(object):
    def __init__(self) -> None:
        self.path: Path = Path("data/")
        self.data: Path = self.path / "raw/vllm-0.10.1/"
        self.output: Path = self.path / "output/"
        self.chunks: list[dict[str, Any]] = []
        self.ext: set[str] = {".py", ".md", ".txt"}
        self.model: Lmachine = Lmachine()
        self.lexical: Lexical = Lexical()
        self.semantic: Semantic = Semantic()

    def index(self, max_chunk_size: int = 2000) -> None:
        config = {
            "chunk_size": max_chunk_size,
            "chunk_overlap": int(max_chunk_size * 0.05),
            "add_start_index": True,
        }
        splitters: dict[str, RecursiveCharacterTextSplitter] = {
            ".py": RecursiveCharacterTextSplitter.from_language(
                Language.PYTHON, **config
            ),
            ".txt": RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN, **config
            ),
            ".md": RecursiveCharacterTextSplitter.from_language(
                Language.MARKDOWN, **config
            ),
        }

        files = [
            file for file in self.data.rglob("*") if file.suffix in self.ext
        ]
        for file in tqdm(files, desc="Chunking: "):
            splitter = splitters[file.suffix]
            splitted_text: list[Document] = splitter.create_documents(
                [file.read_text()]
            )
            self._save_chunks(
                splitted_text,
                file,
            )

        documents: list[str] = [doc["content"] for doc in self.chunks]
        self.lexical.index(documents, self.chunks)

        self.semantic.index(documents, self.chunks)

        print(
            f"Ingestion complete! Indexed {len(self.chunks)} "
            "chunks under data/processed"
        )

    def _save_chunks(self, splitted_text: list[Document], path: Path) -> None:
        for i, document in enumerate(splitted_text):
            start: int = document.metadata["start_index"]
            content: str = document.page_content
            _id: str = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"{str(path)}:{content}:{start}"
                )
            )
            self.chunks.append(
                {
                    "id": _id,
                    "file_path": str(path),
                    "content": content,
                    "first_character_index": start,
                    "last_character_index": start + len(content),
                }
            )

    def search(self, query: str, k: int = 1) -> MinimalSearchResults:
        semantic_results = self.semantic.search(query, k)
        lexical_results = self.lexical.search(query, k)
        combined_results = reciprocal_rank_fusion(
            [semantic_results, lexical_results]
        )
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
        return search_result

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        search_result: list[MinimalSearchResults] = []
        with open(dataset_path) as file:
            dataset: RagDataset = RagDataset(**json.load(file))

            for question in tqdm(dataset.rag_questions, "Searching: "):
                res = self.search(question.question, k)
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
        search_result: MinimalSearchResults = self.search(query, k)
        augmented_query = (
            "system:\n"
            "You are a helpful assistant. Answer the question "
            "using ONLY the context below.\n"
            "If the answer isn't in the context, say you don't know.\n"
            "user:\n"
            "Context:\n\n"
        )
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
        augmented_query += "Answer, citing sources by number where relevant."
        content = self.model.dwiha(augmented_query)
        if content is None:
            raise ValueError("Ok bon8 a khoya!")
        return MinimalAnswer(answer=content, **search_result.model_dump())

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        answers: list[MinimalAnswer] = []
        with open(student_search_results_path) as file:
            search_results = StudentSearchResults(**json.load(file))
            with Progress() as progress:
                task3 = progress.add_task(
                    "answering questions: ",
                    total=len(search_results.search_results),
                )
                for res in search_results.search_results:
                    answers.append(self.answer(res.question, search_results.k))
                    progress.update(
                        task3,
                        advance=len(search_results.search_results) * 0.01,
                    )

        search_results_and_answer = StudentSearchResultsAndAnswer(
            search_results=answers, k=search_results.k
        )

        path = Path(save_directory) / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w+") as file:
            json.dump(search_results_and_answer.model_dump(), file, indent=4)

        print(f"Saved Student_search_results_and_answer to {path}")


if __name__ == "__main__":
    fire.Fire(RAG)
