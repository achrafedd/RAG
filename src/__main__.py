import json
from pathlib import Path
from typing import Any

from rich.traceback import install
from rich.progress import Progress
from LLM import Lmachine
import fire
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import bm25s
from Models import *

install()


class RAG(object):
    def __init__(self) -> None:
        self.path: Path = Path("data/")
        self.data: Path = self.path / "raw/vllm-0.10.1/"
        self.processed: Path = self.path / "processed/bm25s_index"
        self.dataset: Path = self.path / "datasets_public/"
        self.output: Path = self.path / "output/"
        self.chunks: list[dict[str, Any]] = []
        self.ext: set[str] = {".py", ".md", ".txt"}
        self.retriever: bm25s.BM25 = bm25s.BM25()
        self.model: Lmachine = Lmachine()

    def index(self, max_chunk_size: int = 2000) -> None:
        config = {
            "chunk_size": max_chunk_size,
            "chunk_overlap": max_chunk_size * 0.05,
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

        with Progress() as progress:
            files = [
                file
                for file in self.data.rglob("*")
                if file.suffix in self.ext
            ]
            task = progress.add_task("Chunking: ", total=len(files))
            for idx, file in enumerate(files, start=1):
                splitter = splitters[file.suffix]
                splitted_text = splitter.split_text(file.read_text())
                self._save_chunks(splitted_text, file)
                progress.update(task, advance=(len(files) * 0.01))

            chunks_tokens = bm25s.tokenize([el["data"] for el in self.chunks])
            self.retriever.index(chunks_tokens)
            self.retriever.save(str(self.processed))
            print(
                f"Ingestion complete! Indexed {len(self.chunks)} chunks under data/processed"
            )
        res = [MinimalSource(**el).model_dump() for el in self.chunks]
        with open("chunks.json", "w") as file:
            json.dump(res, file, indent=4)

    def _save_chunks(self, splitted_text: list[str], path: Path) -> None:
        for text in splitted_text:
            start: int = path.read_text().index(text)
            self.chunks.append(
                {
                    "file_path": str(path),
                    "data": text,
                    "first_character_index": start,
                    "last_character_index": start + len(text),
                }
            )

    def search(self, query: str, k: int = 1) -> MinimalSearchResults:
        retriever = self.retriever.load(
            str(self.processed), show_progress=False
        )
        query_tokens = bm25s.tokenize([query], show_progress=False)
        results, _ = retriever.retrieve(query_tokens, k=k, show_progress=False)
        with open("chunks.json") as file:
            chunks = json.load(file)

        retrieved_sources: list[MinimalSearchResults] = []
        for idx in results[0]:
            retrieved_sources.append(MinimalSource(**chunks[idx]))

        qid = str(uuid.uuid4())
        search_result = MinimalSearchResults(
            question_id=qid,
            question=query,
            retrieved_sources=retrieved_sources,
        )
        return search_result

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        search_result: list[MinimalSearchResults] = []
        with open(dataset_path) as file:
            dataset = json.load(file)

            for question in dataset["rag_questions"]:
                search_result.append(self.search(question["question"], k))
            student_search = StudentSearchResults(
                search_results=search_result, k=k
            )

        full_path = (
            self.output
            / f"{save_directory.strip('/')}/student_search_results.json"
        )
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w+") as file:
            json.dump(student_search.model_dump(), file, indent=4)

    def answer(self, query: str, k: int = 1) -> MinimalAnswer:
        search_result: MinimalSearchResults = self.search(query, k)
        augmented_query = (
            "system:\n"
            "You are a helpful assistant. Answer the question using ONLY the context below.\n"
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
                augmented_query += f"[ {idx} ] {data}.\n"
                augmented_query += f"(source: {result.file_path} {result.first_character_index} {result.last_character_index}).\n\n"

        augmented_query += f"Question: {query}\n\n"
        augmented_query += "Answer, citing sources by number where relevant."
        content = self.model.dwiha(augmented_query)
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
