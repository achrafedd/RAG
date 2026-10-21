from pathlib import Path
import pickle
import hashlib
from typing import Any
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm


class Chunker:
    """Split repository files into retrieval-friendly chunks with metadata.

    Attributes:
        cache: Directory used for incremental indexing metadata.
        file: Serialized hash file for change detection.
        ext: Supported file extensions for chunking.
        data: Root directory of the indexed codebase.
        chunks: Collected chunk metadata after processing.
    """

    def __init__(self) -> None:
        """Initialize the chunker paths and configuration."""
        self.cache: Path = Path("cache")
        self.file: Path = self.cache / "hashed_files.pkl"
        self.ext: set[str] = {".py", ".md", ".txt"}
        self.path: Path = Path("data/")
        self.data: Path = self.path / "raw/vllm-0.10.1"
        self.chunks: list[dict[str, Any]] = []
        self.max_chunk_size_cache: Path = self.cache / "max_chunk_size.txt"

    def chunk(self, max_chunk_size: int = 2001) -> tuple[str, Any]:
        """Chunk all supported files and return the indexing state.

        Args:
            max_chunk_size: Maximum number of characters allowed in each chunk.

        Returns:
            A tuple containing the indexing state and the chunk list.
        """
        changed = False
        if self.max_chunk_size_cache.exists():
            res = self.max_chunk_size_cache.read_text()
            if int(res) != max_chunk_size:
                self.max_chunk_size_cache.write_text(str(max_chunk_size))
                changed = True

        stat: str = ""
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
        new = {}
        for file in files:
            h = hashlib.sha256()
            h.update(file.read_text().encode())
            new[str(file)] = h.hexdigest()

        if self.file.exists() and not changed:
            with open("cache/hashed_files.pkl", "rb") as f:
                old = pickle.load(f)
            if old == new:
                print("files already indexed")
                return "exit", None
            else:
                changed_files: dict = dict(new.items() - old.items())
                files = [Path(file) for file in changed_files]
            stat = "sync"
        else:
            stat = "build"

        self.cache.mkdir(parents=True, exist_ok=True)
        with open(self.file, "wb+") as f:
            pickle.dump(new, f)

        for file in tqdm(files, desc="Chunking: "):
            splitter = splitters[file.suffix]
            splitted_text: list[Document] = splitter.create_documents(
                [file.read_text()]
            )
            self._save_chunks(
                splitted_text,
                file,
            )
        return stat, self.chunks

    def _save_chunks(self, splitted_text: list[Document], path: Path) -> None:
        """Store chunk metadata for a set of text segments.

        Args:
            splitted_text: The chunks generated from a single file.
            path: Source file path associated with those chunks.
        """
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


if __name__ == "__main__":
    chunker = Chunker()
    chunker.chunk()
