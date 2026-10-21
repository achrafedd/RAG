from pathlib import Path
import pickle
from typing import Any


def save_cache_query(directory: Path, querys: dict[str, list]) -> None:
    """Persist cached query results to disk.

    Args:
        directory: Target file path where the cached query
        dictionary will be stored.
        querys: Mapping of queries to their serialized retrieval results.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    with open(directory, "wb+") as file:
        pickle.dump(querys, file)


def load_cache_query(directory: Path) -> Any:
    """Load cached query results from disk.

    Args:
        directory: Path to the serialized cache file.

    Returns:
        The cached query dictionary stored on disk.
    """
    with open(directory, "rb") as file:
        querys = pickle.load(file)
    return querys
