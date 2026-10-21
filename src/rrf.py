from typing import Any


def reciprocal_rank_fusion(
    results_list: list[list[tuple[str, dict[str, Any]]]], k: int = 60
) -> list[Any]:
    """Combine multiple ranked retrieval lists using reciprocal rank fusion.

    Args:
        results_list: A list of ranked result lists from different
        retrieval systems.
        k: Constant used by RRF to smooth rankings and avoid
        domination by a single system.

    Returns:
        A single list of documents sorted by the fused score.
    """
    doc_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict[str, Any]] = {}

    for results in results_list:
        for rank, (doc_id, doc) in enumerate(results, start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + k)
            doc_lookup.setdefault(doc_id, doc)

    ranked_ids = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)
    return [doc_lookup[doc_id] for doc_id in ranked_ids]
