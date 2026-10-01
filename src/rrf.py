from typing import Any


def reciprocal_rank_fusion(
    results_list: list[list[tuple[str, dict[str, Any]]]], k: int = 60
):
    doc_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict[str, Any]] = {}

    for results in results_list:
        for rank, (doc_id, doc) in enumerate(results):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + k + 1)
            doc_lookup.setdefault(doc_id, doc)  # first-seen wins

    ranked_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
    return [doc_lookup[doc_id] for doc_id in ranked_ids]
