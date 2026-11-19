"""Small metric primitives, not an implemented experiment runner."""

from collections.abc import Sequence


def retrieval_metrics(
    ranked_ids: Sequence[str], relevant_ids: set[str], *, k: int = 5
) -> dict[str, float | None]:
    """Recall against known positives; undefined for queries without known positives."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Rankings must be deduplicated at issue level before evaluation.")
    if not relevant_ids:
        return {"recall": None, "hit": None, "reciprocal_rank": None}
    hits = [rank for rank, key in enumerate(ranked_ids[:k], 1) if key in relevant_ids]
    return {
        "recall": len(hits) / len(relevant_ids),
        "hit": float(bool(hits)),
        "reciprocal_rank": 1 / hits[0] if hits else 0.0,
    }


def selective_metrics(correct: Sequence[bool], accepted: Sequence[bool]) -> dict:
    if len(correct) != len(accepted):
        raise ValueError("Predictions and acceptance flags must have equal length.")
    count = sum(accepted)
    return {
        "total": len(correct),
        "accepted": count,
        "coverage": count / len(correct) if correct else None,
        "accepted_accuracy": (
            sum(ok and keep for ok, keep in zip(correct, accepted, strict=True)) / count
            if count
            else None
        ),
    }
