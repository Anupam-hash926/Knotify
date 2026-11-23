"""Retrieval evaluation helpers: run the three frozen methods on frozen queries and build
method-blind judgement pools. Metrics are computed in the Day 4 evaluation runner."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from knotify.contracts import Issue
from knotify.retrieval import (
    BM25Retriever,
    DenseRetriever,
    eligible_history,
    reciprocal_rank_fusion,
)


def method_rankings(
    query: Issue,
    history: Sequence[Issue],
    bm25: BM25Retriever,
    dense: DenseRetriever,
    *,
    rrf_k: int = 60,
) -> dict[str, list[str]]:
    eligible = eligible_history(query, history)
    ranked_bm25 = bm25.rank(query, eligible)
    ranked_dense = dense.rank(query, eligible)
    fused = reciprocal_rank_fusion({"bm25": ranked_bm25, "dense": ranked_dense}, k=rrf_k)
    return {
        "bm25": [i.issue_id for i, _ in ranked_bm25],
        "dense": [i.issue_id for i, _ in ranked_dense],
        "hybrid": [i.issue_id for i, _ in fused][: bm25.depth],
    }


def blind_pool(
    rankings: dict[str, list[str]], known: Sequence[str], *, depth: int = 5, seed: int = 42
) -> list[str]:
    """Union of each method's top ``depth`` plus known targets, shuffled (method-blind)."""
    pool = sorted({x for ranked in rankings.values() for x in ranked[:depth]} | set(known))
    random.Random(seed).shuffle(pool)
    return pool


def stable_seed(text: str) -> int:
    """Deterministic across processes (built-in hash() of str is randomised)."""
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def pool_record(
    query: Issue, rankings: dict[str, list[str]], known: Sequence[str]
) -> dict[str, Any]:
    return {
        "query": query.issue_id,
        "known_duplicates": list(known),
        "pool": blind_pool(rankings, known, seed=stable_seed(query.issue_id)),
    }
