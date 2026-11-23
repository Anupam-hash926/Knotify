"""Historical eligibility, BM25 retrieval and the retriever boundary (dense/RRF: Day 2)."""

from collections.abc import Sequence
from typing import Protocol

from knotify.contracts import Issue, RetrievedIssue
from knotify.data import model_text
from knotify.errors import ComponentUnavailable
from knotify.features import prepare, tokenize


def eligible_history(query: Issue, history: Sequence[Issue]) -> tuple[Issue, ...]:
    """Filter BEFORE BM25 statistics or dense ranking. Dates use aware timestamps."""
    records: dict[str, Issue] = {}
    for issue in history:
        if (
            issue.repository == query.repository
            and issue.issue_id != query.issue_id
            and issue.created_at < query.created_at
        ):
            if issue.issue_id in records and records[issue.issue_id] != issue:
                raise ValueError("Conflicting snapshots for one issue in the frozen corpus.")
            records[issue.issue_id] = issue
    return tuple(sorted(records.values(), key=lambda item: (item.created_at, item.issue_id)))


class Retriever(Protocol):
    def search(
        self, query: Issue, eligible_corpus: Sequence[Issue], *, top_k: int
    ) -> tuple[RetrievedIssue, ...]: ...


class UnavailableRetriever:
    def search(
        self, query: Issue, eligible_corpus: Sequence[Issue], *, top_k: int
    ) -> tuple[RetrievedIssue, ...]:
        raise ComponentUnavailable("BM25, dense retrieval and RRF are not connected yet.")


def _bm25(docs: list[list[str]]):
    """rank_bm25 Okapi BM25 with the Lucene IDF ``log(1 + (N - n + 0.5) / (n + 0.5))``.

    The Okapi IDF is zero/negative for terms in half or more of the documents, which empties
    results for small early-history corpora; the Lucene form is always positive."""
    import math

    from rank_bm25 import BM25Okapi

    class LuceneIdfBM25(BM25Okapi):
        def _calc_idf(self, nd):
            for word, freq in nd.items():
                self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    return LuceneIdfBM25(docs)


class BM25Retriever:
    """BM25 over the *eligible* corpus only: statistics are rebuilt per query from the
    same-repository, strictly earlier issues, so future issues never influence IDF or ranks.
    Token lists are cached per immutable issue snapshot."""

    method = "bm25"

    def __init__(self, *, depth: int = 50, query_transform=None):
        self.depth = depth
        self.query_transform = query_transform or (lambda issue: model_text(issue))
        self._tokens: dict[tuple[str, str], list[str]] = {}

    def _doc_tokens(self, issue: Issue) -> list[str]:
        key = (issue.issue_id, issue.updated_at.isoformat() if issue.updated_at else "")
        if key not in self._tokens:
            self._tokens[key] = tokenize(prepare(model_text(issue)))
        return self._tokens[key]

    def rank(self, query: Issue, eligible_corpus: Sequence[Issue]) -> list[tuple[Issue, float]]:
        """Up to ``depth`` (issue, score) pairs with positive score, deterministic ties."""
        if not eligible_corpus:
            return []
        query_tokens = tokenize(prepare(self.query_transform(query)))
        if not query_tokens:
            return []
        docs = [self._doc_tokens(issue) or ["\x00"] for issue in eligible_corpus]
        scores = _bm25(docs).get_scores(query_tokens)
        order = sorted(
            range(len(eligible_corpus)), key=lambda i: (-scores[i], eligible_corpus[i].issue_id)
        )
        return [
            (eligible_corpus[i], float(scores[i])) for i in order[: self.depth] if scores[i] > 0
        ]

    def search(
        self, query: Issue, eligible_corpus: Sequence[Issue], *, top_k: int
    ) -> tuple[RetrievedIssue, ...]:
        ranked = self.rank(query, eligible_corpus)[:top_k]
        return tuple(
            RetrievedIssue(issue=issue, rank=rank, method_scores={self.method: score})
            for rank, (issue, score) in enumerate(ranked, 1)
        )


class DenseRetriever:
    """Cosine similarity over cached normalised embeddings, restricted to the eligible corpus
    *before* ranking (exact FAISS IndexFlatIP with an ID selector, or exact NumPy)."""

    method = "dense"

    def __init__(
        self, cache, *, encoder=None, depth: int = 50, query_transform=None, backend: str = "auto"
    ):
        import numpy as np

        self.cache, self.encoder, self.depth = cache, encoder, depth
        self.query_transform = query_transform
        self._np = np
        self.backend = "numpy"
        self._faiss_index = None
        if backend in ("auto", "faiss"):
            try:
                import faiss

                index = faiss.IndexFlatIP(cache.vectors.shape[1])
                index.add(np.ascontiguousarray(cache.vectors, dtype=np.float32))
                self._faiss, self._faiss_index, self.backend = faiss, index, "faiss"
            except ImportError:
                if backend == "faiss":
                    raise

    def query_vector(self, query: Issue):
        from knotify.embeddings import encode

        if self.query_transform is None:
            cached = self.cache.vector_for(query)
            if cached is not None:
                return cached
        if self.encoder is None:
            raise ComponentUnavailable(
                "Dense search needs the local embedding model for text that is not cached."
            )
        text = self.query_transform(query) if self.query_transform else model_text(query)
        from knotify.features import prepare

        return encode(self.encoder, [prepare(text)])[0]

    def rank(self, query: Issue, eligible_corpus: Sequence[Issue]) -> list[tuple[Issue, float]]:
        np = self._np
        rows, issues = [], []
        for issue in eligible_corpus:
            row = self.cache.index.get(issue.issue_id)
            if row is not None:
                rows.append(row)
                issues.append(issue)
        if not rows:
            return []
        vector = np.asarray(self.query_vector(query), dtype=np.float32)
        by_row = dict(zip(rows, issues, strict=True))
        fetch = min(len(rows), self.depth + 20)
        if self._faiss_index is not None:
            faiss = self._faiss
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.asarray(rows, "int64")))
            scores, found = self._faiss_index.search(vector[None, :], fetch, params=params)
            pairs = [
                (by_row[int(r)], float(s))
                for s, r in zip(scores[0], found[0], strict=True)
                if r >= 0
            ]
        else:
            sims = self.cache.vectors[rows] @ vector
            top = np.argsort(-sims, kind="stable")[:fetch]
            pairs = [(issues[i], float(sims[i])) for i in top]
        pairs.sort(key=lambda p: (-round(p[1], 6), p[0].issue_id))
        return pairs[: self.depth]

    def search(self, query: Issue, eligible_corpus: Sequence[Issue], *, top_k: int):
        return tuple(
            RetrievedIssue(issue=issue, rank=rank, method_scores={self.method: score})
            for rank, (issue, score) in enumerate(self.rank(query, eligible_corpus)[:top_k], 1)
        )


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[Issue, float]]], *, k: int = 60
) -> list[tuple[Issue, dict[str, float]]]:
    """``sum(1 / (k + rank))`` with one-based ranks; absent candidates contribute 0;
    ties broken by issue ID."""
    fused: dict[str, dict[str, float]] = {}
    issues: dict[str, Issue] = {}
    for method, ranked in rankings.items():
        for rank, (issue, score) in enumerate(ranked, 1):
            entry = fused.setdefault(issue.issue_id, {"rrf": 0.0})
            entry["rrf"] += 1.0 / (k + rank)
            entry[method] = score
            entry[f"{method}_rank"] = float(rank)
            issues[issue.issue_id] = issue
    order = sorted(fused, key=lambda key: (-fused[key]["rrf"], key))
    return [(issues[key], fused[key]) for key in order]


class HybridRetriever:
    """BM25 + dense fused by RRF. If dense search is unavailable for a query (e.g. pasted
    text without the local model), results are BM25-only and carry no ``rrf`` score."""

    method = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever | None, *, rrf_k: int = 60):
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k

    def rankings(self, query: Issue, eligible_corpus: Sequence[Issue]) -> dict[str, list]:
        result = {"bm25": self.bm25.rank(query, eligible_corpus)}
        if self.dense is not None:
            try:
                result["dense"] = self.dense.rank(query, eligible_corpus)
            except ComponentUnavailable:
                pass
        return result

    def search(self, query: Issue, eligible_corpus: Sequence[Issue], *, top_k: int):
        rankings = self.rankings(query, eligible_corpus)
        if "dense" not in rankings:
            return self.bm25.search(query, eligible_corpus, top_k=top_k)
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)[:top_k]
        return tuple(
            RetrievedIssue(issue=issue, rank=rank, method_scores=scores)
            for rank, (issue, scores) in enumerate(fused, 1)
        )
