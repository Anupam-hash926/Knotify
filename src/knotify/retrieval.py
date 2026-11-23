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
