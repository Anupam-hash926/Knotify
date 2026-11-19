"""Historical eligibility and retriever boundary; ranking implementations are not built yet."""

from collections.abc import Sequence
from typing import Protocol

from knotify.contracts import Issue, RetrievedIssue
from knotify.errors import ComponentUnavailable


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
