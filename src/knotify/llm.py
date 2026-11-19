"""Future optional drafting adapter. No provider SDK, API call or credentials loaded here."""

from typing import Protocol

from knotify.contracts import ChecklistItem, Issue, RetrievedIssue, Suggestion


class DraftProvider(Protocol):
    def draft(
        self,
        issue: Issue,
        checklist: tuple[ChecklistItem, ...],
        related: tuple[RetrievedIssue, ...],
    ) -> tuple[Suggestion, ...]: ...
