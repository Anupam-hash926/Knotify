"""Completeness interface. Missing implementation must not become 'all fields missing'."""

from typing import Protocol

from knotify.contracts import ChecklistItem, ClassificationResult, Issue
from knotify.errors import ComponentUnavailable


class Checklist(Protocol):
    def check(
        self, issue: Issue, classification: ClassificationResult
    ) -> tuple[ChecklistItem, ...]: ...


class UnavailableChecklist:
    def check(
        self, issue: Issue, classification: ClassificationResult
    ) -> tuple[ChecklistItem, ...]:
        raise ComponentUnavailable("Checklist rules and applicability have not been implemented.")
