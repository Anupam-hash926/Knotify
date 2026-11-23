"""Evidence-linked template questions and reference validation (no LLM).

Each *missing* checklist item yields one follow-up question citing ``checklist:<item_id>``.
``reference_valid`` is a mechanical check (cited items exist with status ``missing``; cited
issues are in the eligible retrieved set). It is not semantic support: ``semantic_support``
stays ``not_checked`` until a person reviews the suggestion.
"""

from __future__ import annotations

from collections.abc import Sequence

from knotify.checklist import checklist_applies
from knotify.contracts import (
    ChecklistItem,
    ChecklistItemId,
    ClassificationResult,
    RetrievedIssue,
    Suggestion,
)

VERSION_HINTS = {
    "scikit-learn/scikit-learn": " The output of `sklearn.show_versions()` covers this.",
    "pandas-dev/pandas": " The output of `pd.show_versions()` covers this.",
}

TEMPLATES = {
    ChecklistItemId.REPRODUCTION: (
        "Could you share a minimal, self-contained code example (or numbered steps) that "
        "reproduces the problem?"
    ),
    ChecklistItemId.EXPECTED: "What result or behaviour did you expect?",
    ChecklistItemId.ACTUAL: "What happened instead? Please describe the output you observed.",
    ChecklistItemId.ENVIRONMENT: "Which operating system and Python version are you using?",
    ChecklistItemId.VERSION: "Which version of the library (and its main dependencies) are you "
    "using?{hint}",
    ChecklistItemId.LOGS: "Could you paste the full error message or traceback?",
}

CONDITIONAL_PREFIX = "If this is a bug report: "


def evidence_id(item: ChecklistItemId) -> str:
    return f"checklist:{item.value}"


def template_suggestions(
    checklist: Sequence[ChecklistItem],
    classification: ClassificationResult,
    repository: str = "",
) -> tuple[Suggestion, ...]:
    if not checklist_applies(classification):
        return ()
    conditional = classification.deferred or classification.predicted_label is None
    suggestions = []
    for item in checklist:
        if item.status != "missing":
            continue
        text = TEMPLATES[item.item_id].format(hint=VERSION_HINTS.get(repository, ""))
        if conditional:
            text = CONDITIONAL_PREFIX + text[0].lower() + text[1:]
        suggestions.append(
            Suggestion(text=text, evidence_ids=(evidence_id(item.item_id),), source="template")
        )
    return tuple(suggestions)


def validate_references(
    suggestions: Sequence[Suggestion],
    checklist: Sequence[ChecklistItem],
    related: Sequence[RetrievedIssue],
) -> tuple[Suggestion, ...]:
    """Set ``reference_valid`` for each suggestion; unknown evidence IDs make it invalid."""
    missing = {evidence_id(item.item_id) for item in checklist if item.status == "missing"}
    issues = {f"issue:{item.issue.issue_id}" for item in related}
    allowed = missing | issues
    return tuple(
        s.model_copy(
            update={"reference_valid": bool(s.evidence_ids) and set(s.evidence_ids) <= allowed}
        )
        for s in suggestions
    )


def compose_reply(suggestions: Sequence[Suggestion]) -> str:
    """Draft comment for the maintainer to edit; never posted automatically."""
    valid = [s for s in suggestions if s.reference_valid]
    if not valid:
        return ""
    lines = ["Thanks for the report! A few details would help us look into this:"]
    for s in valid:
        text = s.text.removeprefix(CONDITIONAL_PREFIX)
        lines.append(f"- {text[0].upper()}{text[1:]}")
    if any(s.text.startswith(CONDITIONAL_PREFIX) for s in valid):
        lines.insert(0, "(Draft — category not confirmed; relevant if this is a bug report.)")
    return "\n".join(lines)
