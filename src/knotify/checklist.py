"""Deterministic missing-information checklist (rule version ``checklist-v1``).

Six items: reproduction steps, expected behaviour, actual behaviour, environment, version and
error logs. A template heading alone is never "present": the text under it must be real content
(HTML comments, ``_No response_`` and similar placeholders are removed first). Error logs are
conditional: they apply only when the report describes an error, crash, exception or warning.

Applicability by category: items are checked when the predicted category is ``bug`` or when the
category is deferred/unknown (advice is then phrased conditionally). An *accepted* non-bug
prediction makes every item ``not_applicable``.
"""

from __future__ import annotations

import re
from typing import Protocol

from knotify.contracts import (
    Category,
    ChecklistItem,
    ChecklistItemId,
    ClassificationResult,
    Issue,
)
from knotify.errors import ComponentUnavailable

RULE_VERSION = "checklist-v1"


class Checklist(Protocol):
    def check(
        self, issue: Issue, classification: ClassificationResult
    ) -> tuple[ChecklistItem, ...]: ...


class UnavailableChecklist:
    def check(
        self, issue: Issue, classification: ClassificationResult
    ) -> tuple[ChecklistItem, ...]:
        raise ComponentUnavailable("Checklist rules and applicability have not been implemented.")


# ----------------------------------------------------------------------------- parsing
_COMMENT = re.compile(r"<!--.*?(?:-->|$)", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$", re.MULTILINE)
_PLACEHOLDER = re.compile(
    r"^\s*(?:_?no response_?|n/?a|none|nothing|tbd|todo|\.\.\.|-|same as above)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _clean(text: str) -> str:
    text = _COMMENT.sub(" ", text or "")
    return _PLACEHOLDER.sub(" ", text)


def sections(body: str) -> dict[str, str]:
    """Map lower-cased heading -> cleaned content; text before any heading is under ''."""
    body = _COMMENT.sub(" ", body or "")
    parts: dict[str, str] = {}
    matches = list(_HEADING.finditer(body))
    start = 0
    current = ""
    for match in matches:
        parts[current] = parts.get(current, "") + body[start : match.start()]
        current = match.group(1).strip().lower()
        start = match.end()
    parts[current] = parts.get(current, "") + body[start:]
    return {k: _clean(v).strip() for k, v in parts.items()}


def _substantive(text: str, minimum: int = 12) -> bool:
    return len(re.sub(r"[\s`*_>#-]+", "", text)) >= minimum


def _section(parts: dict[str, str], pattern: str) -> tuple[str | None, str | None]:
    """Return (heading, content) for the first heading matching ``pattern``."""
    for heading, content in parts.items():
        if heading and re.search(pattern, heading, re.IGNORECASE):
            return heading, content
    return None, None


def _snippet(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ----------------------------------------------------------------------------- rules
_TRACEBACK = re.compile(r"Traceback \(most recent call last\)|^\s*File \".+\", line \d+", re.M)
_ERROR_LINE = re.compile(r"\b[A-Z][A-Za-z]*(?:Error|Exception|Warning)\b\s*:", re.M)
_FAILURE_WORDS = re.compile(
    r"\b(error|exception|traceback|crash(?:es|ed)?|segfault|segmentation fault|warning|"
    r"raises?|fails? with|abort(?:s|ed)?|core dumped|killed)\b",
    re.IGNORECASE,
)
_OS = re.compile(
    r"\b(windows(?:\s*\d+)?|win32|win-amd64|mac\s?os(?:\s*x)?|macos-|os x|darwin|linux|ubuntu|"
    r"debian|fedora|centos|red hat|rhel|arch linux|wsl|manylinux|aarch64|arm64|x86_64)\b",
    re.IGNORECASE,
)
_PY_VERSION = re.compile(r"\bpython\s*(?:version)?\s*[:=]?\s*v?3\.\d+", re.IGNORECASE)
_LIB_VERSION = re.compile(
    r"\b(?:scikit-learn|sklearn|pandas|numpy|scipy)\s*(?:version)?\s*(?:[:=]{1,2}|==|v|\s)\s*"
    r"v?\d+\.\d+",
    re.IGNORECASE,
)
_VERSION_WORD = re.compile(r"\bversion\b[^\n]{0,30}\d+\.\d+", re.IGNORECASE)
_STEPS_LIST = re.compile(r"^\s*(?:\d+[.)]|step\s*\d+)\s+\S", re.IGNORECASE | re.MULTILINE)
_CODE_LIKE = re.compile(r"(\bimport\b|>>>|\w+\s*=\s*\S|\w+\(.*\))")
_EXPECTED = re.compile(
    r"\b(expected|i expect|would expect|should (?:be|return|give|work|produce|raise|not)|"
    r"supposed to)\b",
    re.IGNORECASE,
)
_ACTUAL = re.compile(
    r"\b(actual(?:ly)?|instead|but (?:it |i |we )?(?:get|got|returns?|gives?|raises?)|"
    r"i get|we get|it returns|returns? (?:an? )?(?:wrong|incorrect|nan|none)|output is)\b",
    re.IGNORECASE,
)


def _code_blocks(text: str) -> list[str]:
    return [b for b in _FENCE.findall(text) if b.strip()]


def _item(item, status, reason, matched=None) -> ChecklistItem:
    return ChecklistItem(
        item_id=item,
        status=status,
        reason=reason,
        matched_text=_snippet(matched) if matched else None,
        rule_version=RULE_VERSION,
    )


def _reproduction(parts, body) -> ChecklistItem:
    item = ChecklistItemId.REPRODUCTION
    heading, content = _section(parts, r"reproduc|steps|code|example")
    if heading is not None:
        if _substantive(content or "", 20):
            return _item(item, "present", f"Section “{heading}” has content.", content)
        return _item(item, "missing", f"Section “{heading}” is empty or a placeholder.")
    code = [b for b in _code_blocks(body) if _CODE_LIKE.search(b)]
    if code and len(code[0].strip().splitlines()) >= 2:
        return _item(item, "present", "A code block that looks runnable is included.", code[0])
    steps = _STEPS_LIST.findall(body)
    if len(steps) >= 2:
        return _item(item, "present", "Numbered steps are listed.")
    if code:
        return _item(item, "uncertain", "Only a very short code snippet is included.", code[0])
    return _item(item, "missing", "No reproduction section, runnable code or numbered steps.")


def _expected(parts, body) -> ChecklistItem:
    item = ChecklistItemId.EXPECTED
    heading, content = _section(parts, r"expected")
    if heading is not None:
        if _substantive(content or "", 8):
            return _item(item, "present", f"Section “{heading}” has content.", content)
        return _item(item, "missing", f"Section “{heading}” is empty or a placeholder.")
    match = _EXPECTED.search(_clean(body))
    if match:
        line = _clean(body)[max(0, match.start() - 40) : match.end() + 60]
        return _item(item, "present", "The text states the expected behaviour.", line)
    return _item(item, "missing", "No statement of what should have happened.")


def _actual(parts, body) -> ChecklistItem:
    item = ChecklistItemId.ACTUAL
    heading, content = _section(parts, r"actual|observed|current behavio")
    if heading is not None:
        if _substantive(content or "", 8):
            return _item(item, "present", f"Section “{heading}” has content.", content)
        return _item(item, "missing", f"Section “{heading}” is empty or a placeholder.")
    text = _clean(body)
    if _TRACEBACK.search(text) or _ERROR_LINE.search(text):
        return _item(item, "present", "An error or traceback shows what actually happened.")
    match = _ACTUAL.search(text)
    if match:
        line = text[max(0, match.start() - 40) : match.end() + 60]
        return _item(item, "present", "The text describes the observed behaviour.", line)
    if _substantive(parts.get("", ""), 80):
        return _item(item, "uncertain", "A description exists but no explicit observed result.")
    return _item(item, "missing", "No description of what actually happened.")


def _environment(parts, body) -> ChecklistItem:
    item = ChecklistItemId.ENVIRONMENT
    text = _clean(body)
    match = _OS.search(text)
    if match:
        return _item(item, "present", "Operating system / platform is mentioned.", match.group(0))
    if _PY_VERSION.search(text):
        return _item(item, "uncertain", "Python version given, but no operating system.")
    return _item(item, "missing", "No operating system or platform information.")


def _version(parts, body) -> ChecklistItem:
    item = ChecklistItemId.VERSION
    text = _clean(body)
    match = _LIB_VERSION.search(text)
    if match:
        return _item(item, "present", "A library version is stated.", match.group(0))
    heading, content = _section(parts, r"version")
    if heading is not None and _substantive(content or "", 8):
        return _item(
            item,
            "uncertain",
            f"Section “{heading}” has content without a clear library version.",
            content,
        )
    match = _VERSION_WORD.search(text)
    if match:
        return _item(
            item, "uncertain", "A version number appears without a library name.", match.group(0)
        )
    if heading is not None:
        return _item(item, "missing", f"Section “{heading}” is empty or a placeholder.")
    return _item(item, "missing", "No library version stated.")


def _logs(parts, body) -> ChecklistItem:
    item = ChecklistItemId.LOGS
    text = _clean(body)
    if _TRACEBACK.search(text):
        return _item(item, "present", "A traceback is included.")
    match = _ERROR_LINE.search(text)
    if match:
        line = text[match.start() : match.end() + 80]
        return _item(item, "present", "An error message is included.", line)
    failure = _FAILURE_WORDS.search(text)
    if failure:
        return _item(
            item,
            "missing",
            f"The report mentions “{failure.group(0)}” but includes no error message or traceback.",
        )
    return _item(item, "not_applicable", "No error, crash or warning is described.")


_RULES = (_reproduction, _expected, _actual, _environment, _version, _logs)


def checklist_applies(classification: ClassificationResult) -> bool:
    if classification.predicted_label is None or classification.deferred:
        return True
    return classification.predicted_label == Category.BUG


class RuleChecklist:
    """Deterministic rules; no model or network access."""

    def check(
        self, issue: Issue, classification: ClassificationResult
    ) -> tuple[ChecklistItem, ...]:
        if not checklist_applies(classification):
            label = classification.predicted_label.value if classification.predicted_label else "?"
            return tuple(
                _item(item_id, "not_applicable", f"Accepted category is {label}, not bug.")
                for item_id in ChecklistItemId
            )
        parts = sections(issue.body)
        return tuple(rule(parts, issue.body) for rule in _RULES)
