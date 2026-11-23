"""Shared feature transform for classification and retrieval (title/body only).

``mask_template_cues`` removes reporter-template artefacts that encode the template choice
rather than the issue content: conventional title type prefixes (``BUG:``, ``ENH:``...),
Markdown heading lines and checkbox boilerplate. Both audited repositories' templates
auto-apply the type label, so these cues would otherwise let a model learn the template.
"""

from __future__ import annotations

import re

from knotify.data import clean_markdown

_PREFIX = re.compile(
    r"^\s*(?:[\[(]?\s*(?:bug|bugs|enh|enhancement|feat|feature|fea|doc|docs|documentation|qst|"
    r"question|build|perf|tst|test|ci|cln|ref|api|maint|mnt|sty|typ|depr|regr|regression|"
    r"rfc|request)\s*[\])]?\s*[:\-|]\s*)+",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
_CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s.*$", re.MULTILINE)
TOKEN_PATTERN = r"(?u)v?\d+(?:\.\d+)+|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_TOKEN = re.compile(TOKEN_PATTERN)


def mask_template_cues(text: str) -> str:
    text = _PREFIX.sub("", text, count=1)
    text = _HEADING.sub(" ", text)
    text = _CHECKBOX.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_DUP_SENTENCE = re.compile(
    r"[^\n.]*\b(?:duplicate|dup|dupe)s?\s+(?:of|to)\b[^\n.]*[.\n]?", re.IGNORECASE
)


def strip_duplicate_markers(text: str) -> str:
    """For retrieval *evaluation* queries: remove explicit duplicate statements from the text."""
    return _DUP_SENTENCE.sub(" ", text)


def prepare(text: str, *, mask: bool = True) -> str:
    """``text`` is ``data.model_text(issue)``: title, blank line, body."""
    title, _, body = text.partition("\n\n")
    combined = f"{title.strip()}\n\n{clean_markdown(body)}".strip()
    return mask_template_cues(combined) if mask else combined


def tokenize(text: str) -> list[str]:
    """Lower-cased identifiers (dotted names kept whole) and version numbers."""
    return [token.lower() for token in _TOKEN.findall(text)]
