"""Input text, label mapping, exact-duplicate removal and frozen chronological splits.

Only title/body ever enter model features. Labels, state, comments and duplicate evidence are
targets or evaluation metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knotify.contracts import Category, Issue

LABELS = tuple(c.value for c in Category)


def model_text(issue: Issue) -> str:
    """Preserve technical content; never include labels, state or author metadata."""
    return f"{issue.title.strip()}\n\n{issue.body.strip()}".strip()


def stable_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def input_hash(issue: Issue) -> str:
    return stable_hash({"title": issue.title, "body": issue.body})


# --------------------------------------------------------------------------- cleaning
_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|$)", re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?(?:details|summary|br|p|img|sub|sup|kbd|b|i|em|strong)\b[^>]*>", re.I)


def clean_markdown(text: str) -> str:
    """Remove template comments, images and presentational HTML; keep code and identifiers."""
    text = _HTML_COMMENT.sub(" ", text or "")
    text = _IMAGE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def feature_text(title: str, body: str) -> str:
    return f"{title.strip()}\n\n{clean_markdown(body)}".strip()


def normalized_for_dedup(title: str, body: str) -> str:
    text = unicodedata.normalize("NFKC", feature_text(title, body)).casefold()
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- mapping
@dataclass(frozen=True)
class RepositoryMapping:
    name: str
    role: str
    labels: dict[str, tuple[str, ...]]
    window_start: str | None = None
    exclude_labels: tuple[str, ...] = ()

    def label_to_target(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for target, names in self.labels.items():
            for name in names:
                if name in out and out[name] != target:
                    raise ValueError(f"Label {name!r} is mapped to two targets.")
                out[name] = target
        return out


def load_mappings(path: Path) -> dict[str, RepositoryMapping]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    result = {}
    for item in raw["repositories"]:
        labels = {t: tuple(item["label_mapping"].get(t, ())) for t in LABELS}
        result[item["name"]] = RepositoryMapping(
            item["name"],
            item["role"],
            labels,
            item.get("window_start"),
            tuple(item.get("exclude_labels", ())),
        )
    return result


def map_labels(labels: Iterable[str], mapping: RepositoryMapping) -> tuple[str | None, str]:
    """Return (target, status): status is ``mapped``, ``unmapped`` or ``conflict``."""
    lookup = mapping.label_to_target()
    targets = {lookup[name] for name in labels if name in lookup}
    if not targets:
        return None, "unmapped"
    if len(targets) > 1:
        return None, "conflict"
    return targets.pop(), "mapped"


@dataclass(frozen=True)
class SupervisedSet:
    rows: list[dict[str, Any]]
    exclusions: dict[str, int]


def build_supervised(
    records: Sequence[dict[str, Any]], mapping: RepositoryMapping
) -> SupervisedSet:
    """Single-target issues inside the window, exact normalized text duplicates removed."""
    exclusions: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for record in sorted(records, key=lambda r: (r["created_at"], r["number"])):
        if mapping.window_start and record["created_at"] < mapping.window_start:
            exclusions["before_window"] += 1
            continue
        target, status = map_labels(record["labels"], mapping)
        if target is not None and set(record["labels"]) & set(mapping.exclude_labels):
            exclusions["excluded_label"] += 1
            continue
        if target is None:
            exclusions[status] += 1
            continue
        key = normalized_for_dedup(record["title"], record["body"])
        if key in seen_text:
            exclusions["exact_text_duplicate"] += 1
            continue
        seen_text.add(key)
        kept.append({**record, "target": target})
    return SupervisedSet(kept, dict(exclusions))


# --------------------------------------------------------------------------- splits
def _boundary(rows: Sequence[dict[str, Any]], index: int) -> int:
    """Move a cut forward so equal creation timestamps stay on one side."""
    while 0 < index < len(rows) and rows[index]["created_at"] == rows[index - 1]["created_at"]:
        index += 1
    return index


def chronological_split(
    rows: Sequence[dict[str, Any]], fractions: tuple[float, float, float]
) -> dict[str, list[dict[str, Any]]]:
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Split fractions must sum to 1.")
    ordered = sorted(rows, key=lambda r: (r["created_at"], r["number"]))
    n = len(ordered)
    a = _boundary(ordered, round(n * fractions[0]))
    b = _boundary(ordered, max(a, round(n * (fractions[0] + fractions[1]))))
    return {"train": ordered[:a], "validation": ordered[a:b], "test": ordered[b:]}


def split_validation(
    validation: Sequence[dict[str, Any]], *, min_size: int = 100, min_per_class: int = 10
) -> dict[str, Any]:
    """Chronological halves V-cal / V-policy when feasible, else the declared reuse fallback."""
    half = _boundary(validation, len(validation) // 2)
    cal, policy = list(validation[:half]), list(validation[half:])
    counts = Counter(r["target"] for r in cal)
    feasible = (
        len(cal) >= min_size
        and len(policy) >= min_size
        and all(counts.get(label, 0) >= min_per_class for label in LABELS)
    )
    if feasible:
        return {"mode": "disjoint_halves", "v_cal": cal, "v_policy": policy}
    return {
        "mode": "reuse_full_validation",
        "v_cal": list(validation),
        "v_policy": list(validation),
        "reason": f"V-cal size {len(cal)}, policy {len(policy)}, V-cal class counts {dict(counts)}",
    }


def class_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(r["target"] for r in rows)
    return {label: counts.get(label, 0) for label in LABELS}
