"""Input-only text and canonical hashes; label mapping/split generation remain future work."""

import hashlib
import json

from knotify.contracts import Issue


def model_text(issue: Issue) -> str:
    """Preserve technical content; never include labels, state or author metadata."""
    return f"{issue.title.strip()}\n\n{issue.body.strip()}".strip()


def stable_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def input_hash(issue: Issue) -> str:
    return stable_hash({"title": issue.title, "body": issue.body})
