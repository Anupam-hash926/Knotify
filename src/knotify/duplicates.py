"""Explicit duplicate evidence for retrieval evaluation (read-only, resumable).

Candidates are issues whose text mentions "duplicate of" (issue search, partitioned by
creation date). Their comments are fetched once and cached; only explicit statements naming
a same-repository target (``duplicate of #N`` or its issue URL) are kept. Comment author
*association* (MEMBER, CONTRIBUTOR...) is kept as evidence strength; identities are not.
With a token, GitHub's ``MarkedAsDuplicateEvent`` is added via GraphQL.

A pair is valid only when the target is an issue in the frozen snapshot (PRs are absent from
it) created strictly before the query. Evidence is never a model feature.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knotify.audit import extract_duplicate_targets
from knotify.github import GitHubClient, RateLimited


class DuplicatesPaused(RuntimeError):
    pass


def _cached(path: Path, compute, deadline: float | None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if deadline is not None and time.monotonic() > deadline:
        raise DuplicatesPaused("Time budget reached; rerun to resume.")
    try:
        value = compute()
    except RateLimited as exc:
        raise DuplicatesPaused(f"GitHub rate limit: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def candidate_numbers(
    client: GitHubClient, repo: str, created_from: str, cache: Path, deadline: float | None
) -> list[int]:
    query = f'repo:{repo} is:issue "duplicate of" created:>={created_from}'

    def fetch():
        numbers, page = [], 1
        while True:
            result = client.search_issues(query, per_page=100, page=page, sort="created")
            numbers += [i["number"] for i in result["items"] if "pull_request" not in i]
            if page * 100 >= min(result["total_count"], 1000):
                return {"query": query, "total_count": result["total_count"], "numbers": numbers}
            page += 1

    found = _cached(cache / "candidates.json", fetch, deadline)
    if found["total_count"] > 1000:
        raise RuntimeError("More than 1,000 candidates; narrow created_from.")
    return found["numbers"]


def comment_statements(
    client: GitHubClient, repo: str, number: int, cache: Path, deadline: float | None
) -> list[dict[str, Any]]:
    def fetch():
        rows = []
        for comment in client.paginate(f"/repos/{repo}/issues/{number}/comments"):
            for target in extract_duplicate_targets(comment.get("body") or "", repo):
                rows.append(
                    {
                        "target": target,
                        "comment_id": comment["id"],
                        "created_at": comment["created_at"],
                        "author_association": comment.get("author_association"),
                    }
                )
        return rows

    return _cached(cache / "comments" / f"{number}.json", fetch, deadline)


def collect(
    client: GitHubClient,
    repo: str,
    records: list[dict[str, Any]],
    out_dir: Path,
    *,
    created_from: str,
    time_budget: float | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + time_budget if time_budget else None
    by_number = {r["number"]: r for r in records}
    evidence: list[dict[str, Any]] = []
    for record in records:
        if record["created_at"] >= created_from:
            for target in extract_duplicate_targets(record["body"], repo):
                evidence.append({"query": record["number"], "target": target, "evidence": "body"})
    numbers = candidate_numbers(client, repo, created_from, out_dir, deadline)
    done = 0
    try:
        for number in numbers:
            for row in comment_statements(client, repo, number, out_dir, deadline):
                evidence.append({"query": number, "evidence": "comment", **row})
            done += 1
    except DuplicatesPaused as exc:
        status = {"complete": False, "paused": str(exc)}
    else:
        status = {"complete": True}
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for row in evidence:
        query, target = by_number.get(row["query"]), by_number.get(row["target"])
        valid = bool(query and target and target["created_at"] < query["created_at"])
        key = (row["query"], row["target"])
        entry = pairs.setdefault(
            key,
            {
                "query": row["query"],
                "target": row["target"],
                "query_created_at": query["created_at"] if query else None,
                "target_created_at": target["created_at"] if target else None,
                "target_in_snapshot_as_issue": target is not None,
                "valid_earlier_issue_target": valid,
                "evidence": [],
            },
        )
        entry["evidence"].append({k: v for k, v in row.items() if k not in ("query", "target")})
    summary = {
        "repository": repo,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "created_from": created_from,
        "candidates": len(numbers),
        "candidates_with_comments_checked": done,
        "pairs": len(pairs),
        "valid_pairs": sum(p["valid_earlier_issue_target"] for p in pairs.values()),
        **status,
    }
    (out_dir / "pairs.json").write_text(
        json.dumps({"summary": summary, "pairs": list(pairs.values())}, indent=2) + "\n", "utf-8"
    )
    return summary


def freeze_retrieval_queries(pairs_path: Path, out_path: Path, *, dev_count: int = 5) -> dict:
    """Freeze explicit-duplicate retrieval queries before any method is scored on them.

    The earliest ``dev_count`` queries are for debugging only; the rest form the evaluation set.
    Relevance here is *duplicate* relevance from explicit statements; broader "related"
    judgements are added later by pooled manual review.
    """
    if out_path.exists():
        raise FileExistsError(f"Retrieval query set already frozen: {out_path}")
    data = json.loads(pairs_path.read_text(encoding="utf-8"))
    queries: dict[int, dict[str, Any]] = {}
    for pair in data["pairs"]:
        if not pair["valid_earlier_issue_target"]:
            continue
        entry = queries.setdefault(
            pair["query"],
            {"query": pair["query"], "query_created_at": pair["query_created_at"], "relevant": []},
        )
        entry["relevant"].append(
            {
                "target": pair["target"],
                "relevance": "duplicate",
                "source": "explicit_statement",
                "author_associations": sorted(
                    {e.get("author_association") or "unknown" for e in pair["evidence"]}
                ),
            }
        )
    ordered = sorted(queries.values(), key=lambda q: (q["query_created_at"], q["query"]))
    frozen = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(pairs_path.name),
        "source_summary": data["summary"],
        "dev": ordered[:dev_count],
        "evaluation": ordered[dev_count:],
        "note": "Frozen before scoring any retrieval method. Dev queries are never reported.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen
