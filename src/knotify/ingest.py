"""Read-only GitHub import into a dated, frozen JSONL snapshot.

* Lists ``/repos/{repo}/issues?state=all`` (newest first), follows pagination, drops every object
  carrying a ``pull_request`` key and deduplicates by stable GitHub ID.
* Optionally stops at a creation-date floor (the endpoint's ``since`` filters *updates*, so it is
  not used).
* Records GitHub's explicit ``MarkedAsDuplicateEvent`` (canonical issue named by GitHub) through
  read-only GraphQL, plus explicit "duplicate of #N" statements in bodies and in comments of
  issues found by text search. Duplicate evidence is evaluation metadata, never a feature.
* Author identity, reactions and assignees are not stored.

A snapshot directory is never overwritten. An interrupted import writes nothing final.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knotify.audit import extract_duplicate_targets
from knotify.contracts import Issue
from knotify.errors import ComponentUnavailable
from knotify.github import GitHubClient

SNAPSHOT_FIELDS = (
    "id",
    "node_id",
    "number",
    "title",
    "body",
    "state",
    "state_reason",
    "created_at",
    "updated_at",
    "closed_at",
    "html_url",
    "comments",
)


def slim(item: dict[str, Any], fetched_at: str) -> dict[str, Any]:
    record = {key: item.get(key) for key in SNAPSHOT_FIELDS}
    record["labels"] = sorted(label["name"] for label in item.get("labels", []))
    record["body"] = record["body"] or ""
    record["fetched_at"] = fetched_at
    return record


def iter_issues(
    client: GitHubClient, repo: str, *, created_floor: str | None = None
) -> Iterator[dict[str, Any]]:
    """Yield raw non-PR issues newest first, stopping below ``created_floor`` (ISO date)."""
    params = {"state": "all", "sort": "created", "direction": "desc", "per_page": 100}
    for item in client.paginate(f"/repos/{repo}/issues", params):
        if created_floor and item["created_at"] < created_floor:
            return
        if "pull_request" in item:
            continue
        yield item


def dedupe(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one record per stable ID (the first seen, i.e. the earliest fetch)."""
    seen: dict[int, dict[str, Any]] = {}
    for record in records:
        seen.setdefault(record["id"], record)
    return sorted(seen.values(), key=lambda r: (r["created_at"], r["number"]))


_DUP_QUERY = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    issues(first: 100, after: $after, orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        timelineItems(itemTypes: [MARKED_AS_DUPLICATE_EVENT], first: 10) {
          nodes {
            ... on MarkedAsDuplicateEvent {
              createdAt
              isCrossRepository
              canonical { __typename ... on Issue { number } ... on PullRequest { number } }
            }
          }
        }
      }
    }
  }
}
"""


def marked_duplicates(client: GitHubClient, repo: str) -> list[dict[str, Any]]:
    owner, name = repo.split("/")
    after, events = None, []
    while True:
        data = client.graphql(_DUP_QUERY, {"owner": owner, "name": name, "after": after})
        issues = data["repository"]["issues"]
        for node in issues["nodes"]:
            for event in node["timelineItems"]["nodes"]:
                canonical = event.get("canonical") or {}
                if event.get("isCrossRepository") or "number" not in canonical:
                    continue
                events.append(
                    {
                        "query": node["number"],
                        "target": canonical["number"],
                        "target_type": canonical["__typename"],
                        "evidence": "marked_as_duplicate_event",
                        "marked_at": event["createdAt"],
                    }
                )
        if not issues["pageInfo"]["hasNextPage"]:
            return events
        after = issues["pageInfo"]["endCursor"]


def text_duplicates(
    client: GitHubClient, repo: str, records: list[dict[str, Any]], *, search_pages: int = 10
) -> list[dict[str, Any]]:
    """Explicit statements in bodies (all records) and comments (text-search hits only)."""
    pairs = []
    for record in records:
        for target in extract_duplicate_targets(record["body"], repo):
            pairs.append({"query": record["number"], "target": target, "evidence": "body"})
    hits: set[int] = set()
    for page in range(1, search_pages + 1):
        result = client.search_issues(
            f'repo:{repo} is:issue "duplicate of"', per_page=100, page=page
        )
        hits |= {item["number"] for item in result["items"] if "pull_request" not in item}
        if page * 100 >= min(result["total_count"], 1000):
            break
    for number in sorted(hits):
        for comment in client.paginate(f"/repos/{repo}/issues/{number}/comments"):
            for target in extract_duplicate_targets(comment.get("body") or "", repo):
                pairs.append({"query": number, "target": target, "evidence": "comment"})
    return pairs


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def snapshot_repository(
    client: GitHubClient,
    repo: str,
    data_dir: Path,
    *,
    created_floor: str | None = None,
    duplicates: bool = True,
) -> Path:
    """Import ``repo`` and freeze it under ``data/raw/<owner>__<name>/<UTC date>/``."""
    if not client.authenticated:
        raise ComponentUnavailable("A GITHUB_TOKEN is required for full collection.")
    fetched_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    target = data_dir / "raw" / repo.replace("/", "__") / fetched_at[:10]
    if target.exists():
        raise FileExistsError(f"Snapshot already exists and is frozen: {target}")
    started = client.request_count
    records = dedupe(
        slim(item, fetched_at) for item in iter_issues(client, repo, created_floor=created_floor)
    )
    dup_rows: list[dict[str, Any]] = []
    if duplicates:
        dup_rows = marked_duplicates(client, repo) + text_duplicates(client, repo, records)
    staging = target.with_name(target.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    issues_hash = _write_jsonl(staging / "issues.jsonl", records)
    dups_hash = _write_jsonl(staging / "duplicate_evidence.jsonl", dup_rows)
    manifest = {
        "repository": repo,
        "fetched_at": fetched_at,
        "endpoint": f"GET /repos/{repo}/issues?state=all&sort=created&direction=desc",
        "created_floor": created_floor,
        "pull_requests_excluded": True,
        "issue_count": len(records),
        "issues_sha256": issues_hash,
        "duplicate_evidence_count": len(dup_rows),
        "duplicate_evidence_sha256": dups_hash,
        "created_range": [records[0]["created_at"], records[-1]["created_at"]] if records else None,
        "api_requests": client.request_count - started,
        "partial": False,
        "stored_fields": [*SNAPSHOT_FIELDS, "labels", "fetched_at"],
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(target)
    return target


def _read_jsonl(text: str) -> list[dict[str, Any]]:
    # Split on "\n" only: str.splitlines() also breaks on U+2028/U+0085 inside JSON strings.
    return [json.loads(line) for line in text.split("\n") if line]


def load_snapshot(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    raw = (path / "issues.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["issues_sha256"]:
        raise ValueError(f"Snapshot hash mismatch: {path}")
    records = _read_jsonl(raw.decode("utf-8"))
    dups = _read_jsonl((path / "duplicate_evidence.jsonl").read_bytes().decode("utf-8"))
    return manifest, records, dups


def to_issue(record: dict[str, Any], repo: str) -> Issue:
    return Issue(
        issue_id=f"{repo}#{record['number']}",
        repository=repo,
        number=record["number"],
        title=record["title"].strip() or "(untitled)",
        body=record["body"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        fetched_at=record["fetched_at"],
        url=record["html_url"],
        source_labels=tuple(record["labels"]),
    )


def import_repository(repository: str) -> tuple[Issue, ...]:
    """Kept for the scaffold contract: use ``snapshot_repository`` for real collection."""
    raise ComponentUnavailable(
        "Use `knotify import` / snapshot_repository(); live import into the UI is not supported."
    )


# --------------------------------------------------------------------------- search mode
class CollectionPaused(RuntimeError):
    """Time budget reached; cached search pages are kept and a rerun resumes."""


def _search_page(client, cache_dir: Path, query: str, page: int, deadline: float | None) -> dict:
    """Fetch (or replay from the staging cache) one search page of 100 issues."""
    import time

    key = hashlib.sha256(f"{query}|{page}".encode()).hexdigest()[:24]
    path = cache_dir / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if deadline is not None and time.monotonic() > deadline:
        raise CollectionPaused("Time budget reached; rerun the same command to resume.")
    result = client.search_issues(query, per_page=100, page=page, sort="created")
    payload = {
        "query": query,
        "page": page,
        "fetched_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "total_count": result["total_count"],
        "incomplete_results": result["incomplete_results"],
        "items": result["items"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def iter_search_partitioned(
    client, repo: str, start, end, cache_dir: Path, deadline: float | None, log: list
) -> Iterator[tuple[dict, str]]:
    """Yield (issue, fetched_at) for ``created:start..end``, bisecting ranges over 1,000 hits."""
    from datetime import timedelta

    query = f"repo:{repo} is:issue created:{start.isoformat()}..{end.isoformat()}"
    first = _search_page(client, cache_dir, query, 1, deadline)
    total = first["total_count"]
    if total > 1000:
        if start == end:
            raise RuntimeError(f"More than 1,000 issues created on {start}; cannot partition.")
        middle = start + (end - start) // 2
        yield from iter_search_partitioned(client, repo, start, middle, cache_dir, deadline, log)
        yield from iter_search_partitioned(
            client, repo, middle + timedelta(days=1), end, cache_dir, deadline, log
        )
        return
    pages = [first] + [
        _search_page(client, cache_dir, query, page, deadline)
        for page in range(2, (total + 99) // 100 + 1)
    ]
    items = [item for page in pages for item in page["items"]]
    log.append(
        {
            "range": [start.isoformat(), end.isoformat()],
            "total_count": total,
            "received": len(items),
            "incomplete": any(p["incomplete_results"] for p in pages),
        }
    )
    for page in pages:
        for item in page["items"]:
            yield item, page["fetched_at"]


def snapshot_via_search(
    client,
    repo: str,
    data_dir: Path,
    *,
    start: str,
    end: str | None = None,
    time_budget: float | None = None,
) -> Path:
    """Freeze ``repo`` issues created in [start, end] using date-partitioned issue search.

    Works without a token (10 searches/min). Completeness is checked per partition
    (received == total_count, no incomplete flags) and against a fresh whole-range count.
    """
    import time
    from datetime import date

    deadline = time.monotonic() + time_budget if time_budget else None
    end_day = date.fromisoformat(end) if end else datetime.now(UTC).date()
    start_day = date.fromisoformat(start)
    staging = data_dir / "raw" / repo.replace("/", "__") / f"search_{start}_{end_day}.partial"
    cache_dir = staging / "pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    collected = list(
        iter_search_partitioned(client, repo, start_day, end_day, cache_dir, deadline, log)
    )
    whole = _search_page(
        client, cache_dir, f"repo:{repo} is:issue created:{start}..{end_day}", 0, deadline
    )
    records = dedupe(
        slim(item, fetched) for item, fetched in collected if "pull_request" not in item
    )
    problems = [p for p in log if p["incomplete"] or p["received"] != p["total_count"]]
    complete = not problems and len(records) == whole["total_count"]
    final = staging.parent / f"{start}_{end_day}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    final.mkdir(parents=True)
    issues_hash = _write_jsonl(final / "issues.jsonl", records)
    _write_jsonl(final / "duplicate_evidence.jsonl", [])
    dup_hash = hashlib.sha256(b"").hexdigest()
    fetched = sorted(r["fetched_at"] for r in records)
    manifest = {
        "repository": repo,
        "collection_method": "GET /search/issues, date-partitioned (<=1000 per partition)",
        "query_template": f"repo:{repo} is:issue created:<start>..<end>",
        "created_window": [start, end_day.isoformat()],
        "fetched_range": [fetched[0], fetched[-1]] if fetched else None,
        "pull_requests_excluded": True,
        "issue_count": len(records),
        "whole_window_search_count": whole["total_count"],
        "partitions": log,
        "partition_problems": problems,
        "partial": not complete,
        "issues_sha256": issues_hash,
        "duplicate_evidence_count": 0,
        "duplicate_evidence_sha256": dup_hash,
        "created_range": [records[0]["created_at"], records[-1]["created_at"]] if records else None,
        "stored_fields": [*SNAPSHOT_FIELDS, "labels", "fetched_at"],
        "note": "Duplicate evidence is collected separately (knotify duplicates).",
    }
    (final / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return final
