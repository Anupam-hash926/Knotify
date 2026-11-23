"""Read-only dataset feasibility audit of candidate GitHub repositories.

Search mode (no token needed) measures, via GitHub issue search with boolean operators:
non-PR issue totals, per-label counts, per-target union/usable/conflict counts, yearly usable
counts and duplicate-evidence volumes. "Usable" means the issue carries labels of exactly one
candidate target. Search counts describe the live tracker at query time; exact post-filter
counts (empty bodies, exact-text duplicates) come from the frozen snapshot, not from here.

Duplicate discovery samples issues whose text mentions "duplicate of" and extracts explicit
``#N`` / issue-URL targets from the body and, when budget allows, comments. A pair is reported
as *verified* only when the target is a same-repository issue (not a PR) created strictly
before the query issue.
"""

from __future__ import annotations

import json
import re
import time
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knotify.github import GitHubClient, RateLimited

TARGETS = ("bug", "feature", "question_documentation", "other")


@dataclass(frozen=True)
class CandidateRepository:
    name: str
    role: str
    mapping: dict[str, tuple[str, ...]]
    duplicate_labels: tuple[str, ...] = ()
    yearly: bool = False


@dataclass(frozen=True)
class AuditPlan:
    repositories: tuple[CandidateRepository, ...]
    year_buckets: tuple[int, ...] = field(default=())


def load_plan(path: Path) -> AuditPlan:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    repos = []
    for item in raw["repositories"]:
        mapping = {t: tuple(item["candidate_mapping"].get(t, ())) for t in TARGETS}
        repos.append(
            CandidateRepository(
                name=item["name"],
                role=item["role"],
                mapping=mapping,
                duplicate_labels=tuple(item.get("duplicate_labels", ())),
                yearly=bool(item.get("yearly", False)),
            )
        )
    return AuditPlan(tuple(repos), tuple(raw.get("year_buckets", ())))


def _label(name: str) -> str:
    return f'label:"{name}"' if re.search(r"[\s:/]", name) else f"label:{name}"


def _any(labels: tuple[str, ...]) -> str:
    terms = [_label(x) for x in labels]
    return terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"


def _none(labels: tuple[str, ...]) -> str:
    return " ".join(f"-{_label(x)}" for x in labels)


def _bucket_qualifier(buckets: tuple[int, ...], index: int) -> tuple[str, str]:
    start = buckets[index]
    if index == 0:
        return f"<{buckets[1]}", f"created:<{buckets[1]}-01-01"
    end = buckets[index + 1] if index + 1 < len(buckets) else None
    if end is None:
        return f"{start}+", f"created:>={start}-01-01"
    return str(start), f"created:{start}-01-01..{start}-12-31"


class AuditPaused(RuntimeError):
    """The time budget ran out; cached results are kept and a rerun resumes."""


class SearchCache:
    """Persisted search counts so a rate-limited audit can resume across short runs."""

    def __init__(self, client: GitHubClient, path: Path | None, deadline: float | None):
        self.client, self.path, self.deadline = client, path, deadline
        self.data: dict[str, Any] = {}
        if path and path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")

    def _check(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            self.save()
            raise AuditPaused("Time budget reached; rerun the same command to resume.")

    def count(self, query: str) -> dict[str, Any]:
        key = f"count:{query}"
        if key not in self.data:
            self._check()
            try:
                result = self.client.search_count(query)
            except Exception:
                self.save()
                raise
            self.data[key] = {**result, "at": datetime.now(UTC).isoformat(timespec="seconds")}
            if len(self.data) % 10 == 0:
                self.save()
        return self.data[key]

    def memo(self, key: str, compute) -> Any:
        if key not in self.data:
            self._check()
            self.data[key] = compute()
            self.save()
        return self.data[key]


class _Counter:
    def __init__(self, cache: SearchCache, repo: str):
        self.cache, self.base = cache, f"repo:{repo} is:issue"
        self.queries: list[dict[str, Any]] = []

    def __call__(self, qualifier: str = "") -> int:
        query = f"{self.base} {qualifier}".strip()
        result = self.cache.count(query)
        self.queries.append({"query": query, **result})
        return result["total_count"]


def count_audit(cache: SearchCache, repo: CandidateRepository, buckets: tuple[int, ...]) -> dict:
    count = _Counter(cache, repo.name)
    targets = [t for t in TARGETS if repo.mapping[t]]
    all_labels = tuple(x for t in targets for x in repo.mapping[t])
    others = {t: tuple(x for o in targets if o != t for x in repo.mapping[o]) for t in targets}

    def usable_q(t: str, labels: tuple[str, ...] | None = None) -> str:
        return f"{_any(labels or repo.mapping[t])} {_none(others[t])}".strip()

    result: dict[str, Any] = {
        "repository": repo.name,
        "role": repo.role,
        "candidate_mapping": {t: list(repo.mapping[t]) for t in TARGETS},
        "issues_total": count(),
        "pull_requests_total": cache.count(f"repo:{repo.name} is:pr")["total_count"],
        "labels": {},
        "targets": {},
    }
    for target in targets:
        union = count(_any(repo.mapping[target]))
        usable = count(usable_q(target))
        per_label = {}
        for name in repo.mapping[target]:
            per_label[name] = {
                "raw": count(_label(name)),
                "usable": count(usable_q(target, (name,))),
            }
        result["labels"].update(per_label)
        result["targets"][target] = {"union": union, "usable": usable, "conflict": union - usable}
    for target in TARGETS:
        if target not in result["targets"]:
            result["targets"][target] = {"union": 0, "usable": 0, "conflict": 0}
    mapped = count(_any(all_labels))
    usable_total = sum(v["usable"] for v in result["targets"].values())
    result["mapped_any"] = mapped
    result["multi_target_conflicts"] = mapped - usable_total
    result["usable_total"] = usable_total
    result["unmapped"] = result["issues_total"] - mapped
    if repo.yearly and len(buckets) > 1:
        yearly: dict[str, dict[str, int]] = {}
        for i in range(len(buckets)):
            name, qualifier = _bucket_qualifier(buckets, i)
            row = {t: count(f"{usable_q(t)} {qualifier}") for t in targets}
            row["all_issues"] = count(qualifier)
            yearly[name] = row
        result["yearly_usable"] = yearly
    result["duplicate_evidence_counts"] = {
        "text_duplicate_of": count('"duplicate of"'),
        "closed_reason_duplicate": count("reason:duplicate"),
        **{f"label:{x}": count(_label(x)) for x in repo.duplicate_labels},
    }
    result["queries"] = count.queries
    return result


# ---------------------------------------------------------------------------- duplicates
_REF = r"(?:#|https://github\.com/{repo}/issues/)(\d+)"


def extract_duplicate_targets(text: str, repo: str) -> list[int]:
    """Explicit 'duplicate of <ref>' statements only; ordinary cross-references are ignored."""
    if not text:
        return []
    ref = _REF.format(repo=re.escape(repo))
    pattern = re.compile(
        rf"\b(?:duplicate|dup|dupe)s?\s+(?:of|to)\s*:?\s*(?:issue\s+)?{ref}", re.IGNORECASE
    )
    seen: list[int] = []
    for match in pattern.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def duplicate_sample(
    client: GitHubClient, repo: str, *, max_queries: int = 100, verify_limit: int = 20
) -> dict:
    """Discover explicit duplicate statements in a sample of issues and verify targets."""
    found = client.search_issues(f'repo:{repo} is:issue "duplicate of"', per_page=max_queries)
    candidates, pairs, notes = [], [], []
    budget_exhausted = False
    for item in found["items"]:
        if "pull_request" in item:
            continue
        number, created = item["number"], item["created_at"]
        targets = [(t, "body") for t in extract_duplicate_targets(item.get("body") or "", repo)]
        if not targets and item.get("comments", 0) and len(candidates) < verify_limit:
            try:
                for comment in client.paginate(f"/repos/{repo}/issues/{number}/comments"):
                    targets += [
                        (t, "comment") for t in extract_duplicate_targets(comment["body"], repo)
                    ]
            except RateLimited:
                budget_exhausted = True
                break
        if targets:
            candidates.append({"query": number, "query_created_at": created, "targets": targets})
    for cand in candidates[:verify_limit]:
        for target, source in dict.fromkeys(cand["targets"]):
            try:
                info = client.get_json(f"/repos/{repo}/issues/{target}")
            except RateLimited:
                budget_exhausted = True
                break
            except Exception as exc:  # 404/410 for deleted or transferred issues
                notes.append(f"#{cand['query']} -> #{target}: {type(exc).__name__}")
                continue
            is_pr = "pull_request" in info
            earlier = info["created_at"] < cand["query_created_at"]
            pairs.append(
                {
                    "query": cand["query"],
                    "target": target,
                    "evidence": source,
                    "query_created_at": cand["query_created_at"],
                    "target_created_at": info["created_at"],
                    "target_is_pr": is_pr,
                    "target_earlier": earlier,
                    "verified": (not is_pr) and earlier,
                    "query_url": f"https://github.com/{repo}/issues/{cand['query']}",
                    "target_url": f"https://github.com/{repo}/issues/{target}",
                }
            )
        if budget_exhausted:
            break
    return {
        "search_hits": found["total_count"],
        "sampled": len(found["items"]),
        "explicit_candidates_in_sample": len(candidates),
        "checked_pairs": pairs,
        "verified_pairs": sum(p["verified"] for p in pairs),
        "budget_exhausted": budget_exhausted,
        "notes": notes,
    }


def run_audit(
    client: GitHubClient,
    plan: AuditPlan,
    *,
    verify_limit: int = 20,
    cache_path: Path | None = None,
    time_budget: float | None = None,
) -> dict:
    deadline = time.monotonic() + time_budget if time_budget else None
    cache = SearchCache(client, cache_path, deadline)
    cache.data.setdefault("started_at", datetime.now(UTC).isoformat(timespec="seconds"))
    repos = []
    for repo in plan.repositories:
        entry = count_audit(cache, repo, plan.year_buckets)

        def sample(name: str = repo.name) -> dict:
            try:
                return duplicate_sample(client, name, verify_limit=verify_limit)
            except RateLimited as exc:
                return {"error": str(exc)}

        entry["duplicate_sample"] = cache.memo(f"dups:{repo.name}", sample)
        repos.append(entry)
    cache.save()
    return {
        "audit": "knotify read-only dataset feasibility audit",
        "mode": "search-count",
        "started_at": cache.data["started_at"],
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "authenticated": client.authenticated,
        "api_requests_final_run": client.request_count,
        "search_queries": sum(k.startswith("count:") for k in cache.data),
        "count_semantics": (
            "GitHub issue search (is:issue excludes PRs) with advanced_search boolean "
            "operators at query time. 'usable' = labels of exactly one candidate target. "
            "Not yet filtered for empty bodies or exact-text duplicates."
        ),
        "repositories": repos,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Dataset feasibility audit (read-only, search counts)",
        "",
        f"- Run: {report['started_at']} → {report['finished_at']} (UTC)",
        f"- Authenticated: {report['authenticated']}; search queries: {report['search_queries']} "
        "(resumable runs; counts can drift slightly while issues are filed)",
        f"- Semantics: {report['count_semantics']}",
        "",
    ]
    for repo in report["repositories"]:
        t = repo["targets"]
        lines += [
            f"## {repo['repository']} ({repo['role']})",
            "",
            f"Issues (non-PR): **{repo['issues_total']}**; PRs: {repo['pull_requests_total']}; "
            f"with any candidate target label: {repo['mapped_any']}; "
            f"unmapped: {repo['unmapped']}; multi-target conflicts: "
            f"{repo['multi_target_conflicts']}; **usable single-target: {repo['usable_total']}**",
            "",
            "| Target | Candidate labels | Union | Conflicting | Usable |",
            "|---|---|---:|---:|---:|",
        ]
        for name in TARGETS:
            labels = ", ".join(f"`{x}`" for x in repo["candidate_mapping"][name]) or "—"
            row = t[name]
            lines.append(
                f"| {name} | {labels} | {row['union']} | {row['conflict']} | {row['usable']} |"
            )
        lines += ["", "| Label | Raw issues | Usable (no other-target label) |", "|---|---:|---:|"]
        for label, row in repo["labels"].items():
            lines.append(f"| `{label}` | {row['raw']} | {row['usable']} |")
        if "yearly_usable" in repo:
            lines += [
                "",
                "Usable single-target issues by creation year:",
                "",
                "| Year | " + " | ".join(TARGETS) + " | all issues |",
                "|---|" + "---:|" * (len(TARGETS) + 1),
            ]
            for year, row in repo["yearly_usable"].items():
                cells = " | ".join(str(row.get(x, 0)) for x in TARGETS)
                lines.append(f"| {year} | {cells} | {row['all_issues']} |")
        lines += ["", "Duplicate evidence volume:", ""]
        for key, value in repo["duplicate_evidence_counts"].items():
            lines.append(f"- {key}: {value}")
        sample = repo.get("duplicate_sample", {})
        if "error" in sample:
            lines += ["", f"Duplicate sample not run: {sample['error']}"]
        elif sample:
            lines += [
                "",
                f"Duplicate sample: {sample['sampled']} of {sample['search_hits']} text hits "
                f"inspected; {sample['explicit_candidates_in_sample']} with an explicit target; "
                f"{len(sample['checked_pairs'])} pairs checked; "
                f"**{sample['verified_pairs']} verified** (issue target, created earlier)."
                + (" Budget exhausted before completion." if sample["budget_exhausted"] else ""),
                "",
            ]
            if sample["checked_pairs"]:
                lines += ["| Query | Target | Evidence | Target PR? | Earlier? | Verified |"]
                lines += ["|---|---|---|---|---|---|"]
                for p in sample["checked_pairs"]:
                    lines.append(
                        f"| [#{p['query']}]({p['query_url']}) "
                        f"| [#{p['target']}]({p['target_url']}) "
                        f"| {p['evidence']} | {p['target_is_pr']} | {p['target_earlier']} "
                        f"| {p['verified']} |"
                    )
        lines.append("")
    return "\n".join(lines)
