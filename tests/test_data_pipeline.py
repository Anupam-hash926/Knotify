import httpx
import pytest

from knotify.audit import extract_duplicate_targets
from knotify.data import (
    RepositoryMapping,
    build_supervised,
    chronological_split,
    clean_markdown,
    map_labels,
    split_validation,
)
from knotify.github import GitHubClient
from knotify.ingest import dedupe, iter_issues, slim

MAPPING = RepositoryMapping(
    "o/r",
    "source",
    {
        "bug": ("Bug", "Regression"),
        "feature": ("Enhancement",),
        "question_documentation": ("Question", "Documentation"),
        "other": ("Build / CI",),
    },
)


def record(number, day, labels, title=None, body="body", hour=0):
    return {
        "id": 1000 + number,
        "number": number,
        "title": title or f"Issue {number}",
        "body": body,
        "labels": labels,
        "created_at": f"2024-01-{day:02d}T{hour:02d}:00:00Z",
    }


def test_label_mapping_conflicts_and_same_target_cooccurrence():
    assert map_labels(["Bug", "module:svm"], MAPPING) == ("bug", "mapped")
    assert map_labels(["Question", "Documentation"], MAPPING) == (
        "question_documentation",
        "mapped",
    )
    assert map_labels(["Bug", "Enhancement"], MAPPING) == (None, "conflict")
    assert map_labels(["Needs Triage"], MAPPING) == (None, "unmapped")


def test_supervised_set_counts_exclusions_and_keeps_earliest_text_duplicate():
    rows = [
        record(3, 3, ["Bug"], title="Same", body="x"),
        record(1, 1, ["Bug"], title="Same", body="x  "),
        record(2, 2, ["Bug", "Enhancement"]),
        record(4, 4, []),
    ]
    result = build_supervised(rows, MAPPING)
    assert [r["number"] for r in result.rows] == [1]
    assert result.exclusions == {"exact_text_duplicate": 1, "conflict": 1, "unmapped": 1}


def test_split_is_chronological_and_keeps_equal_timestamps_together():
    rows = [record(n, 1 + n // 2, ["Bug"]) for n in range(10)]  # pairs share timestamps
    parts = chronological_split(rows, (0.5, 0.25, 0.25))
    train, val, test = parts["train"], parts["validation"], parts["test"]
    assert len(train) + len(val) + len(test) == 10
    assert max(r["created_at"] for r in train) < min(r["created_at"] for r in val)
    assert max(r["created_at"] for r in val) < min(r["created_at"] for r in test)


def test_small_validation_uses_declared_reuse_fallback():
    rows = [{**record(n, 1, ["Bug"], hour=n), "target": "bug"} for n in range(20)]
    result = split_validation(rows)
    assert result["mode"] == "reuse_full_validation"
    assert result["v_cal"] == result["v_policy"]


def test_clean_markdown_drops_template_comments_but_keeps_code():
    text = "<!-- Please describe -->\n```python\nnp.float64(1)\n```\n![img](http://x)"
    cleaned = clean_markdown(text)
    assert "Please describe" not in cleaned and "np.float64(1)" in cleaned and "img" not in cleaned


def test_explicit_duplicate_statements_only():
    repo = "o/r"
    assert extract_duplicate_targets("Duplicate of #12", repo) == [12]
    assert extract_duplicate_targets("see https://github.com/o/r/issues/9, related #3", repo) == []
    assert extract_duplicate_targets("duplicate of https://github.com/x/y/issues/4", repo) == []


def _issue_page(numbers, pr=()):
    return [
        {
            "id": n,
            "number": n,
            "title": f"t{n}",
            "body": None,
            "labels": [{"name": "Bug"}],
            "created_at": f"2024-01-{n:02d}T00:00:00Z",
            "updated_at": "2024-02-01T00:00:00Z",
            **({"pull_request": {}} if n in pr else {}),
        }
        for n in numbers
    ]


def test_import_follows_pagination_excludes_prs_and_dedupes():
    pages = {
        "1": (_issue_page([9, 8, 7], pr=(8,)), '<https://api.github.com/x?page=2>; rel="next"'),
        "2": (_issue_page([7, 6, 5]), None),
    }

    def handler(request):
        page = request.url.params.get("page", "1")
        body, link = pages[page]
        headers = {"link": link.replace("/x", "/repos/o/r/issues")} if link else {}
        return httpx.Response(200, json=body, headers=headers)

    client = GitHubClient("t", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    raw = list(iter_issues(client, "o/r", created_floor="2024-01-06"))
    rows = dedupe(slim(item, "2024-03-01T00:00:00Z") for item in raw)
    assert [r["number"] for r in rows] == [6, 7, 9]  # PR 8 dropped, 7 deduped, 5 below floor
    assert rows[0]["body"] == "" and rows[0]["labels"] == ["Bug"]


def test_client_refuses_writes_and_retries_server_errors():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(502 if len(calls) == 1 else 200, json={"ok": True})

    client = GitHubClient(transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert client.get_json("/rate_limit") == {"ok": True} and len(calls) == 2
    with pytest.raises(ValueError):
        client._request("POST", "/repos/o/r/issues")


def test_untriaged_labels_are_excluded_and_counted():
    mapping = RepositoryMapping("o/r", "source", MAPPING.labels, exclude_labels=("Needs Triage",))
    rows = [record(1, 1, ["Bug", "Needs Triage"]), record(2, 2, ["Bug"])]
    result = build_supervised(rows, mapping)
    assert [r["number"] for r in result.rows] == [2]
    assert result.exclusions == {"excluded_label": 1}


def test_bm25_ignores_future_and_foreign_issues(issue_factory):
    from knotify.retrieval import BM25Retriever, eligible_history

    query = issue_factory(10, day=10, title="KMeans crash with sparse input", body="KMeans sparse")
    past = issue_factory(3, day=3, title="KMeans fails on sparse matrix", body="sparse KMeans")
    noise = issue_factory(4, day=4, title="Docs typo", body="spelling")
    future = issue_factory(20, day=20, title="KMeans crash with sparse input", body="KMeans sparse")
    foreign = issue_factory(
        5, day=5, repository="x/y", title="KMeans crash with sparse input", body="KMeans sparse"
    )
    eligible = eligible_history(query, [past, noise, future, foreign, query])
    results = BM25Retriever().search(query, eligible, top_k=5)
    assert [r.issue.number for r in results] == [3]
    assert results[0].rank == 1 and results[0].method_scores["bm25"] > 0
    assert BM25Retriever().search(query, (), top_k=5) == ()


def test_snapshot_reader_survives_unicode_line_separators(tmp_path):
    from knotify.ingest import _write_jsonl, load_snapshot

    digest = _write_jsonl(tmp_path / "issues.jsonl", [{"body": "a b\x85c"}])
    _write_jsonl(tmp_path / "duplicate_evidence.jsonl", [])
    (tmp_path / "manifest.json").write_text(f'{{"issues_sha256": "{digest}"}}')
    _, records, _ = load_snapshot(tmp_path)
    assert records == [{"body": "a b\x85c"}]


def test_duplicate_markers_are_removed_from_evaluation_queries():
    from knotify.features import strip_duplicate_markers

    text = "Crash in KMeans.\nThis is a duplicate of #12. Traceback follows"
    stripped = strip_duplicate_markers(text)
    assert "#12" not in stripped and "Crash in KMeans" in stripped and "Traceback" in stripped
