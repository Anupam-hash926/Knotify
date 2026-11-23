import numpy as np
import pytest

from knotify.checklist import RuleChecklist
from knotify.contracts import Category, ChecklistItem, ClassificationResult, RetrievedIssue
from knotify.data import input_hash
from knotify.drafting import compose_reply, template_suggestions, validate_references
from knotify.embeddings import EmbeddingCache
from knotify.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    eligible_history,
    reciprocal_rank_fusion,
)

DEFERRED = ClassificationResult()  # no prediction -> checklist applies, advice conditional

FORM_FILLED = """### Describe the bug
KMeans gives different labels.
### Steps/Code to Reproduce
```python
from sklearn.cluster import KMeans
KMeans(3).fit(X)
```
### Expected Results
Same labels every run.
### Actual Results
```
Traceback (most recent call last):
ValueError: bad input
```
### Versions
```
machine: Linux-5.15-x86_64
sklearn: 1.3.2
```
"""

FORM_EMPTY = """### Describe the bug
It crashes.
### Steps/Code to Reproduce
_No response_
### Expected Results
<!-- what did you expect -->
### Actual Results

### Versions
"""


def statuses(body, issue_factory, classification=DEFERRED):
    items = RuleChecklist().check(issue_factory(body=body), classification)
    return {item.item_id.value: item.status for item in items}


def test_filled_form_is_present_everywhere(issue_factory):
    assert set(statuses(FORM_FILLED, issue_factory).values()) == {"present"}


def test_empty_template_headings_are_missing_not_present(issue_factory):
    result = statuses(FORM_EMPTY, issue_factory)
    for item in ("reproduction_steps", "expected_behavior", "actual_behavior", "version"):
        assert result[item] == "missing"
    assert result["environment"] == "missing"
    assert result["error_logs"] == "missing"  # "crashes" mentioned, no traceback


def test_logs_not_applicable_without_failure_and_free_text_rules(issue_factory):
    body = "On Windows 11 with scikit-learn 1.4.0 the score should be 1.0 but it returns 0.5."
    result = statuses(body, issue_factory)
    assert result["error_logs"] == "not_applicable"
    assert result["environment"] == result["version"] == "present"
    assert result["expected_behavior"] == result["actual_behavior"] == "present"
    assert result["reproduction_steps"] == "missing"


def test_accepted_non_bug_makes_checklist_not_applicable(issue_factory):
    probs = {c: 0.25 for c in Category}
    accepted = ClassificationResult(
        predicted_label=Category.FEATURE,
        raw_probabilities=probs,
        calibrated_probabilities=probs,
        deferred=False,
        reason="accepted",
    )
    assert set(statuses(FORM_EMPTY, issue_factory, accepted).values()) == {"not_applicable"}


def test_templates_cite_missing_items_and_validator_rejects_unknown_ids(issue_factory):
    items = RuleChecklist().check(issue_factory(body=FORM_EMPTY), DEFERRED)
    suggestions = validate_references(template_suggestions(items, DEFERRED), items, ())
    missing = [i for i in items if i.status == "missing"]
    assert len(suggestions) == len(missing) and all(s.reference_valid for s in suggestions)
    assert all(s.text.startswith("If this is a bug report") for s in suggestions)
    assert suggestions[0].semantic_support == "not_checked"
    forged = suggestions[0].model_copy(update={"evidence_ids": ("checklist:unknown",)})
    present_ref = suggestions[0].model_copy(update={"evidence_ids": ("issue:o/r#99",)})
    checked = validate_references((forged, present_ref), items, ())
    assert [s.reference_valid for s in checked] == [False, False]
    reply = compose_reply(suggestions)
    assert "A few details would help" in reply and "- Could you share" in reply
    assert compose_reply(checked) == ""


def test_present_item_cannot_be_cited_as_missing(issue_factory):
    items = RuleChecklist().check(issue_factory(body=FORM_FILLED), DEFERRED)
    assert template_suggestions(items, DEFERRED) == ()
    fake = template_suggestions(
        [ChecklistItem(item_id="environment", status="missing", reason="x")], DEFERRED
    )
    assert validate_references(fake, items, ())[0].reference_valid is False


def test_rrf_scores_ties_and_absent_candidates(issue_factory):
    a, b, c = (issue_factory(n, day=n) for n in (1, 2, 3))
    fused = reciprocal_rank_fusion({"bm25": [(a, 9.0), (b, 5.0)], "dense": [(b, 0.9), (c, 0.8)]})
    assert [i.number for i, _ in fused] == [2, 1, 3]  # b in both lists wins
    assert fused[0][1]["rrf"] == pytest.approx(1 / 62 + 1 / 61)
    assert "dense" not in fused[1][1]  # a absent from dense contributes nothing
    tie = reciprocal_rank_fusion({"bm25": [(b, 1.0)], "dense": [(a, 1.0)]})
    assert [i.issue_id for i, _ in tie] == ["1", "2"]  # equal RRF -> issue ID order


def _cache(issues, vectors):
    return EmbeddingCache(
        np.asarray(vectors, dtype=np.float32),
        [i.issue_id for i in issues],
        [input_hash(i) for i in issues],
    )


def test_future_more_similar_issue_cannot_hide_best_eligible(issue_factory):
    query = issue_factory(10, day=10, title="Query")
    past_good = issue_factory(3, day=3, title="Past good")
    past_bad = issue_factory(4, day=4, title="Past bad")
    future = issue_factory(20, day=20, title="Future twin")
    issues = [query, past_good, past_bad, future]
    vectors = [[1, 0], [0.8, 0.6], [0, 1], [1, 0]]
    for backend in ("numpy", "auto"):
        dense = DenseRetriever(_cache(issues, vectors), depth=1, backend=backend)
        ranked = dense.rank(query, eligible_history(query, issues))
        assert [i.number for i, _ in ranked] == [3]
        assert ranked[0][1] == pytest.approx(0.8)


def test_hybrid_falls_back_to_bm25_when_query_not_cached(issue_factory):
    query = issue_factory(10, day=10, title="KMeans sparse crash", body="KMeans sparse")
    past = issue_factory(3, day=3, title="KMeans sparse failure", body="sparse KMeans")
    dense = DenseRetriever(_cache([past], [[1, 0]]), backend="numpy")  # query not cached
    results = HybridRetriever(BM25Retriever(), dense).search(query, [past], top_k=5)
    assert isinstance(results[0], RetrievedIssue)
    assert "rrf" not in results[0].method_scores and "bm25" in results[0].method_scores
