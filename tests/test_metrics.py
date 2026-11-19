import pytest

from knotify.evaluate import retrieval_metrics, selective_metrics


def test_recall_is_not_hit_rate():
    result = retrieval_metrics(["a", "x", "y"], {"a", "b"})
    assert result == {"recall": 0.5, "hit": 1.0, "reciprocal_rank": 1.0}


def test_rank_cutoff_and_no_known_positives():
    assert retrieval_metrics(["x", "a"], {"a"}, k=1)["reciprocal_rank"] == 0
    assert retrieval_metrics(["x", "a"], {"a"})["reciprocal_rank"] == 0.5
    assert retrieval_metrics(["x"], set())["recall"] is None
    with pytest.raises(ValueError):
        retrieval_metrics(["a", "a"], {"a"})


def test_zero_acceptance_is_not_perfect_accuracy():
    result = selective_metrics([True, False], [False, False])
    assert result["coverage"] == 0
    assert result["accepted_accuracy"] is None
    assert selective_metrics([], [])["coverage"] is None
