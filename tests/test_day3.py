import numpy as np
import pytest

from knotify.data import LABELS
from knotify.selection import choose_threshold, reliability, selective


def curve(rows):
    return [
        {"threshold": t, "accepted": n, "coverage": n / 100, "accepted_accuracy": a}
        for t, n, a in rows
    ]


def test_threshold_prefers_coverage_and_requires_minimum_and_target():
    rows = curve([(0.5, 90, 0.85), (0.8, 40, 0.91), (0.9, 25, 0.95), (0.95, 10, 1.0)])
    assert choose_threshold(rows, target=0.9, minimum=20)["threshold"] == 0.8
    assert choose_threshold(rows, target=0.99, minimum=20) is None  # -> defer all
    assert choose_threshold(curve([(0.9, 0, None)]), target=0.9, minimum=20) is None


def test_reliability_and_selective_metrics_handle_empty_bins_and_zero_acceptance():
    proba = np.array([[0.9, 0.05, 0.03, 0.02], [0.3, 0.3, 0.2, 0.2]])
    y = ["bug", "feature"]
    bins = reliability(proba, y)
    assert sum(b["count"] for b in bins) == 2 and bins[0]["accuracy"] is None
    rows = selective(proba, y, [0.5, 0.99])
    assert rows[0]["accepted"] == 1 and rows[0]["accepted_accuracy"] == 1.0
    assert rows[1]["accepted"] == 0 and rows[1]["accepted_accuracy"] is None


@pytest.fixture
def toy_model():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    from knotify.classifier import TextModel, build_tfidf_lr

    words = {
        "bug": "crash error traceback",
        "feature": "add option support",
        "question_documentation": "docs typo how",
        "other": "ci build pipeline",
    }
    texts, labels = [], []
    for i in range(40):
        for label, w in words.items():
            texts.append(f"{w} sample{i % 7}\n\nbody {w}")
            labels.append(label)
    model = TextModel(build_tfidf_lr(), mask=True)
    model.pipeline.fit(model.transform(texts), labels)
    calibrator = CalibratedClassifierCV(FrozenEstimator(model.pipeline), method="sigmoid")
    calibrator.fit(model.transform(texts), labels)
    return model, calibrator, model.transform(texts)


def test_calibrated_classifier_accepts_defers_and_explains(toy_model):
    from knotify.classifier import CalibratedTextClassifier
    from knotify.explain import TfidfShapExplainer

    model, calibrator, prepared = toy_model
    tfidf = model.pipeline.named_steps["tfidf"]
    mean = np.asarray(tfidf.transform(prepared).mean(axis=0))
    explainer = TfidfShapExplainer(model.pipeline, mean)
    text = "App crash with error\n\ntraceback crash"
    accepted = CalibratedTextClassifier(model, calibrator, 0.5, "o/r", explainer).predict(text)
    assert accepted.predicted_label.value == "bug" and not accepted.deferred
    assert accepted.explanation and accepted.explanation[0].token in {"crash", "error", "traceback"}
    assert "raw score" in accepted.explanation_note
    strict = CalibratedTextClassifier(model, calibrator, 1.0, "o/r").predict(text)
    assert strict.deferred and "below the review threshold" in strict.reason
    none = CalibratedTextClassifier(model, calibrator, None, "o/r").predict(text)
    assert none.deferred and none.calibrated_probabilities is not None
    assert set(none.calibrated_probabilities) == set(none.raw_probabilities)
    assert len(LABELS) == 4


def test_blind_pool_is_deterministic_deduplicated_and_includes_known_targets():
    from knotify.retrieval_eval import blind_pool, stable_seed

    rankings = {"bm25": ["a", "b", "c"], "dense": ["b", "d"], "hybrid": ["b", "a"]}
    pool = blind_pool(rankings, ["z"], depth=2, seed=stable_seed("q"))
    assert sorted(pool) == ["a", "b", "d", "z"]
    assert pool == blind_pool(rankings, ["z"], depth=2, seed=stable_seed("q"))
    assert stable_seed("q") == stable_seed("q") and stable_seed("q") != stable_seed("r")
