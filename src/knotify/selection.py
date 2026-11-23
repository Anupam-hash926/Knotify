"""Calibration (V-cal), classifier selection and deferral threshold (V-policy).

* Sigmoid (Platt) calibration of each *frozen* candidate on V-cal (``FrozenEstimator``; the
  classifier is never refitted after calibration).
* Serving classifier = best **raw** macro-F1 on V-policy; a difference below 0.01 prefers the
  simpler TF-IDF model (practical tie rule, not a significance claim).
* Threshold on the calibrated top probability from the declared grid: greatest V-policy coverage
  with accepted accuracy >= target and >= ``minimum_accepted`` accepted examples; otherwise
  defer everything (no threshold).
* Test is never read here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from knotify.config import AppConfig
from knotify.data import LABELS

TIE_MARGIN = 0.01


def align(proba: np.ndarray, classes: list[str]) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(LABELS)))
    for j, label in enumerate(LABELS):
        if label in classes:
            out[:, j] = proba[:, classes.index(label)]
    return out / out.sum(axis=1, keepdims=True)


def log_loss(y: list[str], proba: np.ndarray, eps: float = 1e-12) -> float:
    idx = [LABELS.index(t) for t in y]
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), idx], eps, 1))))


def reliability(proba: np.ndarray, y: list[str], bins: int = 5) -> list[dict[str, Any]]:
    """Top-probability reliability table; empty bins are reported with count 0."""
    conf = proba.max(axis=1)
    correct = np.array([LABELS[i] == t for i, t in zip(proba.argmax(axis=1), y, strict=True)])
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for k in range(bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (conf >= lo) & ((conf < hi) if k < bins - 1 else (conf <= hi))
        n = int(mask.sum())
        rows.append(
            {
                "bin": [round(float(lo), 2), round(float(hi), 2)],
                "count": n,
                "mean_confidence": float(conf[mask].mean()) if n else None,
                "accuracy": float(correct[mask].mean()) if n else None,
            }
        )
    return rows


def selective(proba: np.ndarray, y: list[str], thresholds) -> list[dict[str, Any]]:
    from sklearn.metrics import f1_score

    conf = proba.max(axis=1)
    pred = [LABELS[i] for i in proba.argmax(axis=1)]
    rows = []
    for t in thresholds:
        keep = conf >= t
        n = int(keep.sum())
        yt = [a for a, k in zip(y, keep, strict=True) if k]
        yp = [a for a, k in zip(pred, keep, strict=True) if k]
        rows.append(
            {
                "threshold": t,
                "accepted": n,
                "coverage": n / len(y),
                "accepted_accuracy": float(np.mean([a == b for a, b in zip(yt, yp, strict=True)]))
                if n
                else None,
                "accepted_macro_f1": float(
                    f1_score(yt, yp, labels=list(LABELS), average="macro", zero_division=0)
                )
                if n
                else None,
            }
        )
    return rows


def choose_threshold(curve: list[dict], *, target: float, minimum: int) -> dict[str, Any] | None:
    ok = [
        r
        for r in curve
        if r["accepted"] >= minimum
        and r["accepted_accuracy"] is not None
        and r["accepted_accuracy"] >= target
    ]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["coverage"], -r["threshold"]))


def _candidates(config: AppConfig, repo: str, split_dir: Path, rows: dict) -> dict[str, dict]:
    import joblib

    from knotify.classifier import TextModel
    from knotify.embeddings import EmbeddingCache
    from knotify.experiment import embeddings_dir, slug, text_of

    art = config.paths.artifacts / slug(repo) / split_dir.name
    out: dict[str, dict] = {}
    tfidf = TextModel.load(art / "tfidf_lr_masked.joblib")
    out["tfidf_lr_masked"] = {
        "artifact": art / "tfidf_lr_masked.joblib",
        "estimator": tfidf.pipeline,
        "X": {k: tfidf.transform([text_of(r) for r in v]) for k, v in rows.items()},
        "family": "tfidf",
    }
    emb_path = art / "embedding_lr.joblib"
    if emb_path.exists():
        cache = EmbeddingCache.load(embeddings_dir(config, repo))
        model = joblib.load(emb_path)["model"]

        def matrix(records):
            return np.stack([cache.vectors[cache.index[f"{repo}#{r['number']}"]] for r in records])

        out["embedding_lr"] = {
            "artifact": emb_path,
            "estimator": model,
            "X": {k: matrix(v) for k, v in rows.items()},
            "family": "embedding",
        }
    return out


def calibrate_and_select(config: AppConfig, repo: str) -> dict[str, Any]:
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    from knotify.experiment import classification_metrics, latest_split, load_split_rows, slug

    split_dir = latest_split(config.paths.data, repo)
    split_manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = load_split_rows(config, split_dir, ("v_cal", "v_policy"))
    y_cal = [r["target"] for r in rows["v_cal"]]
    y_pol = [r["target"] for r in rows["v_policy"]]
    cls = config.classification
    art = config.paths.artifacts / slug(repo) / split_dir.name
    report: dict[str, Any] = {
        "repository": repo,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "validation_mode": split_manifest["validation_mode"],
        "v_cal_count": len(y_cal),
        "v_policy_count": len(y_pol),
        "test_used": False,
        "candidates": {},
    }
    for name, cand in _candidates(config, repo, split_dir, rows).items():
        est = cand["estimator"]
        calibrator = CalibratedClassifierCV(FrozenEstimator(est), method="sigmoid")
        calibrator.fit(cand["X"]["v_cal"], y_cal)
        raw = align(est.predict_proba(cand["X"]["v_policy"]), list(est.classes_))
        cal = align(calibrator.predict_proba(cand["X"]["v_policy"]), list(calibrator.classes_))
        cal_path = art / f"calibrator_{name}.joblib"
        joblib.dump({"calibrator": calibrator, "fitted_on": "v_cal", "method": "sigmoid"}, cal_path)
        curve = selective(cal, y_pol, cls.policy_thresholds)
        report["candidates"][name] = {
            "family": cand["family"],
            "artifact": str(cand["artifact"].relative_to(config.paths.artifacts)),
            "calibrator": str(cal_path.relative_to(config.paths.artifacts)),
            "raw": {
                **classification_metrics(y_pol, [LABELS[i] for i in raw.argmax(1)]),
                "log_loss": log_loss(y_pol, raw),
                "reliability": reliability(raw, y_pol),
            },
            "calibrated": {
                **classification_metrics(y_pol, [LABELS[i] for i in cal.argmax(1)]),
                "log_loss": log_loss(y_pol, cal),
                "reliability": reliability(cal, y_pol),
            },
            "selective_curve": curve,
        }
    ranked = sorted(report["candidates"].items(), key=lambda kv: -kv[1]["raw"]["macro_f1"])
    best_name, best = ranked[0]
    tfidf = report["candidates"]["tfidf_lr_masked"]
    if (
        best_name != "tfidf_lr_masked"
        and best["raw"]["macro_f1"] - tfidf["raw"]["macro_f1"] < TIE_MARGIN
    ):
        best_name, best = "tfidf_lr_masked", tfidf
    chosen = choose_threshold(
        best["selective_curve"], target=cls.target_accepted_accuracy, minimum=cls.minimum_accepted
    )
    report["selection"] = {
        "rule": "best raw macro-F1 on V-policy; difference < 0.01 prefers TF-IDF",
        "chosen": best_name,
        "raw_macro_f1": {k: v["raw"]["macro_f1"] for k, v in report["candidates"].items()},
    }
    report["policy"] = {
        "rule": (
            f"greatest V-policy coverage with accepted accuracy >= {cls.target_accepted_accuracy} "
            f"and >= {cls.minimum_accepted} accepted; grid {list(cls.policy_thresholds)}"
        ),
        "threshold": chosen["threshold"] if chosen else None,
        "v_policy_outcome": chosen,
        "defer_all": chosen is None,
    }
    serving = {
        "repository": repo,
        "split": split_dir.name,
        "model": best_name,
        "family": best["family"],
        "artifact": best["artifact"],
        "calibrator": best["calibrator"],
        "threshold": report["policy"]["threshold"],
        "calibration_repository": repo,
        "frozen_at": report["created_at"],
    }
    (art / "serving.json").write_text(json.dumps(serving, indent=2) + "\n", encoding="utf-8")
    run_dir = config.paths.runs / "baseline" / slug(repo) / split_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "calibration_selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    report["serving"] = serving
    return report
