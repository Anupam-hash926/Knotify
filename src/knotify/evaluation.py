"""Day 4 evaluation runner: the only code that reads the Test split.

Uses exactly the frozen files recorded in ``docs/freeze/day3_freeze.json`` (refuses to run if
any changed). Nothing here selects or tunes anything; Test curves are reported, not used.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from knotify.config import PROJECT_ROOT, AppConfig
from knotify.data import LABELS

FREEZE = PROJECT_ROOT / "docs" / "freeze" / "day3_freeze.json"
BOOTSTRAP_RESAMPLES = 500


def check_freeze() -> dict:
    from knotify.freeze import verify_freeze

    problems = verify_freeze(FREEZE)
    if problems:
        raise RuntimeError(f"Frozen files changed since Day 3: {problems}. Refusing to evaluate.")
    return json.loads(FREEZE.read_text(encoding="utf-8"))


def _macro_f1(y, pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, pred, labels=list(LABELS), average="macro", zero_division=0))


def paired_bootstrap(y, pred_a, pred_b, *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES):
    """95% percentile interval for macro-F1(a) - macro-F1(b), resampling issues jointly."""
    rng = np.random.default_rng(seed)
    y, pred_a, pred_b = np.array(y), np.array(pred_a), np.array(pred_b)
    diffs = []
    for _ in range(resamples):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(_macro_f1(y[idx], pred_a[idx]) - _macro_f1(y[idx], pred_b[idx]))
    return {
        "difference": _macro_f1(y, pred_a) - _macro_f1(y, pred_b),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "resamples": resamples,
        "seed": seed,
    }


def bootstrap_mean(values, *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(resamples)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ----------------------------------------------------------------------------- classification
def evaluate_classification(config: AppConfig, repo: str) -> dict[str, Any]:
    import joblib

    from knotify.classifier import MajorityBaseline, TextModel
    from knotify.embeddings import EmbeddingCache
    from knotify.experiment import (
        classification_metrics,
        embeddings_dir,
        latest_split,
        load_split_rows,
        slug,
        text_of,
    )
    from knotify.selection import align, log_loss, reliability, selective

    split = latest_split(config.paths.data, repo)
    art = config.paths.artifacts / slug(repo) / split.name
    serving = json.loads((art / "serving.json").read_text(encoding="utf-8"))
    rows = load_split_rows(config, split, ("train", "test"))
    train, test = rows["train"], rows["test"]
    y = [r["target"] for r in test]
    texts = [text_of(r) for r in test]
    preds: dict[str, list[str]] = {}
    out: dict[str, Any] = {"test_count": len(test), "test_support": {}, "models": {}}
    out["test_support"] = {label: y.count(label) for label in LABELS}

    majority = MajorityBaseline().fit([], [r["target"] for r in train])
    preds["majority"] = majority.predict(texts)
    for name in ("tfidf_lr_masked", "tfidf_lr_raw"):
        model = TextModel.load(art / f"{name}.joblib")
        preds[name] = [LABELS[i] for i in model.predict_proba(texts).argmax(1)]
    cache = EmbeddingCache.load(embeddings_dir(config, repo))
    emb = joblib.load(art / "embedding_lr.joblib")["model"]
    matrix = np.stack([cache.vectors[cache.index[f"{repo}#{r['number']}"]] for r in test])
    emb_raw = align(emb.predict_proba(matrix), list(emb.classes_))
    preds["embedding_lr"] = [LABELS[i] for i in emb_raw.argmax(1)]
    for name, pred in preds.items():
        out["models"][name] = classification_metrics(y, pred)

    # Selected (serving) model: raw vs calibrated, coverage at the frozen threshold.
    model = TextModel.load(config.paths.artifacts / serving["artifact"])
    calibrator = joblib.load(config.paths.artifacts / serving["calibrator"])["calibrator"]
    features = model.transform(texts)
    raw = align(model.pipeline.predict_proba(features), list(model.pipeline.classes_))
    cal = align(calibrator.predict_proba(features), list(calibrator.classes_))
    curve = selective(cal, y, config.classification.policy_thresholds)
    frozen = next((r for r in curve if r["threshold"] == serving["threshold"]), None)
    emb_cal = joblib.load(art / "calibrator_embedding_lr.joblib")["calibrator"]
    emb_cal_p = align(emb_cal.predict_proba(matrix), list(emb_cal.classes_))
    out["selected"] = {
        "model": serving["model"],
        "threshold": serving["threshold"],
        "raw_log_loss": log_loss(y, raw),
        "calibrated_log_loss": log_loss(y, cal),
        "raw_reliability": reliability(raw, y),
        "calibrated_reliability": reliability(cal, y),
        "calibrated_macro_f1_all": classification_metrics(y, [LABELS[i] for i in cal.argmax(1)])[
            "macro_f1"
        ],
        "coverage_curve": curve,
        "at_frozen_threshold": frozen,
        "coverage_at_fixed_f1": "omitted (secondary proposal measure; not pre-registered)",
    }
    out["embedding_lr_log_loss"] = {
        "raw": log_loss(y, emb_raw),
        "calibrated": log_loss(y, emb_cal_p),
    }
    out["bootstrap"] = {
        "tfidf_masked_minus_embedding": paired_bootstrap(
            y, preds["tfidf_lr_masked"], preds["embedding_lr"], seed=config.seed
        ),
        "tfidf_raw_minus_masked": paired_bootstrap(
            y, preds["tfidf_lr_raw"], preds["tfidf_lr_masked"], seed=config.seed
        ),
    }
    out["_predictions"] = {"numbers": [r["number"] for r in test], "target": y, **preds}
    return out


# ----------------------------------------------------------------------------- transfer
def evaluate_transfer(config: AppConfig, source: str, target: str) -> dict[str, Any]:
    import joblib

    from knotify.classifier import TextModel
    from knotify.data import build_supervised, class_counts, load_mappings
    from knotify.experiment import classification_metrics, latest_snapshot, slug, text_of
    from knotify.ingest import load_snapshot
    from knotify.selection import align, selective

    split_name = json.loads(FREEZE.read_text(encoding="utf-8"))["serving"]["split"]
    art = config.paths.artifacts / slug(source) / split_name
    serving = json.loads((art / "serving.json").read_text(encoding="utf-8"))
    snapshot = latest_snapshot(config.paths.data, target)
    manifest, records, _ = load_snapshot(snapshot)
    mapping = load_mappings(PROJECT_ROOT / "configs" / "repositories.toml")[target]
    supervised = build_supervised(records, mapping)
    y = [r["target"] for r in supervised.rows]
    texts = [text_of(r) for r in supervised.rows]
    model = TextModel.load(config.paths.artifacts / serving["artifact"])
    calibrator = joblib.load(config.paths.artifacts / serving["calibrator"])["calibrator"]
    features = model.transform(texts)
    raw = align(model.pipeline.predict_proba(features), list(model.pipeline.classes_))
    cal = align(calibrator.predict_proba(features), list(calibrator.classes_))
    frozen = selective(cal, y, [serving["threshold"]])[0]
    unmasked = TextModel.load(art / "tfidf_lr_raw.joblib")
    return {
        "source": source,
        "target": target,
        "target_snapshot": snapshot.name,
        "target_window": manifest.get("created_window"),
        "count": len(y),
        "support": class_counts(supervised.rows),
        "exclusions": supervised.exclusions,
        "selected_model_raw": classification_metrics(y, [LABELS[i] for i in raw.argmax(1)]),
        "at_frozen_threshold": frozen,
        "unmasked_model_raw": classification_metrics(
            y, [LABELS[i] for i in unmasked.predict_proba(texts).argmax(1)]
        ),
        "note": "Calibration and threshold were fitted on the source repository; their "
        "reliability on the target is not verified. B labels are used for scoring only.",
    }


# ----------------------------------------------------------------------------- retrieval
def verify_files(names: list[str]) -> None:
    from knotify.freeze import _sha

    manifest = json.loads(FREEZE.read_text(encoding="utf-8"))
    for name in names:
        entry = manifest["files"][name]
        if _sha(PROJECT_ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Frozen file changed: {name}")


def evaluate_retrieval(config: AppConfig, repo: str, encoder) -> dict[str, Any]:
    """BM25 / dense / hybrid on the frozen evaluation queries (explicit duplicate relevance).

    Explicit duplicate markers are stripped from query text. Encoding uses the local model.
    """
    from knotify.data import model_text
    from knotify.embeddings import EmbeddingCache
    from knotify.evaluate import retrieval_metrics
    from knotify.experiment import embeddings_dir, latest_snapshot, latest_split, slug
    from knotify.features import strip_duplicate_markers
    from knotify.ingest import load_snapshot, to_issue
    from knotify.retrieval import BM25Retriever, DenseRetriever
    from knotify.retrieval_eval import method_rankings

    verify_files(["snapshot_issues", "retrieval_queries", "embedding_vectors"])
    split = latest_split(config.paths.data, repo)
    ann = config.paths.data / "annotations" / slug(repo) / split.name
    queries = json.loads((ann / "retrieval_queries.json").read_text(encoding="utf-8"))
    _, records, _ = load_snapshot(latest_snapshot(config.paths.data, repo))
    history = [to_issue(r, repo) for r in records]
    by_number = {i.number: i for i in history}

    def transform(issue):
        return strip_duplicate_markers(model_text(issue))

    depth, k = config.retrieval.candidate_depth, config.retrieval.rrf_k
    bm25 = BM25Retriever(depth=depth, query_transform=transform)
    dense = DenseRetriever(
        EmbeddingCache.load(embeddings_dir(config, repo)),
        encoder=encoder,
        depth=depth,
        query_transform=transform,
    )
    per_query = []
    for item in queries["evaluation"]:
        issue = by_number[item["query"]]
        relevant = {f"{repo}#{r['target']}" for r in item["relevant"]}
        rankings = method_rankings(issue, history, bm25, dense, rrf_k=k)
        row = {"query": item["query"], "relevant": sorted(relevant), "top5": {}}
        for method, ranked in rankings.items():
            row[method] = retrieval_metrics(ranked[:5], relevant, k=5)
            row["top5"][method] = ranked[:5]
            first = next((i + 1 for i, x in enumerate(ranked) if x in relevant), None)
            row[method]["first_relevant_rank_top50"] = first
        per_query.append(row)
    methods = ("bm25", "dense", "hybrid")
    summary: dict[str, Any] = {}
    for method in methods:
        summary[method] = {}
        for metric in ("recall", "hit", "reciprocal_rank"):
            values = [q[method][metric] for q in per_query]
            name = {"recall": "recall@5", "hit": "hit@5", "reciprocal_rank": "mrr@5"}[metric]
            summary[method][name] = float(np.mean(values))
            summary[method][f"{name}_ci95"] = bootstrap_mean(values, seed=config.seed)
    rng = np.random.default_rng(config.seed)
    diffs = {}
    for a, b in (("hybrid", "bm25"), ("hybrid", "dense"), ("dense", "bm25")):
        va = np.array([q[a]["reciprocal_rank"] for q in per_query])
        vb = np.array([q[b]["reciprocal_rank"] for q in per_query])
        boots = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            idx = rng.integers(0, len(va), len(va))
            boots.append(float(va[idx].mean() - vb[idx].mean()))
        diffs[f"mrr@5_{a}_minus_{b}"] = {
            "difference": float(va.mean() - vb.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    return {
        "queries": len(per_query),
        "relevance": "explicit duplicate statements (one known target per query)",
        "query_periods": {
            "validation": sum(q["query_created_at"] < "2024-09-11" for q in queries["evaluation"]),
            "test": sum(q["query_created_at"] >= "2024-09-11" for q in queries["evaluation"]),
        },
        "dense_backend": dense.backend,
        "summary": summary,
        "paired_bootstrap": diffs,
        "per_query": per_query,
        "note": "Recall against known explicit duplicates only, not an exhaustive relevant set. "
        "MRR is truncated at 5. Hit@5 equals Recall@5 here because each query has one target.",
    }


# ----------------------------------------------------------------------------- evidence/latency
def evidence_mechanical(config: AppConfig, repo: str) -> dict[str, Any]:
    """Reference checks on the frozen sample's template questions (no human input needed)."""
    from knotify.experiment import latest_split, slug

    verify_files(["evaluation_samples"])
    split = latest_split(config.paths.data, repo)
    path = config.paths.data / "annotations" / slug(repo) / split.name / "evaluation_samples.json"
    samples = json.loads(path.read_text(encoding="utf-8"))["checklist"]
    suggestions = [s for e in samples for s in e["suggestions"]]
    return {
        "issues": len(samples),
        "suggestions": len(suggestions),
        "invalid_references": sum(not s["reference_valid"] for s in suggestions),
        "retained_after_validation": sum(s["reference_valid"] for s in suggestions),
        "issues_with_no_suggestion": sum(not e["suggestions"] for e in samples),
        "per_item": {
            item: sum(s["evidence_ids"] == [f"checklist:{item}"] for s in suggestions)
            for item in (
                "reproduction_steps",
                "expected_behavior",
                "actual_behavior",
                "environment",
                "version",
                "error_logs",
            )
        },
        "note": "Mechanical reference validity only. Whether suggestions are supported and "
        "appropriate comes from the human suggestion sheet.",
    }


def measure_latency(config: AppConfig, repo: str, *, issues: int = 30, repeats: int = 3) -> dict:
    """Median warm analyze() latency over Test-period issues (runtime loaded once, not timed)."""
    import os
    import platform
    import statistics
    from time import perf_counter

    from knotify.experiment import latest_split, load_split_rows
    from knotify.ingest import to_issue
    from knotify.runtime import load_runtime
    from knotify.triage import analyze

    started = perf_counter()
    runtime = load_runtime(config, repo)
    load_seconds = perf_counter() - started
    split = latest_split(config.paths.data, repo)
    test = load_split_rows(config, split, ("test",))["test"][:issues]
    sample = [to_issue(r, repo) for r in test]
    analyze(sample[0], runtime.history, runtime.bundle, config)  # warm-up (token cache)
    times = []
    for _ in range(repeats):
        for issue in sample:
            t = perf_counter()
            analyze(issue, runtime.history, runtime.bundle, config)
            times.append((perf_counter() - t) * 1000)
    retriever = type(runtime.bundle.retriever).__name__
    return {
        "runs": len(times),
        "median_ms": statistics.median(times),
        "p90_ms": float(np.percentile(times, 90)),
        "runtime_load_seconds": load_seconds,
        "retriever": retriever,
        "llm_calls": 0,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "notes": list(runtime.notes),
    }


# ----------------------------------------------------------------------------- report
def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def _f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def render_results(run_dir: Path) -> str:
    def load(name):
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    c, tr, r = load("classification.json"), load("transfer.json"), load("retrieval.json")
    ev, ck, sg = load("evidence_mechanical.json"), load("checklist.json"), load("suggestions.json")
    lat = sorted(run_dir.glob("latency_*.json"))
    lines = [
        "# Knotify — evaluation results",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from the Day 3 freeze "
        "(`docs/freeze/day3_freeze.json`). Test split read only by `knotify.evaluation`.",
        "",
    ]
    if c:
        lines += [
            f"## 1. Category classification — scikit-learn Test (n = {c['test_count']})",
            "",
            "Support: " + ", ".join(f"{k} {v}" for k, v in c["test_support"].items()),
            "",
            "| Model | Macro-F1 | Accuracy | bug | feature | question/docs | other |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        names = {
            "majority": "Majority class",
            "tfidf_lr_masked": "TF-IDF LR, template cues masked (**served**)",
            "tfidf_lr_raw": "TF-IDF LR, unmasked text",
            "embedding_lr": "MiniLM embedding LR",
        }
        for key, label in names.items():
            m = c["models"][key]
            pc = m["per_class_f1"]
            lines.append(
                f"| {label} | {_f(m['macro_f1'])} | {_f(m['accuracy'])} | {_f(pc['bug'])} | "
                f"{_f(pc['feature'])} | {_f(pc['question_documentation'])} | {_f(pc['other'])} |"
            )
        b = c["bootstrap"]
        s = c["selected"]
        lines += [
            "",
            "Paired bootstrap (500 resamples, seed 42), macro-F1 differences: masked TF-IDF − "
            f"embedding {b['tfidf_masked_minus_embedding']['difference']:+.3f} "
            f"(95% CI {b['tfidf_masked_minus_embedding']['ci95'][0]:+.3f} to "
            f"{b['tfidf_masked_minus_embedding']['ci95'][1]:+.3f}); unmasked − masked "
            f"{b['tfidf_raw_minus_masked']['difference']:+.3f} (95% CI "
            f"{b['tfidf_raw_minus_masked']['ci95'][0]:+.3f} to "
            f"{b['tfidf_raw_minus_masked']['ci95'][1]:+.3f}).",
            "",
            "### Calibration and deferral (served model)",
            "",
            f"Log loss raw {_f(s['raw_log_loss'])} → calibrated {_f(s['calibrated_log_loss'])} "
            f"(embedding LR {_f(c['embedding_lr_log_loss']['raw'])} → "
            f"{_f(c['embedding_lr_log_loss']['calibrated'])}).",
            "",
            "| Calibrated top-probability bin | Count | Mean confidence | Accuracy |",
            "|---|---:|---:|---:|",
        ]
        for row in s["calibrated_reliability"]:
            lines.append(
                f"| {row['bin'][0]:.1f}–{row['bin'][1]:.1f} | {row['count']} | "
                f"{_f(row['mean_confidence'], 2)} | {_f(row['accuracy'], 2)} |"
            )
        lines += [
            "",
            "| Threshold | Accepted | Coverage | Accepted accuracy | Accepted macro-F1 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in s["coverage_curve"]:
            mark = " (**frozen**)" if row["threshold"] == s["threshold"] else ""
            lines.append(
                f"| {row['threshold']:.2f}{mark} | {row['accepted']} | {_pct(row['coverage'])} | "
                f"{_pct(row['accepted_accuracy'])} | {_f(row['accepted_macro_f1'])} |"
            )
        lines += [
            "",
            "The threshold was chosen on V-policy before Test was read (rule: greatest coverage "
            "with ≥ 90% accepted accuracy and ≥ 20 accepted). Test rows illustrate behaviour and "
            "were not used to pick it. Coverage at a fixed F1 is omitted (not pre-registered).",
            "",
        ]
    if tr:
        m, u, fz = tr["selected_model_raw"], tr["unmasked_model_raw"], tr["at_frozen_threshold"]
        lines += [
            f"## 2. Transfer — frozen scikit-learn model on pandas (n = {tr['count']})",
            "",
            "Support: "
            + ", ".join(f"{k} {v}" for k, v in tr["support"].items())
            + f". Window {tr['target_window'][0]} → {tr['target_window'][1]} (after the source's "
            "validation cutoff).",
            "",
            "| Model | Macro-F1 | Accuracy | bug | feature | question/docs | other |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, mm in (("Served (masked)", m), ("Unmasked text", u)):
            pc = mm["per_class_f1"]
            lines.append(
                f"| {label} | {_f(mm['macro_f1'])} | {_f(mm['accuracy'])} | {_f(pc['bug'])} | "
                f"{_f(pc['feature'])} | {_f(pc['question_documentation'])} | {_f(pc['other'])} |"
            )
        lines += [
            "",
            f"At the frozen threshold {fz['threshold']}: {fz['accepted']} accepted "
            f"(coverage {_pct(fz['coverage'])}; "
            f"accepted accuracy {_pct(fz['accepted_accuracy'])}). " + tr["note"],
            "",
        ]
    if r:
        lines += [
            f"## 3. Related-issue retrieval — {r['queries']} frozen queries",
            "",
            f"Queries: {r['query_periods']['validation']} validation-period, "
            f"{r['query_periods']['test']} test-period. Relevance: {r['relevance']}. "
            f"Dense backend: {r['dense_backend']}.",
            "",
            "| Method | Recall@5 = Hit@5 (95% CI) | MRR@5 (95% CI) |",
            "|---|---|---|",
        ]
        for method in ("bm25", "dense", "hybrid"):
            sm = r["summary"][method]
            lines.append(
                f"| {method} | {_f(sm['recall@5'])} ({_f(sm['recall@5_ci95'][0])}–"
                f"{_f(sm['recall@5_ci95'][1])}) | {_f(sm['mrr@5'])} ({_f(sm['mrr@5_ci95'][0])}–"
                f"{_f(sm['mrr@5_ci95'][1])}) |"
            )
        lines += ["", "Paired bootstrap MRR@5 differences:"]
        for key, val in r["paired_bootstrap"].items():
            lines.append(
                f"- {key.replace('mrr@5_', '').replace('_', ' ')}: {val['difference']:+.3f} "
                f"(95% CI {val['ci95'][0]:+.3f} to {val['ci95'][1]:+.3f})"
            )
        lines += ["", r["note"], ""]
    if c and r:
        s, m = c["selected"], c["models"]["tfidf_lr_masked"]
        fz = s["at_frozen_threshold"]
        lines += [
            "## Classical vs selected pipeline (same examples)",
            "",
            "| Component | Classical (TF-IDF + BM25) | Selected pipeline |",
            "|---|---|---|",
            f"| Category (Test, n = {c['test_count']}) | always answers: coverage 100%, accuracy "
            f"{_pct(m['accuracy'])} | calibrated + deferral: coverage {_pct(fz['coverage'])}, "
            f"accepted accuracy {_pct(fz['accepted_accuracy'])}; rest deferred to a maintainer |",
            f"| Related issues ({r['queries']} queries) | BM25 Recall@5 "
            f"{_f(r['summary']['bm25']['recall@5'])}, MRR@5 {_f(r['summary']['bm25']['mrr@5'])} | "
            f"hybrid Recall@5 {_f(r['summary']['hybrid']['recall@5'])}, MRR@5 "
            f"{_f(r['summary']['hybrid']['mrr@5'])} |",
            "| Missing information | none | rule checklist: "
            + (
                f"missing-item F1 {_f(ck['overall']['f1'])} on human-confirmed bug reports"
                if ck
                else "pending human sheet"
            )
            + " (sections 4–5) |",
            "",
            "Both pipelines use the same TF-IDF classifier (selected on validation), so the "
            "category difference is calibration and deferral, not a different model.",
            "",
        ]
    lines += ["## 4. Missing-information checklist", ""]
    if ck:
        lines += [ck.get("markdown", "")]
    else:
        lines += ["*Pending: needs the human checklist sheet.*", ""]
    lines += ["## 5. Evidence-linked suggestions", ""]
    if ev:
        lines += [
            f"Mechanical checks on {ev['suggestions']} template questions from {ev['issues']} "
            f"sampled bug reports: invalid references {ev['invalid_references']}, retained "
            f"{ev['retained_after_validation']}, reports with no question "
            f"{ev['issues_with_no_suggestion']}.",
            "",
        ]
    lines += [
        sg.get("markdown", "")
        if sg
        else "*Manual support check pending: needs the human suggestion sheet.*",
        "",
    ]
    lines += ["## 6. Latency (warm `analyze()`, no LLM calls)", ""]
    if lat:
        lines += [
            "| Machine | Runs | Median | p90 | One-time load | Retriever |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for path in lat:
            d = json.loads(path.read_text(encoding="utf-8"))
            hw = d["hardware"]
            lines.append(
                f"| {hw['platform'][:40]} ({hw['cpu_count']} CPU) | {d['runs']} | "
                f"{d['median_ms'] / 1000:.2f} s | {d['p90_ms'] / 1000:.2f} s | "
                f"{d['runtime_load_seconds']:.1f} s | {d['retriever']} |"
            )
    lines += [
        "",
        "## 7. Not completed",
        "",
        "- LLM-only comparison and LLM drafting/grounding ablation (deferred by the team).",
        "- Pooled 'related' relevance judgements (optional retrieval sheet skipped); retrieval "
        "uses explicit-duplicate relevance only.",
        "- Target MacBook not tested.",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- human sheets
ITEMS = (
    "reproduction_steps",
    "expected_behavior",
    "actual_behavior",
    "environment",
    "version",
    "error_logs",
)


def _sheet(workbook, title: str):
    """Find the data tab by name: Excel may reorder tabs when the file is saved."""
    if title in workbook.sheetnames:
        return workbook[title]
    return next(ws for ws in workbook.worksheets if ws.title != "How to fill")


def read_checklist_sheet(path: Path) -> dict[int, dict[str, Any]]:
    from openpyxl import load_workbook

    ws = _sheet(load_workbook(path, read_only=True), "Checklist")
    rows = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        r = tuple(r) + (None,) * (13 - len(r))  # Excel may store short/empty rows
        if r[1] is None:
            continue
        clean = [(str(v).strip().lower() if v is not None else "") for v in r]
        rows[int(r[1])] = {
            "is_bug": clean[4],
            "items": dict(zip(ITEMS, clean[5:11], strict=True)),
            "initials": str(r[11] or "").strip(),
        }
    return rows


def score_checklist(samples: list[dict], gold: dict[int, dict]) -> dict[str, Any]:
    """Missing = positive class, on items the human judged present/missing (applicable).

    Predicted uncertain on a truly missing item counts as a miss. Gold uncertain/not_applicable
    items are excluded from binary scoring and counted. Non-bug reports are excluded.
    """
    per_item = {i: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for i in ITEMS}
    excluded = {
        "not_bug_or_blank": 0,
        "gold_uncertain": 0,
        "gold_not_applicable": 0,
        "gold_blank": 0,
    }
    predicted_uncertain = 0
    scored_items = 0
    for entry in samples:
        g = gold.get(entry["number"])
        if not g or g["is_bug"] != "yes":
            excluded["not_bug_or_blank"] += 1
            continue
        for item in ITEMS:
            truth, pred = g["items"][item], entry["rule_output"][item]
            if truth in ("uncertain", "not_applicable", ""):
                key = {
                    "uncertain": "gold_uncertain",
                    "not_applicable": "gold_not_applicable",
                    "": "gold_blank",
                }[truth]
                excluded[key] += 1
                continue
            scored_items += 1
            predicted_uncertain += pred == "uncertain"
            positive_truth, positive_pred = truth == "missing", pred == "missing"
            cell = per_item[item]
            if positive_truth and positive_pred:
                cell["tp"] += 1
            elif positive_pred:
                cell["fp"] += 1
            elif positive_truth:
                cell["fn"] += 1
            else:
                cell["tn"] += 1

    def prf(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
        f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        return {"precision": p, "recall": r, "f1": f, **c}

    total = {k: sum(c[k] for c in per_item.values()) for k in ("tp", "fp", "fn", "tn")}
    return {
        "bug_reports_scored": sum(
            1 for e in samples if gold.get(e["number"], {}).get("is_bug") == "yes"
        ),
        "items_scored": scored_items,
        "overall": prf(total),
        "per_item": {item: prf(c) for item, c in per_item.items()},
        "excluded": excluded,
        "predicted_uncertain_on_scored_items": predicted_uncertain,
        "reviewers": sorted({g["initials"] for g in gold.values() if g["initials"]}),
    }


def prefill_suggestion_support(suggestion_sheet: Path, gold: dict[int, dict]) -> int:
    """Fill column F from the same person's checklist judgement; marks provenance in Notes."""
    from openpyxl import load_workbook

    wb = load_workbook(suggestion_sheet)
    ws = _sheet(wb, "Suggestions")
    filled = 0
    for row in range(2, ws.max_row + 1):
        number, item = ws.cell(row, 2).value, ws.cell(row, 5).value
        g = gold.get(int(number)) if number is not None else None
        if not g or g["is_bug"] != "yes" or ws.cell(row, 6).value:
            continue
        truth = g["items"].get(item, "")
        answer = {"missing": "yes", "present": "no"}.get(truth, "unclear")
        ws.cell(row, 6).value = answer
        note = ws.cell(row, 9).value or ""
        ws.cell(row, 9).value = (note + " " if note else "") + "[F derived from checklist sheet]"
        filled += 1
    wb.save(suggestion_sheet)
    return filled


def score_suggestions(path: Path) -> dict[str, Any]:
    from openpyxl import load_workbook

    ws = _sheet(load_workbook(path, read_only=True), "Suggestions")
    rows = [
        tuple(r) + (None,) * (9 - len(r))
        for r in ws.iter_rows(min_row=2, values_only=True)
        if len(r) > 1 and r[1] is not None
    ]

    def norm(v):
        return str(v).strip().lower() if v is not None else ""

    supported = [norm(r[5]) for r in rows]
    appropriate = [norm(r[6]) for r in rows]
    return {
        "suggestions": len(rows),
        "issues": len({r[1] for r in rows}),
        "supported_yes": supported.count("yes"),
        "unsupported_no": supported.count("no"),
        "support_unclear_or_blank": len(rows) - supported.count("yes") - supported.count("no"),
        "appropriate_yes": appropriate.count("yes"),
        "inappropriate_no": appropriate.count("no"),
        "appropriate_blank": appropriate.count(""),
        "unsupported_rate": supported.count("no") / len(rows) if rows else None,
    }


def human_sections(config: AppConfig, repo: str, run_dir: Path) -> dict[str, bool]:
    """Score the filled checklist and suggestion sheets, if present, into the run directory."""
    from openpyxl import load_workbook

    from knotify.experiment import latest_split, slug

    verify_files(["evaluation_samples"])
    split = latest_split(config.paths.data, repo)
    ann = config.paths.data / "annotations" / slug(repo) / split.name
    samples = json.loads((ann / "evaluation_samples.json").read_text(encoding="utf-8"))
    done = {"checklist": False, "suggestions": False}
    sheet = ann / "checklist_review.xlsx"
    if not sheet.exists():
        return done
    gold = read_checklist_sheet(sheet)
    judged = [g for g in gold.values() if g["is_bug"]]
    if len(judged) < len(gold):
        return done
    ck = score_checklist(samples["checklist"], gold)
    o = ck["overall"]
    not_bug = [n for n, g in gold.items() if g["is_bug"] != "yes"]
    lines = [
        f"Human gold: {len(gold)} bug-labelled reports judged by {', '.join(ck['reviewers'])}; "
        f"**{len(not_bug)} of {len(gold)} were judged not to be bug reports** despite the `Bug` "
        "label and are excluded. Rules were frozen (checklist-v1) before judging; rule outputs "
        "were hidden from the sheet.",
        "",
        f"Missing = positive class on {ck['items_scored']} applicable items from "
        f"{ck['bug_reports_scored']} bug reports: **precision {_f(o['precision'])}, recall "
        f"{_f(o['recall'])}, F1 {_f(o['f1'])}** (TP {o['tp']}, FP {o['fp']}, FN {o['fn']}, "
        f"TN {o['tn']}).",
        "",
        "| Item | Truly missing | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item, v in ck["per_item"].items():
        lines.append(
            f"| {item} | {v['tp'] + v['fn']} | {_f(v['precision'])} | {_f(v['recall'])} | "
            f"{_f(v['f1'])} |"
        )
    ex = ck["excluded"]
    lines += [
        "",
        f"Excluded from binary scoring: gold uncertain {ex['gold_uncertain']}, gold "
        f"not-applicable {ex['gold_not_applicable']}. Rule predicted `uncertain` on "
        f"{ck['predicted_uncertain_on_scored_items']} scored items (counted as not detecting "
        "missing). Per-item counts are small; rely on the overall figure. "
        "The LLM-only checklist comparison was not run (deferred).",
        "",
    ]
    ck["markdown"] = "\n".join(lines)
    (run_dir / "checklist.json").write_text(json.dumps(ck, indent=2), encoding="utf-8")
    done["checklist"] = True

    sug_path = ann / "suggestion_review.xlsx"
    if sug_path.exists():
        ws = _sheet(load_workbook(sug_path, read_only=True), "Suggestions")
        rows = [
            tuple(r) + (None,) * (9 - len(r))
            for r in ws.iter_rows(min_row=2, values_only=True)
            if len(r) > 1 and r[1] is not None
        ]

        def norm(v):
            return str(v).strip().lower() if v is not None else ""

        if rows and all(norm(r[6]) in ("yes", "no") for r in rows):
            overall = score_suggestions(sug_path)
            bug_rows = [r for r in rows if gold.get(int(r[1]), {}).get("is_bug") == "yes"]
            other_rows = [r for r in rows if r not in bug_rows]

            def count(rs, col, value):
                return sum(norm(r[col]) == value for r in rs)

            on_bugs = {
                "suggestions": len(bug_rows),
                "issues": len({r[1] for r in bug_rows}),
                "supported": count(bug_rows, 5, "yes"),
                "unsupported": count(bug_rows, 5, "no"),
                "unclear": len(bug_rows) - count(bug_rows, 5, "yes") - count(bug_rows, 5, "no"),
                "appropriate": count(bug_rows, 6, "yes"),
            }
            on_other = {
                "suggestions": len(other_rows),
                "issues": len({r[1] for r in other_rows}),
                "appropriate": count(other_rows, 6, "yes"),
            }
            ev = json.loads((run_dir / "evidence_mechanical.json").read_text(encoding="utf-8"))
            md = [
                f"Manual check of all {overall['suggestions']} template questions "
                f'({overall["issues"]} issues). Column F ("is the cited item really missing?") '
                "was derived from the same reviewer's checklist judgements; column G "
                '("appropriate to ask?") was judged directly.',
                "",
                "| Subset | Questions | Supported | Unsupported | Unclear | Appropriate |",
                "|---|---:|---:|---:|---:|---:|",
                f"| Human-confirmed bug reports ({on_bugs['issues']} issues) | "
                f"{on_bugs['suggestions']} | {on_bugs['supported']} | {on_bugs['unsupported']} | "
                f"{on_bugs['unclear']} | {on_bugs['appropriate']} |",
                f"| Issues judged not bug reports ({on_other['issues']}) | "
                f"{on_other['suggestions']} | — | — | {on_other['suggestions']} | "
                f"{on_other['appropriate']} |",
                f"| All | {overall['suggestions']} | {overall['supported_yes']} | "
                f"{overall['unsupported_no']} | {overall['support_unclear_or_blank']} | "
                f"{overall['appropriate_yes']} |",
                "",
                f"Unsupported-suggestion rate: **{ev['invalid_references']}/{ev['suggestions']}** "
                "by the proposal's definition (a suggestion citing no retrieved issue or checklist "
                f"item); **{overall['unsupported_no']}/{overall['suggestions']}** manually "
                "unsupported (cited item actually present). On confirmed bug reports, "
                f"{on_bugs['appropriate']}/{on_bugs['suggestions']} questions were judged "
                f"appropriate. On non-bug issues {on_other['appropriate']}/"
                f"{on_other['suggestions']}: the label error, not the rules, drives most "
                "inappropriate questions. Only 18 of 30 sampled issues received questions "
                "(plan target 20). The LLM-drafting grounding ablation was not run (deferred).",
                "",
            ]
            (run_dir / "suggestions.json").write_text(
                json.dumps(
                    {
                        "overall": overall,
                        "bug_reports": on_bugs,
                        "non_bug": on_other,
                        "markdown": "\n".join(md),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            done["suggestions"] = True
    return done
