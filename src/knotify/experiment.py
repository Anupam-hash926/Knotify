"""Offline dataset preparation (frozen splits) and baseline training/validation.

The Test split is written to the manifest but never read by any function in this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knotify.config import AppConfig
from knotify.data import (
    LABELS,
    build_supervised,
    chronological_split,
    class_counts,
    load_mappings,
    map_labels,
    split_validation,
)
from knotify.ingest import load_snapshot


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def latest_snapshot(data_dir: Path, repo: str) -> Path:
    root = data_dir / "raw" / slug(repo)
    complete = []
    for path in sorted(root.glob("*")):
        manifest = path / "manifest.json"
        if path.is_dir() and not path.name.endswith(".partial") and manifest.exists():
            if not json.loads(manifest.read_text(encoding="utf-8")).get("partial", True):
                complete.append(path)
    if not complete:
        raise FileNotFoundError(f"No complete frozen snapshot for {repo} under {root}")
    return complete[-1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _span(rows: list[dict]) -> list[str] | None:
    return [rows[0]["created_at"], rows[-1]["created_at"]] if rows else None


def prepare_splits(config: AppConfig, mapping_path: Path, repo: str) -> Path:
    snapshot = latest_snapshot(config.paths.data, repo)
    manifest, records, _ = load_snapshot(snapshot)
    mapping = load_mappings(mapping_path)[repo]
    status_counts: dict[str, int] = {}
    for record in records:
        if mapping.window_start and record["created_at"] < mapping.window_start:
            continue
        status = map_labels(record["labels"], mapping)[1]
        status_counts[status] = status_counts.get(status, 0) + 1
    supervised = build_supervised(records, mapping)
    cls = config.classification
    parts = chronological_split(
        supervised.rows, (cls.train_fraction, cls.validation_fraction, cls.test_fraction)
    )
    validation = split_validation(parts["validation"])
    out = config.paths.data / "splits" / slug(repo) / snapshot.name
    if out.exists():
        raise FileExistsError(f"Frozen split already exists: {out}")
    out.mkdir(parents=True)
    ids = {name: [r["number"] for r in rows] for name, rows in parts.items()}
    ids["v_cal"] = [r["number"] for r in validation["v_cal"]]
    ids["v_policy"] = [r["number"] for r in validation["v_policy"]]
    (out / "split_ids.json").write_text(json.dumps(ids) + "\n", encoding="utf-8")
    split_manifest = {
        "repository": repo,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "snapshot": str(snapshot.relative_to(config.paths.data)),
        "snapshot_issues_sha256": manifest["issues_sha256"],
        "mapping_file": mapping_path.name,
        "mapping_sha256": _sha(mapping_path),
        "label_mapping": {k: list(v) for k, v in mapping.labels.items()},
        "exclude_labels": list(mapping.exclude_labels),
        "window_start": mapping.window_start,
        "snapshot_issue_count": len(records),
        "window_label_status": status_counts,
        "exclusions": supervised.exclusions,
        "supervised_count": len(supervised.rows),
        "fractions": [cls.train_fraction, cls.validation_fraction, cls.test_fraction],
        "splits": {
            name: {"count": len(rows), "classes": class_counts(rows), "created": _span(rows)}
            for name, rows in parts.items()
        },
        "validation_mode": validation["mode"],
        "validation_reason": validation.get("reason"),
        "v_cal": {"count": len(ids["v_cal"]), "classes": class_counts(validation["v_cal"])},
        "v_policy": {
            "count": len(ids["v_policy"]),
            "classes": class_counts(validation["v_policy"]),
        },
        "split_ids_sha256": _sha(out / "split_ids.json"),
    }
    (out / "manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", "utf-8")
    return out


def latest_split(data_dir: Path, repo: str) -> Path:
    candidates = sorted(p for p in (data_dir / "splits" / slug(repo)).glob("*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No frozen split for {repo}; run `knotify split` first.")
    return candidates[-1]


def load_split_rows(config: AppConfig, split_dir: Path, names: tuple[str, ...]) -> dict:
    """Rows for the requested split names only (callers in this module never ask for test)."""
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot = config.paths.data / manifest["snapshot"]
    _, records, _ = load_snapshot(snapshot)
    by_number = {r["number"]: r for r in records}
    ids = json.loads((split_dir / "split_ids.json").read_text(encoding="utf-8"))
    mapping = manifest["label_mapping"]
    lookup = {name: target for target, names in mapping.items() for name in names}
    out = {}
    for name in names:
        rows = []
        for number in ids[name]:
            record = by_number[number]
            targets = {lookup[x] for x in record["labels"] if x in lookup}
            rows.append({**record, "target": targets.pop()})
        out[name] = rows
    return out


def text_of(record: dict[str, Any]) -> str:
    return f"{record['title'].strip()}\n\n{record['body'].strip()}".strip()


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    per_class = f1_score(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)
    return {
        "n": len(y_true),
        "support": {label: y_true.count(label) for label in LABELS},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(per_class.mean()),
        "per_class_f1": {label: float(v) for label, v in zip(LABELS, per_class, strict=True)},
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
        },
        "macro_f1_note": "Macro average over the fixed four labels; absent classes score 0.",
    }


def train_baseline(config: AppConfig, repo: str) -> dict[str, Any]:
    from knotify.classifier import MajorityBaseline, TextModel, build_tfidf_lr

    split_dir = latest_split(config.paths.data, repo)
    rows = load_split_rows(config, split_dir, ("train", "v_policy"))
    train, policy = rows["train"], rows["v_policy"]
    x_train, y_train = [text_of(r) for r in train], [r["target"] for r in train]
    x_eval, y_eval = [text_of(r) for r in policy], [r["target"] for r in policy]
    cls = config.classification
    artifact_dir = config.paths.artifacts / slug(repo) / split_dir.name
    run_dir = config.paths.runs / "baseline" / slug(repo) / split_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "repository": repo,
        "split": str(split_dir.relative_to(config.paths.data)),
        "evaluated_on": "v_policy (validation); Test untouched",
        "train_count": len(train),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": {},
    }
    majority = MajorityBaseline().fit(x_train, y_train)
    results["models"]["majority"] = {
        "label": majority.label,
        **classification_metrics(y_eval, majority.predict(x_eval)),
    }
    predictions: dict[str, list[str]] = {}
    for mask in (True, False):
        name = "tfidf_lr_masked" if mask else "tfidf_lr_raw"
        pipeline = build_tfidf_lr(c=cls.c, max_iter=cls.max_iter, seed=config.seed)
        model = TextModel(pipeline, mask=mask)
        pipeline.fit(model.transform(x_train), y_train)
        import numpy as np

        proba = model.predict_proba(x_eval)
        pred = [LABELS[i] for i in np.argmax(proba, axis=1)]
        predictions[name] = pred
        version = f"{name}:{split_dir.name}"
        model.metadata = {
            "model_version": version,
            "repository": repo,
            "split": results["split"],
            "mask_template_cues": mask,
            "c": cls.c,
            "class_weight": cls.class_weight,
            "seed": config.seed,
            "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
            "calibrated": False,
        }
        path = artifact_dir / f"{name}.joblib"
        model.save(path)
        results["models"][name] = {
            "artifact": str(path.relative_to(config.paths.artifacts)),
            "vocabulary_size": model.metadata["vocabulary_size"],
            **classification_metrics(y_eval, pred),
        }
    with (run_dir / "v_policy_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "created_at", "target", *predictions])
        for i, record in enumerate(policy):
            writer.writerow(
                [record["number"], record["created_at"], record["target"]]
                + [predictions[k][i] for k in predictions]
            )
    (run_dir / "baseline_validation.json").write_text(json.dumps(results, indent=2) + "\n", "utf-8")
    return results


def label_audit_sheet(config: AppConfig, repo: str, per_class: int = 10) -> Path:
    """Stratified (class x time) sample from Train for the ~40-example human label audit.

    Judgement columns are left blank: they must be filled by a person, never generated.
    """
    import random

    from knotify.data import clean_markdown

    split_dir = latest_split(config.paths.data, repo)
    train = load_split_rows(config, split_dir, ("train",))["train"]
    rng = random.Random(config.seed)
    chosen = []
    for label in LABELS:
        rows = [r for r in train if r["target"] == label]
        if len(rows) <= per_class:
            chosen += rows
            continue
        step = len(rows) / per_class  # one draw per chronological stratum
        chosen += [rows[int(i * step) + rng.randrange(max(1, int(step)))] for i in range(per_class)]
    out = config.paths.data / "annotations" / slug(repo) / split_dir.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "label_audit_sheet.csv"
    if path.exists():
        raise FileExistsError(f"Review sheet already exists (may contain judgements): {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "number",
                "url",
                "created_at",
                "source_labels",
                "mapped_target",
                "title",
                "body_excerpt",
                "human_agrees_with_mapping (yes/no/unclear)",
                "human_target_if_no",
                "reviewer_initials",
                "notes",
            ]
        )
        for r in chosen:
            excerpt = " ".join(clean_markdown(r["body"]).split())[:500]
            writer.writerow(
                [
                    r["number"],
                    r["html_url"],
                    r["created_at"],
                    "; ".join(r["labels"]),
                    r["target"],
                    r["title"],
                    excerpt,
                    "",
                    "",
                    "",
                    "",
                ]
            )
    return path


def embeddings_dir(config: AppConfig, repo: str) -> Path:
    snapshot = latest_snapshot(config.paths.data, repo)
    return config.paths.artifacts / slug(repo) / "embeddings" / snapshot.name


def build_embeddings(config: AppConfig, repo: str) -> Path:
    """Encode every issue in the latest frozen snapshot once (CPU) and cache the vectors."""
    from knotify.embeddings import build_cache, load_encoder, model_revision
    from knotify.ingest import to_issue

    snapshot = latest_snapshot(config.paths.data, repo)
    manifest, records, _ = load_snapshot(snapshot)
    out = embeddings_dir(config, repo)
    if (out / "vectors.npy").exists():
        raise FileExistsError(f"Embedding cache already exists: {out}")
    encoder = load_encoder(config.embeddings.local_path, device=config.embeddings.device)
    cache = build_cache(
        encoder,
        [to_issue(r, repo) for r in records],
        model_name=config.embeddings.model,
        revision=model_revision(config.embeddings.local_path),
        dataset_version=f"{snapshot.name}:{manifest['issues_sha256'][:12]}",
        batch_size=config.embeddings.batch_size,
    )
    cache.save(out)
    return out


def train_embedding_lr(config: AppConfig, repo: str) -> dict[str, Any]:
    """Frozen-embedding logistic regression with the same settings as TF-IDF; V-policy only."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from knotify.embeddings import EmbeddingCache

    split_dir = latest_split(config.paths.data, repo)
    rows = load_split_rows(config, split_dir, ("train", "v_policy"))
    cache = EmbeddingCache.load(embeddings_dir(config, repo))

    def matrix(records):
        ids = [f"{repo}#{r['number']}" for r in records]
        return np.stack([cache.vectors[cache.index[i]] for i in ids])

    cls = config.classification
    model = LogisticRegression(
        C=cls.c, class_weight="balanced", max_iter=cls.max_iter, random_state=config.seed
    )
    model.fit(matrix(rows["train"]), [r["target"] for r in rows["train"]])
    proba = model.predict_proba(matrix(rows["v_policy"]))
    classes = list(model.classes_)
    pred = [classes[i] for i in np.argmax(proba, axis=1)]
    y_eval = [r["target"] for r in rows["v_policy"]]
    import joblib

    artifact = config.paths.artifacts / slug(repo) / split_dir.name / "embedding_lr.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "metadata": {
                "model_version": f"embedding_lr:{split_dir.name}",
                "encoder": cache.meta.get("model"),
                "encoder_revision": cache.meta.get("revision"),
                "text_transform": cache.meta.get("text_transform"),
                "c": cls.c,
                "class_weight": cls.class_weight,
                "seed": config.seed,
                "calibrated": False,
            },
        },
        artifact,
    )
    run_dir = config.paths.runs / "baseline" / slug(repo) / split_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "repository": repo,
        "evaluated_on": "v_policy (validation); Test untouched",
        "encoder": cache.meta.get("model"),
        "encoder_revision": cache.meta.get("revision"),
        "truncated_fraction": cache.meta.get("truncated_fraction"),
        "artifact": str(artifact.relative_to(config.paths.artifacts)),
        **classification_metrics(y_eval, pred),
    }
    path = run_dir / "baseline_validation.json"
    combined = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"models": {}}
    combined["models"]["embedding_lr"] = result
    path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "v_policy_predictions_embedding_lr.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "target", "embedding_lr"])
        for record, p in zip(rows["v_policy"], pred, strict=True):
            writer.writerow([record["number"], record["target"], p])
    return result
