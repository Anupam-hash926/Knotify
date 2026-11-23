"""Day 3 freeze: record every configuration and evaluation sample that Day 4 results must use.

The manifest stores hashes (not data) so it can be committed; any later change to a frozen file
is detectable with ``verify_freeze``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knotify.config import PROJECT_ROOT, AppConfig


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_files(config: AppConfig, repo: str) -> dict[str, Path]:
    from knotify.experiment import embeddings_dir, latest_snapshot, latest_split, slug

    snapshot = latest_snapshot(config.paths.data, repo)
    split = latest_split(config.paths.data, repo)
    art = config.paths.artifacts / slug(repo) / split.name
    ann = config.paths.data / "annotations" / slug(repo) / split.name
    emb = embeddings_dir(config, repo)
    serving = json.loads((art / "serving.json").read_text(encoding="utf-8"))
    files = {
        "snapshot_issues": snapshot / "issues.jsonl",
        "split_ids": split / "split_ids.json",
        "label_mapping": PROJECT_ROOT / "configs" / "repositories.toml",
        "default_config": PROJECT_ROOT / "configs" / "default.toml",
        "serving_spec": art / "serving.json",
        "serving_model": config.paths.artifacts / serving["artifact"],
        "serving_calibrator": config.paths.artifacts / serving["calibrator"],
        "tfidf_raw_model": art / "tfidf_lr_raw.joblib",
        "embedding_lr_model": art / "embedding_lr.joblib",
        "embedding_lr_calibrator": art / "calibrator_embedding_lr.joblib",
        "embedding_vectors": emb / "vectors.npy",
        "embedding_meta": emb / "meta.json",
        "shap_background": art / "shap_background_mean.npy",
        "retrieval_queries": ann / "retrieval_queries.json",
        "evaluation_samples": ann / "evaluation_samples.json",
        "retrieval_pools": ann / "retrieval_pools.json",
    }
    return {k: v for k, v in files.items() if v.exists()}


def write_freeze(config: AppConfig, repo: str, out: Path) -> dict[str, Any]:
    from knotify.checklist import RULE_VERSION

    if out.exists():
        raise FileExistsError(f"Freeze manifest already exists: {out}")
    files = frozen_files(config, repo)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = None
    serving = json.loads(files["serving_spec"].read_text(encoding="utf-8"))
    embedding_meta = json.loads(files["embedding_meta"].read_text(encoding="utf-8"))
    manifest = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "repository": repo,
        "git_head_at_freeze": commit,
        "serving": {k: serving[k] for k in ("model", "family", "threshold", "split")},
        "checklist_rule_version": RULE_VERSION,
        "retrieval": config.retrieval.model_dump(),
        "classification": config.classification.model_dump(),
        "embedding": {
            k: embedding_meta.get(k)
            for k in ("model", "revision", "max_seq_length", "truncated_fraction", "dimension")
        },
        "llm_comparison": "deferred by the user (not completed)",
        "files": {
            name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha(path)}
            for name, path in files.items()
        },
        "rules": [
            "Test results are produced only by the Day 4 evaluation runner from these files.",
            "Rerun Test only for a documented bug fix, never for tuning.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_freeze(manifest_path: Path) -> list[str]:
    """Return the names of frozen files that changed or disappeared (empty = intact)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for name, entry in manifest["files"].items():
        path = PROJECT_ROOT / entry["path"]
        if not path.exists() or _sha(path) != entry["sha256"]:
            problems.append(name)
    return problems
