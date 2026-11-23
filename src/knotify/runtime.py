"""Load the local serving bundle: frozen history, saved classifier and BM25 retriever.

Nothing is trained or downloaded here. Missing artefacts yield explicit unavailable states.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from knotify.config import PROJECT_ROOT, AppConfig
from knotify.contracts import Issue
from knotify.triage import ModelBundle

SERVING_MODEL = "tfidf_lr_masked.joblib"


@dataclass(frozen=True)
class Runtime:
    bundle: ModelBundle
    repository: str | None = None
    history: tuple[Issue, ...] = ()
    dataset_version: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def issue(self, number: int) -> Issue | None:
        return next((i for i in self.history if i.number == number), None)


def source_repository(mapping_path: Path | None = None) -> str | None:
    path = mapping_path or PROJECT_ROOT / "configs" / "repositories.toml"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        repos = tomllib.load(stream).get("repositories", [])
    return next((r["name"] for r in repos if r.get("role") == "source"), None)


def load_runtime(config: AppConfig, repository: str | None = None) -> Runtime:
    from knotify.experiment import latest_snapshot

    repository = repository or source_repository()
    if repository is None:
        return Runtime(ModelBundle(), notes=("No source repository is configured.",))
    notes: list[str] = []
    history: tuple[Issue, ...] = ()
    version = None
    retriever = None
    classifier = None
    try:
        from knotify.ingest import load_snapshot, to_issue

        snapshot = latest_snapshot(config.paths.data, repository)
        manifest, records, _ = load_snapshot(snapshot)
        history = tuple(to_issue(r, repository) for r in records)
        version = f"{snapshot.name}:{manifest['issues_sha256'][:12]}"
        from knotify.retrieval import BM25Retriever, HybridRetriever

        depth = config.retrieval.candidate_depth
        retriever = BM25Retriever(depth=depth)
        dense = _dense(config, repository, depth, notes)
        if dense is not None:
            retriever = HybridRetriever(retriever, dense, rrf_k=config.retrieval.rrf_k)
    except (FileNotFoundError, ImportError) as exc:
        notes.append(f"Retrieval history unavailable: {exc}")
    try:
        classifier = _classifier(config, repository, notes)
    except (FileNotFoundError, ImportError) as exc:
        notes.append(f"Classifier unavailable: {exc}")
    from knotify.checklist import RuleChecklist

    kwargs = {"checklist": RuleChecklist()}
    if classifier is not None:
        kwargs["classifier"] = classifier
    if retriever is not None:
        kwargs["retriever"] = retriever
    return Runtime(ModelBundle(**kwargs), repository, history, version, tuple(notes))


def _dense(config: AppConfig, repository: str, depth: int, notes: list[str]):
    """Dense retriever from the cached embeddings; the encoder is optional (pasted text)."""
    from knotify.embeddings import EmbeddingCache, load_encoder
    from knotify.errors import ComponentUnavailable
    from knotify.experiment import embeddings_dir
    from knotify.retrieval import DenseRetriever

    try:
        cache = EmbeddingCache.load(embeddings_dir(config, repository))
    except (ComponentUnavailable, FileNotFoundError) as exc:
        notes.append(f"Dense search unavailable: {exc}")
        return None
    encoder = None
    try:
        encoder = load_encoder(config.embeddings.local_path, device=config.embeddings.device)
    except ComponentUnavailable as exc:
        notes.append(f"Pasted reports use BM25 only: {exc}")
    dense = DenseRetriever(cache, encoder=encoder, depth=depth)
    notes.append(f"Dense search backend: {dense.backend}.")
    return dense


def _classifier(config: AppConfig, repository: str, notes: list[str]):
    """Frozen serving choice from ``serving.json`` (calibrated); else the uncalibrated baseline."""
    import json

    from knotify.classifier import CalibratedTextClassifier, TextModel, TextModelClassifier
    from knotify.experiment import latest_split, slug

    split = latest_split(config.paths.data, repository)
    art = config.paths.artifacts / slug(repository) / split.name
    serving = art / "serving.json"
    if serving.exists():
        spec = json.loads(serving.read_text(encoding="utf-8"))
        if spec["family"] != "tfidf":
            raise FileNotFoundError("Only the TF-IDF serving family is implemented in the app.")
        import joblib

        model = TextModel.load(config.paths.artifacts / spec["artifact"])
        calibrator = joblib.load(config.paths.artifacts / spec["calibrator"])["calibrator"]
        explainer = None
        background = art / "shap_background_mean.npy"
        if background.exists():
            try:
                import numpy as np

                from knotify.explain import TfidfShapExplainer

                explainer = TfidfShapExplainer(model.pipeline, np.load(background))
            except ImportError:
                notes.append("Explanations unavailable: install the `explain` extra (SHAP).")
        else:
            notes.append("Explanations unavailable: run `knotify explain-setup`.")
        return CalibratedTextClassifier(
            model, calibrator, spec["threshold"], repository, explainer=explainer
        )
    path = art / SERVING_MODEL
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path.name}; run `knotify train`.")
    notes.append("Classifier is uncalibrated; run `knotify calibrate`.")
    return TextModelClassifier(TextModel.load(path))
