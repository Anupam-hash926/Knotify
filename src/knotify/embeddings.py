"""Sentence-embedding cache and CPU encoder (frozen ``all-MiniLM-L6-v2``).

One L2-normalised vector per issue, computed from ``features.prepare(model_text(issue))`` (the
same template-masked text used by TF-IDF). Truncation at the encoder's maximum sequence length
is measured and recorded. Nothing is downloaded unless the local model folder is missing *and*
the caller explicitly allows it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from knotify.contracts import Issue
from knotify.data import input_hash, model_text
from knotify.errors import ComponentUnavailable
from knotify.features import prepare

TEXT_TRANSFORM = "features.prepare(data.model_text(issue), mask=True)"


def issue_text(issue: Issue) -> str:
    return prepare(model_text(issue))


def model_revision(local_path: Path) -> str | None:
    """Commit hash recorded by huggingface_hub when the model was downloaded, if present."""
    meta = local_path / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    if meta.exists():
        lines = meta.read_text(encoding="utf-8").splitlines()
        return lines[0].strip() if lines else None
    return None


def load_encoder(local_path: Path, *, device: str = "cpu"):
    if not local_path.exists():
        raise ComponentUnavailable(
            f"Embedding model folder not found ({local_path.name}); see IMPLEMENTATION_STATUS.md."
        )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ComponentUnavailable(
            "sentence-transformers is not installed (embeddings extra)."
        ) from exc
    return SentenceTransformer(str(local_path), device=device)


def encode(encoder, texts: list[str], *, batch_size: int = 16) -> np.ndarray:
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def truncation_stats(encoder, texts: list[str]) -> dict[str, Any]:
    limit = int(encoder.max_seq_length)
    lengths = [len(encoder.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    truncated = sum(n > limit for n in lengths)
    return {
        "max_seq_length": limit,
        "truncated": truncated,
        "truncated_fraction": truncated / len(texts) if texts else 0.0,
        "median_tokens": float(np.median(lengths)) if lengths else 0.0,
    }


@dataclass
class EmbeddingCache:
    vectors: np.ndarray
    ids: list[str]
    hashes: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.vectors.shape[0] != len(self.ids) or len(self.ids) != len(self.hashes):
            raise ValueError("Embedding cache rows, IDs and hashes must align.")
        self.index = {issue_id: row for row, issue_id in enumerate(self.ids)}

    def vector_for(self, issue: Issue) -> np.ndarray | None:
        """Cached vector only if the stored input hash matches this issue snapshot."""
        row = self.index.get(issue.issue_id)
        if row is None or self.hashes[row] != input_hash(issue):
            return None
        return self.vectors[row]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "ids.json").write_text(
            json.dumps({"ids": self.ids, "input_hashes": self.hashes}), encoding="utf-8"
        )
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n", "utf-8")

    @classmethod
    def load(cls, directory: Path) -> EmbeddingCache:
        if not (directory / "vectors.npy").exists():
            raise ComponentUnavailable(f"No embedding cache at {directory.name}.")
        ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(np.load(directory / "vectors.npy"), ids["ids"], ids["input_hashes"], meta)


def build_cache(
    encoder,
    issues: list[Issue],
    *,
    model_name: str,
    revision: str | None,
    dataset_version: str,
    batch_size: int = 16,
) -> EmbeddingCache:
    texts = [issue_text(i) for i in issues]
    vectors = encode(encoder, texts, batch_size=batch_size)
    meta = {
        "model": model_name,
        "revision": revision,
        "dimension": int(vectors.shape[1]),
        "count": len(issues),
        "normalized": True,
        "text_transform": TEXT_TRANSFORM,
        "dataset_version": dataset_version,
        "device": "cpu",
        "batch_size": batch_size,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **truncation_stats(encoder, texts),
    }
    return EmbeddingCache(
        vectors, [i.issue_id for i in issues], [input_hash(i) for i in issues], meta
    )
