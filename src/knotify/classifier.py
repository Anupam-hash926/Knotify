"""TF-IDF logistic-regression classifier, majority baseline and saved-artifact adapter.

No model is trained, downloaded or loaded at import time. Calibration (Day 3) is not fitted
yet, so served predictions are always deferred and carry raw probabilities only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from knotify.contracts import Category, ClassificationResult
from knotify.data import LABELS
from knotify.errors import ComponentUnavailable
from knotify.features import TOKEN_PATTERN, prepare


class Classifier(Protocol):
    def predict(self, text: str) -> ClassificationResult: ...


class UnavailableClassifier:
    def predict(self, text: str) -> ClassificationResult:
        raise ComponentUnavailable("No trained classifier is connected yet.")


def build_tfidf_lr(*, c: float = 1.0, max_iter: int = 1000, seed: int = 42):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=TOKEN_PATTERN,
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    max_features=200_000,
                    dtype=np.float64,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    C=c, class_weight="balanced", max_iter=max_iter, random_state=seed
                ),
            ),
        ]
    )


@dataclass
class MajorityBaseline:
    label: str = ""

    def fit(self, texts: Sequence[str], targets: Sequence[str]) -> MajorityBaseline:
        self.label = Counter(targets).most_common(1)[0][0]
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [self.label] * len(texts)


@dataclass
class TextModel:
    """A fitted text pipeline plus the exact input transform used at training time."""

    pipeline: Any
    mask: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform(self, texts: Sequence[str]) -> list[str]:
        return [prepare(t, mask=self.mask) for t in texts]

    def predict_proba(self, texts: Sequence[str]):
        """Probabilities in the fixed ``LABELS`` order (absent training classes get 0)."""
        import numpy as np

        raw = self.pipeline.predict_proba(self.transform(texts))
        classes = list(self.pipeline.classes_)
        out = np.zeros((len(texts), len(LABELS)))
        for j, label in enumerate(LABELS):
            if label in classes:
                out[:, j] = raw[:, classes.index(label)]
        return out

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "mask": self.mask, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: Path) -> TextModel:
        import joblib

        # Only load artefacts produced locally by `knotify train` (joblib uses pickle).
        payload = joblib.load(path)
        return cls(payload["pipeline"], payload["mask"], payload["metadata"])


class TextModelClassifier:
    """Adapter for ``analyze()``: raw probabilities, always deferred until calibrated."""

    def __init__(self, model: TextModel):
        self.model = model

    def predict(self, text: str) -> ClassificationResult:
        probabilities = self.model.predict_proba([text])[0]
        raw = {Category(label): float(p) for label, p in zip(LABELS, probabilities, strict=True)}
        total = sum(raw.values())
        raw = {k: v / total for k, v in raw.items()}
        predicted = max(raw, key=raw.get)
        return ClassificationResult(
            predicted_label=predicted,
            raw_probabilities=raw,
            calibrated_probabilities=None,
            deferred=True,
            reason="Uncalibrated baseline: calibration and the deferral threshold are not fitted "
            "yet, so this category is a suggestion for human review.",
            model_version=self.model.metadata.get("model_version"),
        )
