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


class CalibratedTextClassifier:
    """Serving adapter: TF-IDF raw probabilities, V-cal sigmoid calibration and the V-policy
    threshold. Below the threshold (or with no qualifying threshold) the category is deferred."""

    def __init__(
        self,
        model: TextModel,
        calibrator,
        threshold: float | None,
        repository: str,
        explainer=None,
    ):
        self.model, self.calibrator = model, calibrator
        self.threshold, self.repository = threshold, repository
        self.explainer = explainer

    def predict(self, text: str) -> ClassificationResult:
        import numpy as np

        from knotify.selection import align

        features = self.model.transform([text])
        raw = align(self.model.pipeline.predict_proba(features), list(self.model.pipeline.classes_))
        cal = align(self.calibrator.predict_proba(features), list(self.calibrator.classes_))
        raw_d = {Category(label): float(p) for label, p in zip(LABELS, raw[0], strict=True)}
        cal_d = {Category(label): float(p) for label, p in zip(LABELS, cal[0], strict=True)}
        predicted = Category(LABELS[int(np.argmax(cal[0]))])
        top = float(cal[0].max())
        if self.threshold is None:
            deferred, reason = (
                True,
                (
                    "No confidence threshold reached 90% accepted accuracy on validation, so every "
                    "category is deferred to a maintainer."
                ),
            )
        elif top < self.threshold:
            deferred, reason = (
                True,
                (
                    f"Calibrated confidence {top:.2f} is below the review threshold "
                    f"{self.threshold:.2f}; a maintainer should decide."
                ),
            )
        else:
            deferred, reason = (
                False,
                (
                    f"Calibrated confidence {top:.2f} meets the threshold {self.threshold:.2f} "
                    "(chosen on validation data); still subject to human review."
                ),
            )
        explanation, note = (), None
        if self.explainer is not None:
            try:
                explanation, note = self.explainer.explain(features[0], predicted.value)
            except Exception:  # an explanation failure must not remove the prediction
                note = "Explanation unavailable for this input."
        return ClassificationResult(
            predicted_label=predicted,
            raw_probabilities=raw_d,
            calibrated_probabilities=cal_d,
            explanation=explanation,
            explanation_note=note,
            deferred=deferred,
            reason=reason,
            model_version=self.model.metadata.get("model_version"),
            calibration_repository=self.repository,
        )
