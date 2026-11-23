"""TF-IDF SHAP explanations (LinearExplainer, Train mean background).

What is explained: the *raw* logistic-regression score (logit) of the predicted class of the
TF-IDF model, relative to the Train-mean baseline. Not the calibrated probability, and not the
embedding classifier. Only words present in the issue are listed; the remaining contribution
(absent vocabulary) is reported as one number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from knotify.contracts import TokenContribution

NOTE = (
    "SHAP values for the TF-IDF model's raw score of the suggested category (Train-mean "
    "baseline). They explain this model, not the calibrated confidence or whether it is right."
)


def build_background(config, repo: str) -> Path:
    from knotify.classifier import TextModel
    from knotify.experiment import latest_split, load_split_rows, slug, text_of

    split = latest_split(config.paths.data, repo)
    art = config.paths.artifacts / slug(repo) / split.name
    model = TextModel.load(art / "tfidf_lr_masked.joblib")
    train = load_split_rows(config, split, ("train",))["train"]
    tfidf = model.pipeline.named_steps["tfidf"]
    matrix = tfidf.transform(model.transform([text_of(r) for r in train]))
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    path = art / "shap_background_mean.npy"
    np.save(path, mean)
    (art / "shap_background.json").write_text(
        json.dumps({"background": "Train TF-IDF mean", "train_count": len(train)}, indent=2),
        encoding="utf-8",
    )
    return path


class TfidfShapExplainer:
    def __init__(self, pipeline, background_mean: np.ndarray, top_k: int = 8):
        import shap

        self.tfidf = pipeline.named_steps["tfidf"]
        self.lr = pipeline.named_steps["lr"]
        self.classes = list(self.lr.classes_)
        self.vocab = self.tfidf.get_feature_names_out()
        self.top_k = top_k
        self.explainer = shap.LinearExplainer(
            (self.lr.coef_, self.lr.intercept_),
            shap.maskers.Independent(background_mean, max_samples=1),
        )

    def explain(self, prepared_text: str, label: str):
        x = self.tfidf.transform([prepared_text])
        values = np.asarray(self.explainer.shap_values(x))
        k = self.classes.index(label)
        row = values[0, :, k] if values.ndim == 3 else values[k][0]
        present = x.nonzero()[1]
        ordered = sorted(present, key=lambda j: -abs(row[j]))[: self.top_k]
        rest = float(row.sum() - row[present].sum())
        tokens = tuple(
            TokenContribution(token=str(self.vocab[j]), contribution=float(row[j])) for j in ordered
        )
        return tokens, f"{NOTE} Contribution of words not in this issue: {rest:+.2f}."
