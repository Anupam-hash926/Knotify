"""Classifier boundary. No model is trained, downloaded or loaded at import time."""

from typing import Protocol

from knotify.contracts import ClassificationResult
from knotify.errors import ComponentUnavailable


class Classifier(Protocol):
    def predict(self, text: str) -> ClassificationResult: ...


class UnavailableClassifier:
    def predict(self, text: str) -> ClassificationResult:
        raise ComponentUnavailable("No trained classifier is connected yet.")
