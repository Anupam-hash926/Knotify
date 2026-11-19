"""Typed project configuration without credentials or working-directory-dependent data paths."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(ConfigModel):
    data: Path = Path("data")
    artifacts: Path = Path("artifacts")
    runs: Path = Path("runs")


class ClassificationConfig(ConfigModel):
    labels: tuple[str, ...] = ("bug", "feature", "question_documentation", "other")
    c: float = Field(default=1.0, gt=0)
    class_weight: Literal["balanced"] = "balanced"
    max_iter: int = Field(default=1000, gt=0)
    train_fraction: float = Field(default=0.6, gt=0, lt=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    policy_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    minimum_accepted: int = Field(default=20, gt=0)
    target_accepted_accuracy: float = Field(default=0.9, gt=0, le=1)

    @model_validator(mode="after")
    def check_policy(self):
        expected = ("bug", "feature", "question_documentation", "other")
        if self.labels != expected:
            raise ValueError("The submitted four-class taxonomy and class order must be explicit.")
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-9:
            raise ValueError("Split fractions must sum to 1.")
        if not self.policy_thresholds or any(not 0 <= t <= 1 for t in self.policy_thresholds):
            raise ValueError("Policy thresholds must be nonempty probabilities.")
        return self


class EmbeddingConfig(ConfigModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Literal["cpu"] = "cpu"
    batch_size: int = Field(default=16, gt=0)


class RetrievalConfig(ConfigModel):
    candidate_depth: int = Field(default=50, gt=0)
    display_count: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def check_depth(self):
        if self.display_count > self.candidate_depth:
            raise ValueError("Display count cannot exceed candidate depth.")
        return self


class LLMConfig(ConfigModel):
    enabled: bool = False
    timeout_seconds: int = Field(default=30, gt=0)
    max_retries: int = Field(default=1, ge=0, le=3)


class AppConfig(ConfigModel):
    schema_version: Literal[1] = 1
    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(path: Path | None = None, *, root: Path | None = None) -> AppConfig:
    """Load TOML; resolve runtime paths under the editable checkout or explicit root."""
    root = (root or PROJECT_ROOT).resolve()
    selected = path or Path(os.getenv("KNOTIFY_CONFIG", "configs/default.toml"))
    selected = selected if selected.is_absolute() else root / selected
    with selected.open("rb") as stream:
        config = AppConfig.model_validate(tomllib.load(stream))
    paths = {
        name: value.resolve() if value.is_absolute() else (root / value).resolve()
        for name, value in config.paths.model_dump().items()
    }
    return config.model_copy(update={"paths": Paths(**paths)})
