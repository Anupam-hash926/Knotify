"""Versioned shared data contracts. No model imports and no implicit predictions."""

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Category(StrEnum):
    BUG = "bug"
    FEATURE = "feature"
    QUESTION_DOCUMENTATION = "question_documentation"
    OTHER = "other"


class Issue(Record):
    issue_id: str = Field(min_length=1)
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    number: int | None = Field(default=None, gt=0)
    title: str = Field(min_length=1)
    body: str = ""
    created_at: AwareDatetime
    updated_at: AwareDatetime | None = None
    fetched_at: AwareDatetime | None = None
    url: HttpUrl | None = None
    source_labels: tuple[str, ...] = ()
    mapped_target: Category | None = None
    origin: Literal["github", "pasted"] = "github"

    @model_validator(mode="after")
    def source_identity(self):
        if not self.title.strip():
            raise ValueError("An issue title cannot contain only whitespace.")
        if self.origin == "github":
            if self.number is None or self.url is None:
                raise ValueError("Imported GitHub issues need a number and source URL.")
            expected = f"/{self.repository}/issues/{self.number}"
            if self.url.host != "github.com" or self.url.scheme != "https":
                raise ValueError("Imported source links must use https://github.com.")
            if self.url.path != expected:
                raise ValueError("Source URL must identify this repository and issue number.")
        return self


class ClassificationResult(Record):
    predicted_label: Category | None = None
    raw_probabilities: dict[Category, Probability] | None = None
    calibrated_probabilities: dict[Category, Probability] | None = None
    deferred: bool = True
    reason: str = "Classifier is not implemented."
    model_version: str | None = None
    calibration_repository: str | None = None

    @model_validator(mode="after")
    def probabilities_are_complete(self):
        for probabilities in (self.raw_probabilities, self.calibrated_probabilities):
            if probabilities is not None:
                if set(probabilities) != set(Category):
                    raise ValueError("Probability distributions must include all four classes.")
                if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6):
                    raise ValueError("Class probabilities must sum to 1.")
                if self.predicted_label is None:
                    raise ValueError("Probabilities require a predicted category.")
        if not self.deferred and (
            self.predicted_label is None or self.calibrated_probabilities is None
        ):
            raise ValueError("An accepted category requires calibrated probabilities.")
        return self


class RetrievedIssue(Record):
    issue: Issue
    rank: int = Field(gt=0)
    method_scores: dict[str, float] = Field(default_factory=dict)
    relationship: Literal["related_candidate"] = "related_candidate"


class ChecklistItemId(StrEnum):
    REPRODUCTION = "reproduction_steps"
    EXPECTED = "expected_behavior"
    ACTUAL = "actual_behavior"
    ENVIRONMENT = "environment"
    VERSION = "version"
    LOGS = "error_logs"


class ChecklistItem(Record):
    item_id: ChecklistItemId
    status: Literal["present", "missing", "uncertain", "not_applicable"]
    reason: str
    matched_text: str | None = None
    rule_version: str | None = None


class Suggestion(Record):
    text: str
    evidence_ids: tuple[str, ...]
    reference_valid: bool = False
    semantic_support: Literal["not_checked", "human_supported", "human_unsupported"] = "not_checked"
    source: Literal["template", "llm"] = "template"


class ComponentStatus(Record):
    state: Literal["ready", "unavailable", "failed", "disabled"]
    detail: str


class AnalysisReport(Record):
    schema_version: Literal[1] = 1
    analysis_id: UUID = Field(default_factory=uuid4)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    issue: Issue
    dataset_version: str | None = None
    input_hash: str
    config_hash: str
    classification: ClassificationResult = Field(default_factory=ClassificationResult)
    related_issues: tuple[RetrievedIssue, ...] = ()
    checklist: tuple[ChecklistItem, ...] = ()
    suggestions: tuple[Suggestion, ...] = ()
    components: dict[str, ComponentStatus]
    elapsed_ms: float = Field(ge=0)


class Review(Record):
    review_id: UUID = Field(default_factory=uuid4)
    analysis_id: UUID
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    decision: Literal["accepted", "rejected", "edited"]
    corrected_category: Category | None = None
    note: str = ""

    @model_validator(mode="after")
    def edit_has_content(self):
        if self.decision == "edited" and self.corrected_category is None and not self.note.strip():
            raise ValueError("An edited review needs a correction or note.")
        return self
