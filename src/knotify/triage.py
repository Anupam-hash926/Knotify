"""Shared orchestration shell. All ML adapters are explicitly unavailable by default."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter

from knotify.checklist import Checklist, UnavailableChecklist
from knotify.classifier import Classifier, UnavailableClassifier
from knotify.config import AppConfig
from knotify.contracts import AnalysisReport, ClassificationResult, ComponentStatus, Issue
from knotify.data import input_hash, model_text, stable_hash
from knotify.errors import ComponentUnavailable
from knotify.retrieval import Retriever, UnavailableRetriever, eligible_history

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    """Dependency injection seam; importing this class loads no ML libraries."""

    classifier: Classifier = field(default_factory=UnavailableClassifier)
    retriever: Retriever = field(default_factory=UnavailableRetriever)
    checklist: Checklist = field(default_factory=UnavailableChecklist)


def _run(name, action, fallback, statuses):
    try:
        result = action()
        statuses[name] = ComponentStatus(state="ready", detail="Component completed.")
        return result
    except ComponentUnavailable as exc:
        statuses[name] = ComponentStatus(state="unavailable", detail=str(exc))
    except Exception as exc:
        # No raw input, credentials or exception message in UI/logs.
        logger.warning("%s failed (%s)", name, type(exc).__name__)
        statuses[name] = ComponentStatus(state="failed", detail="Component failed; see local logs.")
    return fallback


def analyze(
    issue: Issue,
    repository_history: Sequence[Issue],
    model_bundle: ModelBundle,
    config: AppConfig,
    *,
    dataset_version: str | None = None,
) -> AnalysisReport:
    started = perf_counter()
    statuses: dict[str, ComponentStatus] = {}
    classification = _run(
        "classification",
        lambda: model_bundle.classifier.predict(model_text(issue)),
        ClassificationResult(),
        statuses,
    )

    def retrieve():
        eligible = eligible_history(issue, repository_history)
        results = model_bundle.retriever.search(
            issue, eligible, top_k=config.retrieval.display_count
        )
        allowed = {item.issue_id: item for item in eligible}
        if len(results) > config.retrieval.display_count:
            raise ValueError("Retriever exceeded the requested result count.")
        if len({result.issue.issue_id for result in results}) != len(results):
            raise ValueError("Retriever returned duplicate issue IDs.")
        for rank, result in enumerate(results, 1):
            if allowed.get(result.issue.issue_id) != result.issue or result.rank != rank:
                raise ValueError("Retriever returned an ineligible snapshot or invalid rank.")
        return results

    related = _run("retrieval", retrieve, (), statuses)
    checklist = _run(
        "checklist", lambda: model_bundle.checklist.check(issue, classification), (), statuses
    )
    statuses["drafting"] = ComponentStatus(
        state="unavailable", detail="Question templates and reference validation are not built yet."
    )
    statuses["llm"] = ComponentStatus(
        state="unavailable" if config.llm.enabled else "disabled",
        detail="No LLM adapter is connected; no API request was made.",
    )
    return AnalysisReport(
        issue=issue,
        dataset_version=dataset_version,
        input_hash=input_hash(issue),
        config_hash=stable_hash(config.model_dump(mode="json")),
        classification=classification,
        related_issues=related,
        checklist=checklist,
        components=statuses,
        elapsed_ms=(perf_counter() - started) * 1000,
    )
