import pytest

from knotify.config import AppConfig
from knotify.contracts import AnalysisReport, RetrievedIssue, Review
from knotify.storage import save_report, save_review
from knotify.triage import ModelBundle, analyze


def test_unimplemented_components_never_fabricate_predictions(issue_factory):
    report = analyze(issue_factory(), (), ModelBundle(), AppConfig())
    assert report.classification.predicted_label is None
    assert report.classification.calibrated_probabilities is None
    assert report.classification.deferred
    assert not report.related_issues and not report.checklist and not report.suggestions
    assert report.components["classification"].state == "unavailable"
    assert report.components["llm"].state == "disabled"
    assert AnalysisReport.model_validate_json(report.model_dump_json()) == report


def test_classifier_failure_does_not_block_retrieval(issue_factory):
    class BrokenClassifier:
        def predict(self, text):
            raise RuntimeError("SECRET should not be returned to UI")

    class WorkingRetriever:
        def search(self, query, eligible_corpus, *, top_k):
            assert len(eligible_corpus) == 1
            return (RetrievedIssue(issue=eligible_corpus[0], rank=1),)

    report = analyze(
        issue_factory(),
        [issue_factory(1, 1), issue_factory(3, 3)],
        ModelBundle(classifier=BrokenClassifier(), retriever=WorkingRetriever()),
        AppConfig(),
    )
    assert report.components["classification"].state == "failed"
    assert report.components["retrieval"].state == "ready"
    assert report.related_issues[0].issue.issue_id == "1"
    assert "SECRET" not in report.model_dump_json()


def test_service_rejects_invented_or_future_retrieval_results(issue_factory):
    class BadRetriever:
        def search(self, query, eligible_corpus, *, top_k):
            return (RetrievedIssue(issue=issue_factory(3, 3), rank=1),)

    report = analyze(issue_factory(), (), ModelBundle(retriever=BadRetriever()), AppConfig())
    assert report.components["retrieval"].state == "failed"
    assert not report.related_issues


def test_storage_is_append_only_and_review_requires_saved_report(issue_factory, tmp_path):
    report = analyze(issue_factory(), (), ModelBundle(), AppConfig())
    review = Review(analysis_id=report.analysis_id, decision="rejected", note="No models yet")
    with pytest.raises(FileNotFoundError):
        save_review(review, tmp_path)
    path = save_report(report, tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_report(report, tmp_path)
    assert save_review(review, tmp_path).exists()
    assert path.read_bytes() == original
