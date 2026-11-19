import pytest
from pydantic import ValidationError

from knotify.config import AppConfig, ClassificationConfig, load_config
from knotify.contracts import Category, ClassificationResult, Issue
from knotify.data import input_hash, model_text
from knotify.retrieval import eligible_history


def test_config_defaults_are_cpu_and_llm_off():
    config = load_config()
    assert config.embeddings.device == "cpu"
    assert not config.llm.enabled
    assert config.paths.data.is_absolute()


def test_invalid_split_configuration_is_rejected():
    with pytest.raises(ValidationError):
        ClassificationConfig(train_fraction=0.9)


def test_gpu_not_required_or_implicitly_enabled():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"embeddings": {"device": "mps"}})


def test_classifier_input_does_not_leak_labels(issue_factory):
    original = issue_factory()
    labelled = issue_factory(source_labels=("secret-target",), mapped_target=Category.BUG)
    assert model_text(original) == model_text(labelled)
    assert input_hash(original) == input_hash(labelled)
    assert "secret-target" not in model_text(labelled)


def test_eligibility_excludes_self_future_ties_and_other_repository(issue_factory):
    query = issue_factory()
    earlier = issue_factory(1, 1)
    history = [query, earlier, earlier, issue_factory(3, 3), issue_factory(4, 2)]
    history.append(issue_factory(5, 1, "different/project"))
    assert eligible_history(query, history) == (earlier,)


def test_conflicting_corpus_snapshot_is_not_silently_selected(issue_factory):
    earlier = issue_factory(1, 1)
    changed = issue_factory(1, 1, body="Edited text")
    with pytest.raises(ValueError, match="Conflicting"):
        eligible_history(issue_factory(), [earlier, changed])


def test_naive_dates_and_wrong_source_urls_are_rejected(issue_factory):
    issue = issue_factory()
    values = issue.model_dump()
    values["created_at"] = issue.created_at.replace(tzinfo=None)
    with pytest.raises(ValidationError):
        Issue(**values)
    with pytest.raises(ValidationError):
        issue_factory(url="https://github.com/another/repo/issues/2")


def test_accepted_prediction_requires_calibrated_distribution():
    with pytest.raises(ValidationError):
        ClassificationResult(predicted_label=Category.BUG, deferred=False)
    with pytest.raises(ValidationError):
        ClassificationResult(
            predicted_label=Category.BUG,
            calibrated_probabilities={category: 0.5 for category in Category},
        )
