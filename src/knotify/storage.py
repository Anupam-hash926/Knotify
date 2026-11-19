"""Local append-only report/review files; no database or user-defined output filename."""

from pathlib import Path

from knotify.contracts import AnalysisReport, Review


def save_report(report: AnalysisReport, runs_dir: Path) -> Path:
    directory = runs_dir / "analyses"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.analysis_id}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    return path


def save_review(review: Review, runs_dir: Path) -> Path:
    report_path = runs_dir / "analyses" / f"{review.analysis_id}.json"
    # Validate saved report before accepting a review for it.
    saved = AnalysisReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if saved.analysis_id != review.analysis_id:
        raise ValueError("Saved analysis does not match the review.")
    directory = runs_dir / "reviews"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{review.review_id}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(review.model_dump_json(indent=2) + "\n")
    return path
