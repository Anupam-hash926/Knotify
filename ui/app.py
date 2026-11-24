"""Streamlit workspace: frozen-history issues or pasted reports through the shared analyze()."""

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from knotify.config import load_config
from knotify.contracts import Category, Issue, Review
from knotify.drafting import compose_reply
from knotify.runtime import load_runtime
from knotify.storage import save_report, save_review
from knotify.triage import analyze

ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(
    page_title="Knotify · issue triage",
    page_icon=str(ASSETS / "knotify_icon.svg"),
    layout="wide",
)


@st.cache_resource
def application_config():
    return load_config()


@st.cache_resource(show_spinner="Loading frozen history and models…")
def application_runtime():
    return load_runtime(application_config())


def escape_md(text: str) -> str:
    """Issue titles are untrusted: show them as literal text, not Markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~])", r"\\\1", text)


# Presentation only: colour and wording for categories and checklist states.
CATEGORY_STYLE = {
    "bug": ("Bug", "red", ":material/bug_report:"),
    "feature": ("Feature request", "blue", ":material/add_circle:"),
    "question_documentation": ("Question / docs", "violet", ":material/menu_book:"),
    "other": ("Maintenance", "gray", ":material/build:"),
}
STATUS_BADGE = {
    "present": ":green-badge[:material/check: present]",
    "missing": ":red-badge[:material/close: missing]",
    "uncertain": ":orange-badge[:material/help: unsure]",
    "not_applicable": ":gray-badge[n/a]",
}
ITEM_LABEL = {
    "reproduction_steps": "Reproduction steps",
    "expected_behavior": "Expected behaviour",
    "actual_behavior": "Actual behaviour",
    "environment": "Environment (OS / platform)",
    "version": "Library version",
    "error_logs": "Error message or traceback",
}


def category_badge(value: str) -> str:
    name, color, icon = CATEGORY_STYLE[value]
    return f":{color}-badge[{icon} {name}]"


def summary_strip(report) -> str:
    """One line of badges: category state, related issues, missing details."""
    parts = []
    c = report.classification
    if c.predicted_label is not None:
        parts.append(category_badge(c.predicted_label.value))
        parts.append(
            ":orange-badge[:material/front_hand: Needs a maintainer's decision]"
            if c.deferred
            else ":green-badge[:material/verified: Auto-labelled]"
        )
    if report.components.get("retrieval") and report.components["retrieval"].state == "ready":
        parts.append(f":blue-badge[:material/link: {len(report.related_issues)} related]")
    missing = sum(item.status == "missing" for item in report.checklist)
    if report.checklist and not all(i.status == "not_applicable" for i in report.checklist):
        parts.append(
            f":red-badge[{missing} missing detail{'s' if missing != 1 else ''}]"
            if missing
            else ":green-badge[details complete]"
        )
    return " ".join(parts)


def score_text(scores: dict[str, float]) -> str:
    parts = []
    if "bm25" in scores:
        rank = f" (#{int(scores['bm25_rank'])})" if "bm25_rank" in scores else ""
        parts.append(f"BM25 {scores['bm25']:.1f}{rank}")
    elif "rrf" in scores:
        parts.append("BM25 —")
    if "dense" in scores:
        parts.append(f"cosine {scores['dense']:.2f} (#{int(scores['dense_rank'])})")
    elif "rrf" in scores:
        parts.append("embedding —")
    return " · ".join(parts)


def render_checklist(report):
    st.markdown("##### Missing information")
    status = report.components.get("checklist")
    if status is None or status.state != "ready":
        st.caption(status.detail if status else "Checklist unavailable.")
        return
    if all(item.status == "not_applicable" for item in report.checklist):
        st.caption("Not applicable: the accepted category is not a bug report.")
        return
    for item in report.checklist:
        st.markdown(f"{STATUS_BADGE[item.status]} **{ITEM_LABEL[item.item_id.value]}**")
        st.caption(item.reason)
    reply = compose_reply(report.suggestions)
    st.markdown("##### Suggested follow-up")
    if not reply:
        st.caption("Nothing to ask: no applicable item is missing.")
        return
    st.text_area(
        "Draft reply (edit before using; nothing is posted automatically)",
        value=reply,
        height=170,
        key=f"reply-{report.analysis_id}",
    )
    st.caption(
        "Each question cites the checklist item it comes from. Citations are checked "
        "mechanically; the checklist rules themselves can be wrong."
    )


def render_review(report):
    st.markdown("##### Maintainer review")
    runs = application_config().paths.runs
    saved = st.session_state.get("saved_reports", {})
    key = str(report.analysis_id)
    if key not in saved:
        st.caption("Save the report first; reviews always refer to a saved, unchanged report.")
        if st.button("Save report", key=f"save-{key}"):
            try:
                saved[key] = save_report(report, runs).name
                st.session_state.saved_reports = saved
                st.rerun()
            except OSError:
                st.error("The report could not be saved locally.")
        return
    st.caption(f"Saved locally as {saved[key]}.")
    with st.form(f"review-{key}"):
        decision = st.radio(
            "Decision", ["accepted", "rejected", "edited"], horizontal=True, key=f"d-{key}"
        )
        options = ["(no change)"] + [c.value for c in Category]
        corrected = st.selectbox("Correct category (if edited)", options, key=f"c-{key}")
        note = st.text_input("Note (optional)", key=f"n-{key}")
        submitted = st.form_submit_button("Record review")
    if submitted:
        try:
            review = Review(
                analysis_id=report.analysis_id,
                decision=decision,
                corrected_category=None if corrected == "(no change)" else Category(corrected),
                note=note,
            )
            path = save_review(review, runs)
        except (ValueError, OSError) as exc:
            st.error(
                "Review not saved: an edited review needs a corrected category or a note."
                if isinstance(exc, ValueError)
                else "Review not saved: local storage failed."
            )
        else:
            st.success(f"Review recorded ({path.name}). The original report is unchanged.")


def render_about(runtime):
    import json

    from knotify.config import PROJECT_ROOT

    st.title("About this model")
    st.write(
        "Knotify suggests a category, finds related earlier issues and drafts questions for "
        "missing information. A maintainer always makes the decision; nothing is posted."
    )
    bundle = runtime.bundle
    classifier = type(bundle.classifier).__name__
    rows = [
        {
            "Part": "Frozen dataset",
            "Detail": f"{runtime.repository or '—'} · {len(runtime.history):,} issues"
            if runtime.history
            else "not loaded",
        },
        {
            "Part": "Category model",
            "Detail": "TF-IDF + logistic regression, calibrated, threshold from validation"
            if classifier == "CalibratedTextClassifier"
            else classifier.replace("Unavailable", "unavailable "),
        },
        {
            "Part": "Related issues",
            "Detail": {
                "HybridRetriever": "BM25 + MiniLM embeddings (rank fusion)",
                "BM25Retriever": "BM25 only",
            }.get(type(bundle.retriever).__name__, "unavailable"),
        },
        {"Part": "Missing information", "Detail": "rule checklist v1 + template questions"},
        {"Part": "LLM", "Detail": "not used"},
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    results = PROJECT_ROOT / "docs" / "results" / "classification.json"
    if results.exists():
        c = json.loads(results.read_text(encoding="utf-8"))
        sel = c["selected"]
        frozen = sel["at_frozen_threshold"]
        st.subheader("Measured on held-out Test issues")
        cols = st.columns(3)
        cols[0].metric("Macro-F1", f"{c['models']['tfidf_lr_masked']['macro_f1']:.3f}")
        cols[1].metric("Auto-categorised", f"{100 * frozen['coverage']:.1f}%")
        cols[2].metric("Accuracy when auto", f"{100 * frozen['accepted_accuracy']:.1f}%")
        st.caption(
            f"{c['test_count']} scikit-learn issues created after the training period. "
            "Full results, limitations and what was not completed: docs/results/results.md."
        )
    if runtime.dataset_version:
        st.caption(f"Dataset version {runtime.dataset_version}")
    for note in runtime.notes:
        st.caption(note)


def render_report(report):
    classification = report.classification
    st.markdown(f"#### {escape_md(report.issue.title)}")
    if report.issue.url:
        st.caption(f"[{report.issue.repository} #{report.issue.number}]({report.issue.url})")
    strip = summary_strip(report)
    if strip:
        st.markdown(strip)
    st.markdown("##### Category")
    if classification.predicted_label is None:
        st.warning("Human review required — classifier not connected.")
    else:
        label = CATEGORY_STYLE[classification.predicted_label.value][0].lower()
        if classification.deferred:
            st.markdown(f"Candidate: {category_badge(classification.predicted_label.value)}")
        else:
            st.markdown(f"Suggested: {category_badge(classification.predicted_label.value)}")
        st.caption(classification.reason)
        probabilities = classification.calibrated_probabilities or classification.raw_probabilities
        kind = "calibrated" if classification.calibrated_probabilities else "raw, uncalibrated"
        for name, value in sorted(probabilities.items(), key=lambda kv: -kv[1]):
            st.progress(value, text=f"{name.value.replace('_', ' / ')} · {value:.2f} ({kind})")
        st.caption("Confidence applies to the category only, not to search results or questions.")
        if classification.explanation:
            with st.expander("Why this candidate? (word contributions)"):
                for token in classification.explanation:
                    sign = "supports" if token.contribution > 0 else "argues against"
                    st.markdown(f"`{token.token}` · {sign} {label} ({token.contribution:+.3f})")
                st.caption(classification.explanation_note)
    st.markdown("##### Related earlier issues")
    if report.components["retrieval"].state != "ready":
        st.caption(report.components["retrieval"].detail)
    elif not report.related_issues:
        st.caption("No earlier issue in this repository matched.")
    else:
        hybrid = any("rrf" in item.method_scores for item in report.related_issues)
        method = "hybrid search (BM25 + embeddings, rank fusion)" if hybrid else "BM25 search"
        st.caption(f"Ranked by {method}. Similarity only, not a probability or duplicate proof.")
        for item in report.related_issues:
            issue = item.issue
            st.markdown(
                f":gray-badge[{item.rank}] [#{issue.number}]({issue.url}) {escape_md(issue.title)}"
            )
            st.caption(f"{issue.created_at:%d %b %Y} · {score_text(item.method_scores)}")
    render_checklist(report)
    render_review(report)
    with st.expander("Component status"):
        for name, status in report.components.items():
            st.write(f"**{name.title()}** · {status.state}")
            st.caption(status.detail)
    with st.expander("Structured report"):
        st.json(report.model_dump(mode="json"))
    st.download_button(
        "Download report JSON",
        data=report.model_dump_json(indent=2),
        file_name=f"knotify-report-{report.analysis_id}.json",
        mime="application/json",
    )


def main():
    config = application_config()
    runtime = application_runtime()
    st.logo(
        str(ASSETS / "knotify_logo.svg"), icon_image=str(ASSETS / "knotify_icon.svg"), size="large"
    )
    with st.sidebar:
        st.caption("Issue triage that shows its evidence.")
        page = st.radio(
            "Workspace", ["Issue workspace", "About this model"], label_visibility="collapsed"
        )
        st.divider()
        st.caption(":material/lock: Runs on this computer. Nothing is posted to GitHub.")
        st.caption(":material/person: A maintainer makes every decision.")

    if page == "About this model":
        render_about(runtime)
        return

    st.title("Less noise. More context.")
    st.write(
        "Open an issue or paste a new report. Knotify suggests a category, finds related earlier "
        "reports and drafts questions for missing details. You make the call."
    )
    left, right = st.columns([0.9, 1.25], gap="large")
    with left:
        if runtime.history:
            with st.container(border=True):
                st.subheader("Open an imported issue")
                st.caption(
                    f"{runtime.repository} · frozen snapshot of {len(runtime.history):,} issues"
                )
                with st.form("imported_issue"):
                    number = st.number_input("Issue number", min_value=1, step=1, value=None)
                    opened = st.form_submit_button("Analyze issue", type="primary")
                if opened:
                    issue = runtime.issue(int(number)) if number else None
                    if issue is None:
                        st.error("That number is not an issue in the frozen snapshot.")
                    else:
                        with st.spinner("Analyzing…"):
                            st.session_state.report = analyze(
                                issue,
                                runtime.history,
                                runtime.bundle,
                                config,
                                dataset_version=runtime.dataset_version,
                            )
        with st.container(border=True):
            st.subheader("Or paste a new report")
            with st.form("issue_input"):
                repository = st.text_input(
                    "Repository", value=runtime.repository or "", placeholder="owner/repository"
                )
                title = st.text_input("Issue title", placeholder="What happened?")
                body = st.text_area("Description", height=220, placeholder="Steps, context, logs…")
                submitted = st.form_submit_button("Analyze report")
            if submitted:
                try:
                    issue = Issue(
                        issue_id=f"pasted:{uuid4()}",
                        repository=repository.strip(),
                        title=title.strip(),
                        body=body,
                        created_at=datetime.now(UTC),
                        origin="pasted",
                    )
                except ValidationError:
                    st.session_state.pop("report", None)
                    st.error("Enter a repository as owner/name and a nonempty issue title.")
                else:
                    st.session_state.report = analyze(
                        issue,
                        runtime.history,
                        runtime.bundle,
                        config,
                        dataset_version=runtime.dataset_version,
                    )
    with right:
        with st.container(border=True):
            st.subheader("Triage overview")
            report = st.session_state.get("report")
            if report is None:
                st.caption("Nothing analysed yet.")
                if runtime.history:
                    st.write(
                        "Try issue **32072** for a full example: an auto-labelled bug whose "
                        "confirmed duplicate is ranked first. Issue **29019** shows missing "
                        "details and a drafted reply."
                    )
                else:
                    st.write(
                        "Paste a report to see its checklist. Categories and related issues "
                        "appear once the shared data and models are installed."
                    )
            else:
                if report.issue.origin == "pasted":
                    st.caption("Pasted report: history cutoff is the analysis time.")
                render_report(report)


main()
