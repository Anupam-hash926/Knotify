"""Streamlit workspace: frozen-history issues or pasted reports through the shared analyze()."""

import re
from datetime import UTC, datetime
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from knotify.config import load_config
from knotify.contracts import Issue
from knotify.drafting import compose_reply
from knotify.runtime import load_runtime
from knotify.triage import analyze

st.set_page_config(page_title="Knotify · Issue intelligence", page_icon="◈", layout="wide")


@st.cache_resource
def application_config():
    return load_config()


@st.cache_resource(show_spinner="Loading frozen history and models…")
def application_runtime():
    return load_runtime(application_config())


def escape_md(text: str) -> str:
    """Issue titles are untrusted: show them as literal text, not Markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~])", r"\\\1", text)


STATUS_MARK = {"present": "✓", "missing": "✗", "uncertain": "?", "not_applicable": "–"}


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
        label = item.item_id.value.replace("_", " ").capitalize()
        st.markdown(f"{STATUS_MARK[item.status]} **{label}** · {item.status.replace('_', ' ')}")
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


def render_report(report):
    classification = report.classification
    st.markdown(f"**{escape_md(report.issue.title)}**")
    if report.issue.url:
        st.caption(f"[{report.issue.repository} #{report.issue.number}]({report.issue.url})")
    st.markdown("##### Category")
    if classification.predicted_label is None:
        st.warning("Human review required — classifier not connected.")
    else:
        label = classification.predicted_label.value.replace("_", " / ")
        st.markdown(f"Suggested: **{label}**")
        if classification.deferred:
            st.caption(f"Deferred to human review. {classification.reason}")
        probabilities = classification.calibrated_probabilities or classification.raw_probabilities
        kind = "calibrated" if classification.calibrated_probabilities else "raw, uncalibrated"
        for name, value in sorted(probabilities.items(), key=lambda kv: -kv[1]):
            st.progress(value, text=f"{name.value.replace('_', ' / ')} · {value:.2f} ({kind})")
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
            st.markdown(f"{item.rank}. [#{issue.number}]({issue.url}) {escape_md(issue.title)}")
            st.caption(f"{issue.created_at:%d %b %Y} · {score_text(item.method_scores)}")
    render_checklist(report)
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
    with st.sidebar:
        st.markdown("## ◈ Knotify")
        st.caption("ISSUE INTELLIGENCE")
        st.divider()
        page = st.radio(
            "Workspace", ["Issue workspace", "Project readiness"], label_visibility="collapsed"
        )
        st.divider()
        st.caption("Local workspace · CPU")
        st.caption("Read-only assistance. Your repository stays in your control.")

    if page == "Project readiness":
        st.title("A clear starting point.")
        st.write(
            "The application shell is ready. The intelligence pipeline is the next build step."
        )
        st.dataframe(
            [
                {"Component": "Configuration & shared contracts", "Status": "Ready"},
                {"Component": "Report orchestration & local storage", "Status": "Ready"},
                {"Component": "GitHub import & frozen dataset", "Status": "Ready"},
                {"Component": "TF-IDF classification (uncalibrated)", "Status": "Ready"},
                {"Component": "Calibration & deferral threshold", "Status": "Not implemented"},
                {"Component": "BM25 search", "Status": "Ready"},
                {"Component": "Dense / hybrid search", "Status": "Ready"},
                {"Component": "Checklist & template questions", "Status": "Ready"},
                {"Component": "Evaluation runner & review controls", "Status": "Not implemented"},
            ],
            hide_index=True,
            use_container_width=True,
        )
        if runtime.dataset_version:
            st.caption(f"Frozen dataset: {runtime.repository} · {runtime.dataset_version}")
        for note in runtime.notes:
            st.caption(note)
        return

    st.caption("WORKSPACE / ISSUE TRIAGE")
    st.title("Less noise. More context.")
    st.write(
        "Bring an issue into focus with clear categories, related reports and useful next steps."
    )
    left, right = st.columns([1.15, 1], gap="large")
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
                st.caption("Your report will appear here.")
                st.write("Category · Related issues · Missing details")
                st.write("Recommendations will keep their evidence and uncertainty visible.")
            else:
                if report.issue.origin == "pasted":
                    st.caption("Pasted report: history cutoff is the analysis time.")
                render_report(report)


main()
