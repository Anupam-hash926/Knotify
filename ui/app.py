"""Runnable Streamlit shell. Product ML remains unavailable until adapters are connected."""

from datetime import UTC, datetime
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from knotify.config import load_config
from knotify.contracts import Issue
from knotify.triage import ModelBundle, analyze

st.set_page_config(page_title="Knotify · Issue intelligence", page_icon="◈", layout="wide")


@st.cache_resource
def application_config():
    return load_config()


def main():
    config = application_config()
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
                {"Component": "GitHub import & frozen dataset", "Status": "Not implemented"},
                {"Component": "Classification & calibration", "Status": "Not implemented"},
                {"Component": "BM25 / dense / hybrid search", "Status": "Not implemented"},
                {"Component": "Checklist & evidence-linked questions", "Status": "Not implemented"},
                {"Component": "Evaluation runner & review controls", "Status": "Not implemented"},
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.info(
            "No credentials, model downloads or repository imports are needed to open this shell."
        )
        return

    st.caption("WORKSPACE / ISSUE TRIAGE")
    st.title("Less noise. More context.")
    st.write(
        "Bring an issue into focus with clear categories, related reports and useful next steps."
    )
    st.info("Scaffold preview: input and report structure work; ML analysis is not connected yet.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Start with an issue")
            with st.form("issue_input"):
                repository = st.text_input("Repository", placeholder="owner/repository")
                title = st.text_input("Issue title", placeholder="What happened?")
                body = st.text_area("Description", height=220, placeholder="Steps, context, logs…")
                submitted = st.form_submit_button("Preview report structure", type="primary")
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
                    st.session_state.report = analyze(issue, (), ModelBundle(), config)
    with right:
        with st.container(border=True):
            st.subheader("Triage overview")
            report = st.session_state.get("report")
            if report is None:
                st.caption("Your report will appear here.")
                st.write("Category · Related issues · Missing details")
                st.write("Recommendations will keep their evidence and uncertainty visible.")
            else:
                st.text(report.issue.title)
                st.caption("Preview of the last submitted input; no prediction has been made.")
                st.warning("Human review required — classifier not connected.")
                for name, status in report.components.items():
                    st.write(f"**{name.title()}** · {status.state}")
                    st.caption(status.detail)
                with st.expander("Structured report"):
                    st.json(report.model_dump(mode="json"))
                st.download_button(
                    "Download preview JSON",
                    data=report.model_dump_json(indent=2),
                    file_name=f"knotify-preview-{report.analysis_id}.json",
                    mime="application/json",
                )


main()
