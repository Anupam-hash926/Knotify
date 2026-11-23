from datetime import UTC, datetime

import pytest

from knotify.contracts import Issue


@pytest.fixture
def issue_factory():
    def make(number=2, day=2, repository="example/project", **changes):
        values = {
            "issue_id": str(number),
            "repository": repository,
            "number": number,
            "title": "Crash during login",
            "body": "A traceback appears after signing in.",
            "created_at": datetime(2026, 1, day, tzinfo=UTC),
            "url": f"https://github.com/{repository}/issues/{number}",
        }
        return Issue(**(values | changes))

    return make


@pytest.fixture
def empty_workspace(tmp_path, monkeypatch):
    """Point the app at empty data/artifact folders so UI tests never use local datasets."""
    config = tmp_path / "test.toml"
    paths = {name: (tmp_path / name).as_posix() for name in ("data", "artifacts", "runs")}
    config.write_text(
        "[paths]\n" + "".join(f'{k} = "{v}"\n' for k, v in paths.items()), encoding="utf-8"
    )
    monkeypatch.setenv("KNOTIFY_CONFIG", str(config))
    import streamlit as st

    st.cache_resource.clear()  # the app caches its config/runtime per process
    return tmp_path
