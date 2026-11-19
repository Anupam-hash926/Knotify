from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "ui" / "app.py"


def test_empty_state_and_report_preview_are_honest():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert "Less noise" in app.title[0].value
    app.text_input[0].set_value("example/project")
    app.text_input[1].set_value("Login fails")
    app.text_area[0].set_value("An error appears after sign-in.")
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert any("classifier not connected" in warning.value for warning in app.warning)
    assert app.session_state["report"].classification.predicted_label is None


def test_invalid_input_and_readiness_page():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    app.button[0].click().run(timeout=20)
    assert app.error
    app.radio[0].set_value("Project readiness").run(timeout=20)
    assert not app.exception
    assert "clear starting point" in app.title[0].value
