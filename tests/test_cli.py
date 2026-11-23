import pytest

from knotify.cli import main


def test_doctor_is_honest_and_does_not_print_credentials(capsys, monkeypatch, empty_workspace):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-not-for-output")
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert '"device": "cpu"' in output
    assert '"frozen_snapshots": []' in output and '"trained_models": []' in output
    assert "secret-not-for-output" not in output


def test_unknown_command_fails_without_claiming_work(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
