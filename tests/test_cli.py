import pytest

from knotify.cli import main


def test_doctor_is_honest_and_does_not_print_credentials(capsys, monkeypatch, empty_workspace):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-not-for-output")
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert '"device": "cpu"' in output
    assert '"frozen_snapshots": []' in output and '"trained_models": []' in output
    assert "secret-not-for-output" not in output


@pytest.mark.parametrize("command", ["calibrate", "evaluate"])
def test_reserved_commands_fail_instead_of_claiming_work(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2
    assert "not implemented" in capsys.readouterr().err
