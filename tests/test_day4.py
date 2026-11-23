from openpyxl import load_workbook

from knotify.annotation import write_checklist_sheet, write_suggestion_sheet
from knotify.evaluation import (
    ITEMS,
    paired_bootstrap,
    prefill_suggestion_support,
    read_checklist_sheet,
    score_checklist,
    score_suggestions,
)


def _record(n):
    return {
        "number": n,
        "title": f"t{n}",
        "body": "b",
        "html_url": f"https://github.com/o/r/issues/{n}",
        "created_at": f"2025-01-0{n}T00:00:00Z",
    }


def _fill(path, rows):
    wb = load_workbook(path)
    ws = wb.worksheets[1]
    for r, (is_bug, statuses) in enumerate(rows, start=2):
        ws.cell(r, 5).value = is_bug
        for j, status in enumerate(statuses):
            ws.cell(r, 6 + j).value = status
        ws.cell(r, 12).value = "AGW"
    wb.save(path)


def test_checklist_scoring_counts_missing_as_positive_and_excludes_correctly(tmp_path):
    sheet = tmp_path / "c.xlsx"
    write_checklist_sheet([_record(1), _record(2), _record(3)], sheet)
    _fill(
        sheet,
        [
            ("yes", ["missing", "present", "present", "missing", "uncertain", "not_applicable"]),
            ("no", [""] * 6),
            ("yes", ["present"] * 6),
        ],
    )
    gold = read_checklist_sheet(sheet)
    samples = [
        {
            "number": 1,
            "rule_output": dict(
                zip(
                    ITEMS,
                    ["missing", "missing", "present", "uncertain", "present", "present"],
                    strict=True,
                )
            ),
        },
        {"number": 2, "rule_output": dict.fromkeys(ITEMS, "missing")},
        {"number": 3, "rule_output": dict.fromkeys(ITEMS, "present")},
    ]
    result = score_checklist(samples, gold)
    overall = result["overall"]
    # issue 1: repro TP, expected FP, environment truth missing / pred uncertain -> FN
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 1)
    assert result["excluded"]["not_bug_or_blank"] == 1
    assert result["excluded"]["gold_uncertain"] == 1
    assert result["excluded"]["gold_not_applicable"] == 1
    assert result["predicted_uncertain_on_scored_items"] == 1
    assert result["bug_reports_scored"] == 2 and result["items_scored"] == 10
    assert result["reviewers"] == ["AGW"]


def test_suggestion_support_is_prefilled_from_checklist_and_scored(tmp_path):
    sheet = tmp_path / "c.xlsx"
    write_checklist_sheet([_record(1)], sheet)
    _fill(sheet, [("yes", ["missing", "present", "present", "present", "present", "present"])])
    gold = read_checklist_sheet(sheet)
    sug = tmp_path / "s.xlsx"
    entries = [
        {
            "number": 1,
            "title": "t",
            "url": "https://github.com/o/r/issues/1",
            "text": "q1",
            "item": "reproduction_steps",
        },
        {
            "number": 1,
            "title": "t",
            "url": "https://github.com/o/r/issues/1",
            "text": "q2",
            "item": "version",
        },
    ]
    write_suggestion_sheet(entries, sug)
    assert prefill_suggestion_support(sug, gold) == 2
    ws = load_workbook(sug).worksheets[1]
    assert [ws.cell(2, 6).value, ws.cell(3, 6).value] == ["yes", "no"]
    assert "derived from checklist" in ws.cell(2, 9).value
    result = score_suggestions(sug)
    assert result["supported_yes"] == 1 and result["unsupported_no"] == 1
    assert result["unsupported_rate"] == 0.5 and result["appropriate_blank"] == 2


def test_paired_bootstrap_is_reproducible():
    y = ["bug", "feature", "bug", "other"] * 10
    a = ["bug", "feature", "bug", "bug"] * 10
    b = ["bug", "bug", "bug", "bug"] * 10
    first = paired_bootstrap(y, a, b, seed=1, resamples=50)
    assert first == paired_bootstrap(y, a, b, seed=1, resamples=50)
    assert first["difference"] > 0
