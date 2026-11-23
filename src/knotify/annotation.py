"""Frozen evaluation samples and Excel review sheets for human judgement.

* Checklist sheet: 30 bug-labelled issues from the validation/test periods (never the 10 Train
  issues used to develop the rules), stratified 15 issue-form / 15 free-text so that missing
  items occur. Retrieval evaluation queries labelled bug are taken first. Rule outputs are
  **hidden** from the sheet (stored separately in the sample key) so judgements are blind.
* Suggestion sheet: up to 20 of those issues that receive template questions; the person judges
  whether each cited item is really missing (support) and whether asking is appropriate.

Test-period issues are used only to draw evaluation samples; no model is tuned on them.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from knotify.contracts import ChecklistItemId
from knotify.data import clean_markdown

FORM_MARKERS = ("steps/code to reproduce", "reproducible example", "expected results")
STATUSES = '"present,missing,uncertain,not_applicable"'
ITEM_HELP = {
    "reproduction_steps": "Code example or steps someone could run to see the problem",
    "expected_behavior": "What the reporter expected to happen",
    "actual_behavior": "What actually happened (output, wrong result, error)",
    "environment": "Operating system / platform (e.g. Windows 11, macOS, Linux)",
    "version": "Version of scikit-learn (or another library) used",
    "error_logs": "Error message or traceback - not_applicable if no error/crash is described",
}


def _uses_form(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in FORM_MARKERS)


def _font(**kwargs):
    return Font(**({"name": "Arial", "size": 10} | kwargs))


def _style_table(ws, widths, input_from: int, height: int) -> None:
    header = PatternFill("solid", fgColor="2F5233")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    for i, width in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = width
        cell = ws.cell(1, i)
        cell.font = _font(bold=True, color="FFFFFF")
        cell.fill = header
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = 45
    for row in range(2, ws.max_row + 1):
        ws.row_dimensions[row].height = height
        for col in range(1, len(widths) + 1):
            cell = ws.cell(row, col)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if cell.hyperlink is None:
                cell.font = _font()
            if col >= input_from:
                cell.fill = yellow
    ws.freeze_panes = "C2"


def _guide(wb, title: str, lines: list[tuple[str, bool]]) -> None:
    ws = wb.create_sheet("How to fill", 0)
    ws.column_dimensions["A"].width = 115
    for text, bold in [(title, True), *lines]:
        ws.append([text])
        cell = ws.cell(ws.max_row, 1)
        cell.font = _font(bold=bold, size=12 if bold else 10)
        cell.alignment = Alignment(wrap_text=True)


def select_checklist_sample(
    rows: list[dict], retrieval_queries: set[int], *, per_stratum: int = 15, seed: int = 42
) -> list[dict]:
    rng = random.Random(seed)
    bugs = [r for r in rows if r["target"] == "bug"]
    chosen: list[dict] = []
    for form in (True, False):
        stratum = [r for r in bugs if _uses_form(r["body"]) == form]
        first = [r for r in stratum if r["number"] in retrieval_queries]
        rest = [r for r in stratum if r["number"] not in retrieval_queries]
        rng.shuffle(rest)
        chosen += (first + rest)[:per_stratum]
    return sorted(chosen, key=lambda r: r["created_at"])


def write_checklist_sheet(sample: list[dict], path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Checklist"
    items = [i.value for i in ChecklistItemId]
    ws.append(
        [
            "#",
            "Issue",
            "Title",
            "Body (cleaned; open the link for full text)",
            "Is this a bug report? (yes/no/unclear)",
            *items,
            "Your initials",
            "Notes",
        ]
    )
    for n, r in enumerate(sample, 1):
        body = clean_markdown(r["body"])
        body = body if len(body) <= 3000 else body[:3000] + "\n…(truncated; see GitHub)"
        ws.append([n, r["number"], r["title"], body, "", *[""] * len(items), "", ""])
        ws.cell(ws.max_row, 2).hyperlink = r["html_url"]
        ws.cell(ws.max_row, 2).font = _font(color="0563C1", underline="single")
    last = ws.max_row
    yes_no = DataValidation(type="list", formula1='"yes,no,unclear"', allow_blank=True)
    yes_no.add(f"E2:E{last}")
    status = DataValidation(type="list", formula1=STATUSES, allow_blank=True)
    status.add(f"F2:K{last}")
    ws.add_data_validation(yes_no)
    ws.add_data_validation(status)
    _style_table(ws, [5, 9, 30, 80, 14, *[15] * len(items), 10, 25], input_from=5, height=230)
    _guide(
        wb,
        "Checklist review: what information does each bug report contain?",
        [
            (
                "30 issues. For each, decide whether each piece of information is in the report. "
                "Only edit the yellow columns.",
                False,
            ),
            (
                "Column E: is it really a bug report? If 'no', leave the item columns empty.",
                False,
            ),
            ("Columns F-K, choose from the dropdown:", False),
            (
                "  present = clearly there   missing = not there   "
                "uncertain = partly / can't tell   not_applicable = doesn't apply here",
                False,
            ),
            *[(f"  {k}: {v}", False) for k, v in ITEM_HELP.items()],
            (
                "An empty template heading (e.g. '### Versions' with nothing under it) counts as "
                "missing.",
                False,
            ),
            (
                "Judge from the report text only (the title and the first message), not later "
                "comments.",
                False,
            ),
            (
                "Example: a report with a code snippet, 'expected X but got Y', a traceback and "
                "'sklearn 1.3 on Ubuntu' -> all present.",
                False,
            ),
            (
                "Save as .xlsx in place, then tell Claude. Don't open the suggestion sheet until "
                "this one is finished.",
                True,
            ),
        ],
    )
    wb.active = 1
    wb.save(path)


def write_suggestion_sheet(entries: list[dict], path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Suggestions"
    ws.append(
        [
            "#",
            "Issue",
            "Title",
            "Suggested question to the reporter",
            "Cites checklist item",
            "Is that item really missing? (yes/no/unclear)",
            "Appropriate to ask? (yes/no)",
            "Your initials",
            "Notes",
        ]
    )
    for n, e in enumerate(entries, 1):
        ws.append([n, e["number"], e["title"], e["text"], e["item"], "", "", "", ""])
        ws.cell(ws.max_row, 2).hyperlink = e["url"]
        ws.cell(ws.max_row, 2).font = _font(color="0563C1", underline="single")
    last = ws.max_row
    for col, formula in (("F", '"yes,no,unclear"'), ("G", '"yes,no"')):
        dv = DataValidation(type="list", formula1=formula, allow_blank=True)
        dv.add(f"{col}2:{col}{last}")
        ws.add_data_validation(dv)
    _style_table(ws, [5, 9, 34, 60, 20, 18, 14, 10, 25], input_from=6, height=60)
    _guide(
        wb,
        "Suggestion review: are the follow-up questions justified?",
        [
            ("Each row is one question Knotify would ask a reporter. Open the issue link.", False),
            (
                "Column F: is the cited information really missing from the report? "
                "(yes = the question is justified)",
                False,
            ),
            (
                "Column G: is it a sensible thing to ask this reporter? (e.g. asking for a "
                "traceback when nothing crashed is not appropriate)",
                False,
            ),
            (
                "Example: question 'Which OS are you using?' citing environment, and the report "
                "never mentions an OS -> F: yes, G: yes.",
                False,
            ),
            ("Save as .xlsx in place, then tell Claude.", True),
        ],
    )
    wb.active = 1
    wb.save(path)


def prepare_sheets(config, repo: str) -> dict[str, Any]:
    from knotify.experiment import latest_split, load_split_rows, slug
    from knotify.ingest import to_issue
    from knotify.triage import analyze

    split = latest_split(config.paths.data, repo)
    out = config.paths.data / "annotations" / slug(repo) / split.name
    key_path = out / "evaluation_samples.json"
    if key_path.exists():
        raise FileExistsError(f"Evaluation samples already frozen: {key_path}")
    rows = load_split_rows(config, split, ("validation", "test"))
    pool = rows["validation"] + rows["test"]
    queries = json.loads((out / "retrieval_queries.json").read_text(encoding="utf-8"))
    query_ids = {q["query"] for q in queries["evaluation"]}
    sample = select_checklist_sample(pool, query_ids, seed=config.seed)

    from knotify.runtime import load_runtime

    runtime = load_runtime(config, repo)
    key: dict[str, Any] = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checklist_rule_version": "checklist-v1",
        "selection": "bug-labelled validation/test issues; 15 form + 15 free-text; "
        "retrieval evaluation queries first; seed=config.seed",
        "checklist": [],
        "suggestions": [],
    }
    suggestion_entries = []
    for record in sample:
        issue = to_issue(record, repo)
        report = analyze(issue, (), runtime.bundle, config)  # retrieval not needed here
        key["checklist"].append(
            {
                "number": record["number"],
                "split": "test" if record in rows["test"] else "validation",
                "uses_form": _uses_form(record["body"]),
                "classification": report.classification.model_dump(
                    mode="json", include={"predicted_label", "deferred"}
                ),
                "rule_output": {i.item_id.value: i.status for i in report.checklist},
                "suggestions": [
                    {
                        "text": s.text,
                        "evidence_ids": list(s.evidence_ids),
                        "reference_valid": s.reference_valid,
                    }
                    for s in report.suggestions
                ],
            }
        )
        for s in report.suggestions:
            suggestion_entries.append(
                {
                    "number": record["number"],
                    "title": record["title"],
                    "url": record["html_url"],
                    "text": s.text,
                    "item": s.evidence_ids[0].split(":", 1)[1],
                }
            )
    rng = random.Random(config.seed)
    with_suggestions = sorted({e["number"] for e in suggestion_entries})
    rng.shuffle(with_suggestions)
    picked = set(with_suggestions[:20])
    suggestion_entries = [e for e in suggestion_entries if e["number"] in picked]
    key["suggestions"] = suggestion_entries
    key["suggestion_issue_count"] = len(picked)
    out.mkdir(parents=True, exist_ok=True)
    write_checklist_sheet(sample, out / "checklist_review.xlsx")
    write_suggestion_sheet(suggestion_entries, out / "suggestion_review.xlsx")
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return {
        "checklist_issues": len(sample),
        "form": sum(e["uses_form"] for e in key["checklist"]),
        "test_period": sum(e["split"] == "test" for e in key["checklist"]),
        "suggestion_rows": len(suggestion_entries),
        "suggestion_issues": len(picked),
        "out": str(out),
    }


def write_retrieval_sheet(pools: dict, history_by_id: dict, path: Path) -> int:
    """Method-blind relevance judgements for pooled candidates (optional spot-check)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Retrieval"
    ws.append(
        [
            "#",
            "Query issue",
            "Query title",
            "Earlier issue",
            "Earlier issue title",
            "Judgement",
            "Your initials",
            "Notes",
        ]
    )
    n = 0
    for pool in pools["pools"]:
        query = history_by_id[pool["query"]]
        for candidate_id in pool["pool"]:
            candidate = history_by_id[candidate_id]
            n += 1
            ws.append([n, query.number, query.title, candidate.number, candidate.title, "", "", ""])
            for col, issue in ((2, query), (4, candidate)):
                ws.cell(ws.max_row, col).hyperlink = str(issue.url)
                ws.cell(ws.max_row, col).font = _font(color="0563C1", underline="single")
    dv = DataValidation(
        type="list", formula1='"duplicate,related,irrelevant,uncertain"', allow_blank=True
    )
    dv.add(f"F2:F{ws.max_row}")
    ws.add_data_validation(dv)
    _style_table(ws, [5, 9, 40, 9, 45, 14, 10, 25], input_from=6, height=45)
    _guide(
        wb,
        "Retrieval spot-check (optional): is each earlier issue relevant to the query issue?",
        [
            (
                "10 query issues, each with 6-14 earlier issues found by Knotify's search methods "
                "(mixed together; you can't tell which method found which - that's deliberate).",
                False,
            ),
            ("Open both links if the titles aren't enough. Column F:", False),
            (
                "  duplicate = the same problem/request   related = clearly connected (same "
                "feature, similar bug) but not the same   irrelevant = unrelated   uncertain = "
                "can't tell",
                False,
            ),
            (
                "Example: query 'KMeans crashes on sparse input' vs earlier 'KMeans fails with "
                "sparse matrix' -> duplicate; vs 'MiniBatchKMeans slow on sparse data' -> related.",
                False,
            ),
            ("Save as .xlsx in place, then tell Claude.", True),
        ],
    )
    wb.active = 1
    wb.save(path)
    return n
