# Implementation status

## Current milestone

**Scaffold complete; product Day 1 is not complete.** No repositories have been selected, no
dataset has been collected, and no trained ML model or measured experiment exists.

## First implementation gate: dataset feasibility

The source is public issues in existing open-source GitHub repositories, not the empty Knotify
repository. Before model implementation/tuning, an audit must establish:

- Source repository A and transfer repository B, with rationale and available time ranges.
- Counts for the four audited target classes, including conflicting/unmapped labels.
- Whether the proposed chronological splits have usable class support.
- Explicit earlier duplicate targets or enough manually judged related pairs for retrieval evaluation.
- A dated raw snapshot, reproducible mapping and frozen split IDs.

The proposal's 2,000–4,000 usable labelled examples are a target, not a confirmed resource.
Additional unlabelled issues can populate retrieval history. A duplicate label without a target
does not constitute pair-level relevance ground truth. The present repository mapping config is
only an example. Data shortages or taxonomy deviations must be recorded rather than concealed.

## Implemented foundation

- Python 3.12 editable package, uv lock, separated optional ML extras, Ruff and pytest.
- Validated CPU-only configuration and secret-free environment template.
- Shared typed issue, classification, checklist, retrieval, report and review contracts.
- Classifier/retrieval/checklist protocols and explicit unavailable adapters.
- Shared orchestration with independent component failure states and retrieval-output checks.
- Title/body feature boundary, stable hashes and historical-corpus eligibility helper.
- Small retrieval/selective metric primitives (not an experiment runner).
- Append-only local report/review storage functions.
- Streamlit input/report-preview shell, readiness view and restrained visual theme.
- Honest CLI diagnostics/schema export; unfinished CLI actions return failure.
- Cross-platform GitHub Actions workflow definition.
- Proposal, final plan and factual `context.md` with complete PDF text appendix.

## Local verification (Windows, Python 3.12.16)

- `uv sync --locked --python 3.12`: succeeded with default/dev dependencies.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `uv run pytest -q`: **22 tests passed**, including Streamlit AppTest interactions.
- `uv run knotify doctor`: CPU selected; optional ML backends absent; LLM disabled.
- Both full PDF pages are represented in the context appendix; the original PDF is unchanged.
- Ignore checks cover credentials, datasets, model artifacts and run output.

The default environment was exercised locally. The target MacBook, optional ML packages, trained
models and external APIs have not been exercised. Hosted CI status is separate from the workflow
definition. AppTest checks application behavior; it is not a pixel-level design approval.

## Next product integration milestone

Dataset audit/import and frozen splits → TF-IDF baseline → eligible-corpus BM25 → a real issue
rendered through the shared report service. Later milestones remain those in the final plan.

Unfinished: GitHub client, mappings/splits, training/calibration/threshold selection, embedding
cache, BM25/FAISS/RRF, checklist rules, evidence registry/validator/templates, LLM adapter,
experiment runner, explanation panel and persistent review controls in the UI.

## Run commands

```sh
uv sync --locked --python 3.12
uv run knotify doctor
uv run streamlit run ui/app.py
uv run pytest
```

No local data/artifact paths need populating to launch the scaffold. The ignored `runs/` directory
is the default destination when storage functions are used; the preview UI only downloads JSON.
