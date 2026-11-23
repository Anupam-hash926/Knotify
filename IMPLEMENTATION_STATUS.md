# Implementation status

## Current milestone

**Day 1 gate met (22 Nov 2026):** a real imported scikit-learn issue is displayed in Streamlit with
its TF-IDF category suggestion (raw, uncalibrated, always deferred) and up to five earlier BM25
candidates with real links. Dataset feasibility is known and the baseline can be rerun from frozen
data. Day 2 work (embeddings, dense/RRF, checklist, templates) has not started.

The dataset decision, counts and corrections to the external research are recorded in
`context.md` → "Verified dataset decision". Audit evidence: `docs/audit/`.

## Completed

| Step | Result | Where |
|---|---|---|
| Read-only audit | 203 boolean issue-search counts over 4 candidates; labels, usable/conflict counts, yearly coverage, duplicate volumes, sampled verified pairs | `docs/audit/search_audit_20261122T113553Z.{md,json}` |
| Template check | Both repos' issue templates auto-apply the type label + `Needs Triage`; pandas also prefixes titles | `context.md` decision |
| A snapshot | scikit-learn, 12,387 issues, 2010-08-31 → 2026-11-21, complete | `data/raw/scikit-learn__scikit-learn/2010-01-01_2026-11-21_20261122T115450Z/` |
| B snapshot | pandas, 1,912 issues, 2024-09-11 → 2026-11-21 (after A's validation cutoff), complete | `data/raw/pandas-dev__pandas/2024-09-11_2026-11-21_20261122T120313Z/` |
| Mapping | Adopted for A and B | `configs/repositories.toml` |
| Frozen split (A) | 3,271 supervised (≥ 2019, single target, triaged, exact-dedup) → Train 1,963 / Val 654 (V-cal 327, V-policy 327) / Test 654, all four classes in each | `data/splits/scikit-learn__scikit-learn/2010-01-01_2026-11-21_20261122T115450Z/` |
| Baseline | TF-IDF (1–2-gram, technical-token pattern) + LR `C=1`, balanced, seed 42; majority baseline | `artifacts/scikit-learn__scikit-learn/…/tfidf_lr_{masked,raw}.joblib` |
| Validation result (V-policy, n=327; **Test untouched**) | macro-F1: masked **0.809**, raw 0.827, majority 0.113. Masked per-class F1: bug 0.835, feature 0.863, question_documentation 0.813, other 0.724 | `runs/baseline/scikit-learn__scikit-learn/…/baseline_validation.json`, `v_policy_predictions.csv` |
| BM25 | `rank_bm25` with Lucene IDF over the eligible corpus only (same repo, strictly earlier, not self); statistics rebuilt per query | `src/knotify/retrieval.py` |
| Serving + UI | `knotify.runtime` loads frozen history, masked model and BM25; UI analyses an imported issue number or a pasted report (cutoff = now) | `ui/app.py` |
| Duplicate evidence | 60 candidates since 2022-11-30, all comments checked → 52 valid earlier-issue targets | `data/duplicates/scikit-learn__scikit-learn/…/pairs.json` |
| Label audit sheet | 40 Train examples (10/class, spread over time); judgement columns blank for a human | `data/annotations/scikit-learn__scikit-learn/…/label_audit_sheet.csv` |

Serving model choice: the template-masked model is served, because the raw model can exploit
template cues. Its 0.018 lower validation macro-F1 is reported, not hidden. Final
selection between TF-IDF and embeddings happens on V-policy (Day 3).

Observed latency on the 2-core / 2 GB Linux sandbox: runtime load ≈ 13 s, first query ≈ 12 s
(tokenising 12k documents once), later queries ≈ 2 s. Not yet measured on the target MacBook.

## In progress / known gaps

- **Retrieval evaluation set is frozen but not yet scored:** 52 explicit, earlier duplicate
  targets (50 stated by repository members) → 5 debug + 47 evaluation queries
  (`data/annotations/…/retrieval_queries.json`; 33 validation-period, 14 test-period). BM25 has
  only been run on the 5 debug queries, as a mechanics check (targets found at ranks 2 and 12; 1/5
  within the top 5). Debug results are not reported. Scoring BM25/dense/hybrid on the 47 waits
  until all three exist (Day 4). Pooled `related` judgements are still to be done. GraphQL
  `MarkedAsDuplicateEvent` evidence (needs a token; `knotify.ingest.marked_duplicates`, not wired
  into the CLI) was not collected.
- **Human label audit (done 23 Nov 2026, one reviewer, AGW, no AI suggestions):**
  `data/annotations/…/label_audit_sheet.xlsx`. Of 40 Train examples, 28 agree (70%), 3 disagree
  (7.5%) and 9 are unclear (22.5%). By class: bug 7/1/2 (yes/no/unclear), feature 9/1/0,
  question_documentation 10/0/0, other 2/1/7. Disagreements: #21490 `Bug` → question_documentation
  (title starts "[DOC]"), #15182 `Enhancement` → bug, #21395 `Build / CI` → bug. Reading: bug,
  feature and question_documentation labels are mostly confirmed (26/30 agree). `other` could not
  be confirmed by this reviewer (7/10 unclear), so it is reported as the least reliable class.
  The mapping is unchanged, because unclear judgements are not evidence that it is wrong. This is a
  single-reviewer check, not an inter-annotator agreement study.
- **LLM comparison:** deferred by the user (optional); report it as not completed unless revisited.
- Not started: calibration/threshold, embeddings + dense/RRF, checklist, evidence templates,
  LLM comparison, SHAP, evaluation runner, transfer scoring, review controls, UI polish.
- The cloud workspace cannot reach api.github.com (proxy requires a linked account); collection
  runs on the local machine. Background processes do not survive between sandboxed calls, so
  long jobs are resumable (`audit`, `import`, `duplicates` each cache progress and exit with
  code 3 when paused; rerun the same command).

## Next concrete step

Day 2: cache MiniLM embeddings, dense retrieval with eligibility-first
filtering, RRF, embedding LR, checklist and template questions.

## Run commands

```sh
uv sync --locked --python 3.12 --extra ml
uv run knotify audit                                   # read-only; resumable
uv run knotify import scikit-learn/scikit-learn --start 2010-01-01 --end 2026-11-21
uv run knotify import pandas-dev/pandas --start 2024-09-11 --end 2026-11-21
uv run knotify split scikit-learn/scikit-learn         # refuses to overwrite a frozen split
uv run knotify train scikit-learn/scikit-learn         # validation (V-policy) only
uv run knotify duplicates scikit-learn/scikit-learn --created-from 2022-11-30
uv run knotify label-sheet scikit-learn/scikit-learn
uv run streamlit run ui/app.py
uv run pytest
```

Commands that page through GitHub (`audit`, `import`, `duplicates`) exit with code 3 when the time
budget or rate limit pauses them; rerun until they finish. A `GITHUB_TOKEN` is optional (read-only,
public repositories; put it in the gitignored `.env`). Re-importing creates a new timestamped
snapshot and never overwrites an existing one.

## Verification (22 Nov 2026, Linux sandbox, Python 3.12.15)

- `ruff check` / `ruff format --check`: pass. `pytest`: 32 passed (UI tests use an empty temporary
  workspace, so they never read local data).
- Real-data smoke test: Streamlit AppTest opened issue #21291 → `feature` (0.71 raw, deferred) and five
  earlier BM25 candidates, with #20137 (its maintainer-confirmed duplicate) ranked first. This is
  an anecdote, not an evaluation.
- Not exercised: the target MacBook, Windows after these changes, embeddings/FAISS/SHAP extras.
