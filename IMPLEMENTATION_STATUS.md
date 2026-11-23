# Implementation status

## Current milestone

**Day 2 gate met (23 Nov 2026):** a real imported issue produces every core output without an
LLM: TF-IDF category suggestion (raw, uncalibrated, always deferred), five earlier related issues
from hybrid BM25 + embedding search (RRF), a six-item missing-information checklist and
evidence-linked template questions shown as an editable draft reply. Both classifier variants
run, and historical filtering passes tests. Day 3 (calibration, threshold, explanations, review
persistence) has not started. Day 1 gate met on 22 Nov 2026.

### Day 2 results

| Item | Result | Where |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, downloaded once by the user (Hugging Face is blocked from the assistant's environments) | `models/all-MiniLM-L6-v2/` (gitignored) |
| Embedding cache | 12,387 × 384 float32, L2-normalised, template-masked text; encoded in 326 s on 2 vCPU (CPU). **51% of issues exceed the 256-token limit** (median 263 tokens), so their embeddings cover only the beginning of the issue. Recorded, not worked around (no chunking, per plan) | `artifacts/scikit-learn__scikit-learn/embeddings/<snapshot>/` (`meta.json`) |
| Embedding LR (V-policy, Test untouched) | macro-F1 **0.740** (bug 0.801, feature 0.828, question_documentation 0.672, other 0.657) vs masked TF-IDF **0.809**. Embeddings do not win; the TF-IDF serving model stays. Formal selection is on Day 3 | `runs/baseline/…/baseline_validation.json`, `artifacts/…/embedding_lr.joblib` |
| Dense search | Eligible rows are selected *before* ranking; exact FAISS `IndexFlatIP` with an ID selector (FAISS 1.16 in the cloud run), exact NumPy fallback (used where FAISS is not installed). Both give the same ranking in tests | `retrieval.DenseRetriever` |
| Hybrid | RRF `1/(60+rank)`, 50 candidates per method, issue-ID tie-break. Pasted text without the local encoder falls back to BM25 and the UI says so | `retrieval.HybridRetriever` |
| Checklist | `checklist-v1`: reproduction, expected, actual, environment, version, error logs (conditional). Empty template headings and `_No response_` count as missing. Rules developed on 10 Train bugs (dev only) | `checklist.py` |
| Template questions | One question per missing item, citing `checklist:<item>`. `reference_valid` is checked mechanically; `semantic_support` stays `not_checked`. Conditional wording while the category is deferred | `drafting.py` |
| Debug-only retrieval check | On the 5 dev queries, dense found 3 targets in its top 50 that BM25 missed. **Not reported; the 47 evaluation queries are untouched** | — |

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
- Not started (Day 3+): calibration/threshold and classifier selection, TF-IDF SHAP, review
  persistence/controls in the UI, checklist evaluation sheet (30–40 human-checked bug reports),
  20-suggestion manual check, evaluation runner, transfer scoring, UI polish.
- The cloud workspace cannot reach api.github.com (proxy requires a linked account); collection
  runs on the local machine. Background processes do not survive between sandboxed calls, so
  long jobs are resumable (`audit`, `import`, `duplicates` each cache progress and exit with
  code 3 when paused; rerun the same command).

## Decisions confirmed by the user (23 Nov 2026)

- Exclude issues still labelled `Needs Triage` from supervised data (reported deviation).
- Mask template cues in classifier input; serve the masked model and report the raw variant.
- Retrieval evaluation uses all 47 frozen queries (33 validation-period, 14 test-period),
  disclosed in the report. Retrieval methods are not tuned on them.
- Stick to the final plan: frozen snapshots only, no live GitHub fetching in the app.
- LLM comparison deferred (optional).

## Next concrete step

Day 3: fit sigmoid calibration on V-cal for both classifiers, select the serving classifier
and threshold on V-policy, add TF-IDF SHAP (or labelled linear contributions), persist
reports/reviews from the UI, prepare the checklist and suggestion review sheets, then freeze all
configurations.

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
# Day 2 (needs the embeddings extra and the local model folder, see README)
uv sync --locked --python 3.12 --extra ml --extra embeddings --extra faiss
uv run knotify embed scikit-learn/scikit-learn          # ~5 min on 2 CPU cores; refuses to overwrite
uv run knotify train scikit-learn/scikit-learn --model embedding
uv run streamlit run ui/app.py
uv run pytest
```

Commands that page through GitHub (`audit`, `import`, `duplicates`) exit with code 3 when the time
budget or rate limit pauses them; rerun until they finish. A `GITHUB_TOKEN` is optional (read-only,
public repositories; put it in the gitignored `.env`). Re-importing creates a new timestamped
snapshot and never overwrites an existing one.

## Verification (22 Nov 2026, Linux sandbox, Python 3.12.15)

- `ruff check` / `ruff format --check`: pass. `pytest`: 41 passed (UI tests use an empty temporary
  workspace, so they never read local data).
- Real-data smoke test: Streamlit AppTest opened issue #21291 → `feature` (0.71 raw, deferred) and five
  earlier BM25 candidates, with #20137 (its maintainer-confirmed duplicate) ranked first. This is
  an anecdote, not an evaluation.
- Not exercised: the target MacBook, Windows after these changes, embeddings/FAISS/SHAP extras.
