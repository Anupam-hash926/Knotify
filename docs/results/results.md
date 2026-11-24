# Knotify — evaluation results

Generated 2026-11-24 00:14 UTC from the Day 3 freeze (`docs/freeze/day3_freeze.json`); copied to `docs/results/` for the repository. Test split read only by `knotify.evaluation`.

## 1. Category classification — scikit-learn Test (n = 654)

Support: bug 298, feature 140, question_documentation 161, other 55

| Model | Macro-F1 | Accuracy | bug | feature | question/docs | other |
|---|---:|---:|---:|---:|---:|---:|
| Majority class | 0.088 | 0.214 | 0.000 | 0.353 | 0.000 | 0.000 |
| TF-IDF LR, template cues masked (**served**) | 0.785 | 0.820 | 0.867 | 0.820 | 0.797 | 0.656 |
| TF-IDF LR, unmasked text | 0.845 | 0.876 | 0.893 | 0.903 | 0.897 | 0.688 |
| MiniLM embedding LR | 0.740 | 0.763 | 0.810 | 0.782 | 0.724 | 0.645 |

Paired bootstrap (500 resamples, seed 42), macro-F1 differences: masked TF-IDF − embedding +0.045 (95% CI +0.012 to +0.078); unmasked − masked +0.060 (95% CI +0.036 to +0.088).

### Calibration and deferral (served model)

Log loss raw 0.715 → calibrated 0.552 (embedding LR 0.733 → 0.601).

| Calibrated top-probability bin | Count | Mean confidence | Accuracy |
|---|---:|---:|---:|
| 0.0–0.2 | 0 | — | — |
| 0.2–0.4 | 12 | 0.35 | 0.58 |
| 0.4–0.6 | 128 | 0.51 | 0.66 |
| 0.6–0.8 | 199 | 0.71 | 0.88 |
| 0.8–1.0 | 315 | 0.89 | 0.88 |

| Threshold | Accepted | Coverage | Accepted accuracy | Accepted macro-F1 |
|---:|---:|---:|---:|---:|
| 0.50 | 593 | 90.7% | 85.3% | 0.810 |
| 0.60 | 514 | 78.6% | 87.9% | 0.831 |
| 0.70 | 427 | 65.3% | 87.8% | 0.817 |
| 0.80 | 315 | 48.2% | 87.6% | 0.805 |
| 0.90 | 139 | 21.3% | 94.2% | 0.439 |
| 0.95 (**frozen**) | 31 | 4.7% | 96.8% | 0.246 |

The threshold was chosen on V-policy before Test was read (rule: greatest coverage with ≥ 90% accepted accuracy and ≥ 20 accepted). Test rows illustrate behaviour and were not used to pick it. Coverage at a fixed F1 is omitted (not pre-registered).

## 2. Transfer — frozen scikit-learn model on pandas (n = 1272)

Support: bug 777, feature 192, question_documentation 221, other 82. Window 2024-09-11 → 2026-11-21 (after the source's validation cutoff).

| Model | Macro-F1 | Accuracy | bug | feature | question/docs | other |
|---|---:|---:|---:|---:|---:|---:|
| Served (masked) | 0.724 | 0.796 | 0.869 | 0.655 | 0.758 | 0.615 |
| Unmasked text | 0.807 | 0.866 | 0.917 | 0.810 | 0.791 | 0.709 |

At the frozen threshold 0.95: 0 accepted (coverage 0.0%; accepted accuracy —). Calibration and threshold were fitted on the source repository; their reliability on the target is not verified. B labels are used for scoring only.

## 3. Related-issue retrieval — 47 frozen queries

Queries: 33 validation-period, 14 test-period. Relevance: explicit duplicate statements (one known target per query). Dense backend: faiss.

| Method | Recall@5 = Hit@5 (95% CI) | MRR@5 (95% CI) |
|---|---|---|
| bm25 | 0.362 (0.234–0.489) | 0.214 (0.129–0.307) |
| dense | 0.447 (0.298–0.596) | 0.303 (0.180–0.418) |
| hybrid | 0.447 (0.298–0.574) | 0.275 (0.173–0.365) |

Paired bootstrap MRR@5 differences:
- hybrid minus bm25: +0.061 (95% CI -0.020 to +0.138)
- hybrid minus dense: -0.028 (95% CI -0.114 to +0.043)
- dense minus bm25: +0.089 (95% CI -0.028 to +0.213)

Recall against known explicit duplicates only, not an exhaustive relevant set. MRR is truncated at 5. Hit@5 equals Recall@5 here because each query has one target.

## Classical vs selected pipeline (same examples)

| Component | Classical (TF-IDF + BM25) | Selected pipeline |
|---|---|---|
| Category (Test, n = 654) | always answers: coverage 100%, accuracy 82.0% | calibrated + deferral: coverage 4.7%, accepted accuracy 96.8%; rest deferred to a maintainer |
| Related issues (47 queries) | BM25 Recall@5 0.362, MRR@5 0.214 | hybrid Recall@5 0.447, MRR@5 0.275 |
| Missing information | none | rule checklist: missing-item F1 0.789 on human-confirmed bug reports (sections 4–5) |

Both pipelines use the same TF-IDF classifier (selected on validation), so the category difference is calibration and deferral, not a different model.

## 4. Missing-information checklist

Human gold: 30 bug-labelled reports judged by AGW; **9 of 30 were judged not to be bug reports** despite the `Bug` label and are excluded. Rules were frozen (checklist-v1) before judging; rule outputs were hidden from the sheet.

Missing = positive class on 110 applicable items from 21 bug reports: **precision 0.833, recall 0.750, F1 0.789** (TP 15, FP 3, FN 5, TN 87).

| Item | Truly missing | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| reproduction_steps | 1 | — | 0.000 | — |
| expected_behavior | 5 | 0.800 | 0.800 | 0.800 |
| actual_behavior | 0 | 0.000 | — | — |
| environment | 7 | 1.000 | 0.714 | 0.833 |
| version | 5 | 0.800 | 0.800 | 0.800 |
| error_logs | 2 | 1.000 | 1.000 | 1.000 |

Excluded from binary scoring: gold uncertain 11, gold not-applicable 5. Rule predicted `uncertain` on 5 scored items (counted as not detecting missing). Per-item counts are small; rely on the overall figure. The LLM-only checklist comparison was not run (deferred).

## 5. Evidence-linked suggestions

Mechanical checks on 56 template questions from 30 sampled bug reports: invalid references 0, retained 56, reports with no question 12.

Manual check of all 56 template questions (18 issues). Column F ("is the cited item really missing?") was derived from the same reviewer's checklist judgements; column G ("appropriate to ask?") was judged directly.

| Subset | Questions | Supported | Unsupported | Unclear | Appropriate |
|---|---:|---:|---:|---:|---:|
| Human-confirmed bug reports (9 issues) | 21 | 15 | 3 | 3 | 13 |
| Issues judged not bug reports (9) | 35 | — | — | 35 | 0 |
| All | 56 | 15 | 3 | 38 | 13 |

Unsupported-suggestion rate: **0/56** by the proposal's definition (a suggestion citing no retrieved issue or checklist item); **3/56** manually unsupported (cited item actually present). On confirmed bug reports, 13/21 questions were judged appropriate. On non-bug issues 0/35: the label error, not the rules, drives most inappropriate questions. Only 18 of 30 sampled issues received questions (plan target 20). The LLM-drafting grounding ablation was not run (deferred).


## 6. Latency (warm `analyze()`, no LLM calls)

| Machine | Runs | Median | p90 | One-time load | Retriever |
|---|---:|---:|---:|---:|---|
| Linux-6.8.0-149-generic-x86_64-with-glib (2 CPU) | 12 | 2.74 s | 6.03 s | 12.7 s | HybridRetriever |
| Windows-11-10.0.26300-SP0 (8 CPU) | 36 | 2.88 s | 4.81 s | 9.8 s | HybridRetriever |

## 7. Not completed

- LLM-only comparison and LLM drafting/grounding ablation (deferred by the team).
- Pooled 'related' relevance judgements (optional retrieval sheet skipped); retrieval uses explicit-duplicate relevance only.
- Target MacBook not tested.
