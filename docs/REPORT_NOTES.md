# Report notes: deviations, limitations and what was not completed

Material for the project report and viva. Numbers come from `docs/results/results.md`; decisions
and their evidence are in `context.md` ("Verified dataset decision") and `IMPLEMENTATION_STATUS.md`.

## Deviations from the submitted proposal

| Proposal | What was done | Why |
|---|---|---|
| 2,000–4,000 issues from 2–3 repositories | scikit-learn (source): 3,271 supervised issues created since 2019, plus 12,387 issues as search history; pandas (transfer): 1,272 issues | A read-only API audit showed both have all four classes; Streamlit and FastAPI did not |
| Labels mapped from repository labels | Same, plus: issues still marked `Needs Triage` were excluded (577), because issue templates auto-apply type labels before a maintainer reviews them | Label provenance found in the repositories' own template files |
| Classifier input: cleaned Markdown | Template cues masked (title prefixes such as `BUG:`, headings, checkboxes); unmasked variant reported | Templates encode the label; masking makes the task about content |
| DistilBERT (optional) | Not attempted | Out of the five-day scope (plan) |
| Sentence-embedding LR vs TF-IDF LR | Both trained and compared; **TF-IDF selected** on validation and better on Test (+0.045, 95% CI +0.012 to +0.078) | Selection rule fixed in advance |
| About 100 manually checked retrieval cases | 47 queries with **explicit maintainer-stated duplicates**; pooled "related" judgements not done | Time; explicit evidence is stronger than similarity-based pairs |
| MRR | MRR@5 (truncated) plus Recall@5 and Hit@5 | The judged pool covers only the top five |
| Missing-information F1, rule checklist vs LLM | Rule checklist scored against a blind human sheet (21 confirmed bug reports, single reviewer). **LLM baseline not run** | LLM comparison deferred by the team |
| LLM-only pipeline variant, LLM-drafted questions | **Not done**; fixed templates only | Deferred (optional in the plan) |
| Coverage at fixed F1 | Omitted; accuracy/F1 vs coverage curve reported | Not pre-registered |
| Schedule and owners (10 Nov – 8 Dec) | Five-day build with coding assistants; no per-person workstreams | Final plan supersedes the proposal schedule |

Completed as proposed: chronological splits, Platt (sigmoid) calibration, abstention, SHAP,
BM25 / FAISS / RRF hybrid retrieval, rule checklist, unsupported-suggestion rate, latency,
bootstrap intervals, the train-on-one/test-on-another transfer check, and the Streamlit interface.

## Limitations

- **Snapshot, not replay.** Issues were frozen on 22 Nov 2026; titles, bodies and labels may
  include edits made after creation. Results are retrospective.
- **Label noise.** Of 30 sampled `Bug`-labelled issues, a reviewer judged **9 not to be bug
  reports**. In the 40-issue label audit, 70% agreed, 7.5% disagreed and 22.5% were unclear;
  `other` could not be confirmed (7/10 unclear). Single reviewer; no agreement study.
- **Template shortcuts remain partly.** Unmasked text scores +0.060 macro-F1 higher on Test,
  evidence that templates leak the label. SHAP later showed that GitHub's `_No response_`
  placeholder, which masking does not remove, is a top feature for some feature requests (e.g.
  issue 21291). This was found after the freeze and is reported rather than fixed.
- **What drives `bug`.** SHAP's top words for bug predictions are often code tokens (`import`,
  `lib`, `python`): the model largely detects "contains code or paths".
- **Low coverage at the chosen policy.** Requiring ≥ 90% accepted accuracy leaves 4.7% of Test
  issues auto-categorised (96.8% correct). At 0.80, coverage would be 48% at 87.6% accuracy;
  the target was not changed after seeing results.
- **Calibration does not transfer.** No pandas issue passes the scikit-learn-calibrated threshold.
- **Small samples.** 55 `other` issues in Test; 47 retrieval queries (33 from the validation
  period, which is fair because search has no trained parameters); per-checklist-item counts
  are 0–7. Retrieval differences are not statistically conclusive.
- **Embedding truncation.** 51% of issues exceed MiniLM's 256-token limit; embeddings cover only
  the start of those issues.
- **Similarity is not duplication.** Related issues are candidates; only explicit statements
  were used as relevance.
- **Suggestions.** Column F of the suggestion check was derived from the same reviewer's
  checklist judgements. Most inappropriate questions (35/43) came from issues that were not bug
  reports, which traces back to label noise.
- **Near-duplicates.** Only exact normalised duplicate texts were removed.
- **Hardware.** Latency: median 2.88 s on a Windows 11 laptop (8 threads, 36 runs) and 2.74 s on a 2-CPU Linux sandbox. The
  target 16 GB MacBook was not tested.

## Not completed

- LLM-only comparison, LLM drafting and the grounding ablation (deferred).
- Pooled "related" relevance judgements (optional sheet skipped).
- DistilBERT fine-tuning (optional).
- Testing on the target MacBook.
