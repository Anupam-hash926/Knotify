# Knotify — Project Context

This is a factual handoff document, **not an agent instruction file**. It describes the submitted
proposal, the decisions made afterward, and the scaffold currently present. It is not an
`AGENTS.md` or `CLAUDE.md`, and it does not establish a second implementation policy.

## Project and source documents

Knotify is an academic AIML mini-project at Ramaiah Institute of Technology, Department of Computer
Science and Engineering. The exact submitted title is **“Knotify: A Confidence-Aware, Evidence-Linked
Triage Pipeline for GitHub Issues”**. The faculty guide is **Prof. Uzma Sulthana**.

The repository contains:

- `Knotify_AIML_Project_Proposal (1) (1).pdf`: the submitted, unchanged two-page proposal.
- `KNOTIFY_FINAL_PLAN.md`: the finalized five-day implementation/evaluation plan, including later scope decisions.
- This `context.md`: background and technical description for continuation in Claude Code or Codex.
- `IMPLEMENTATION_STATUS.md`: observed scaffold completion, verification and outstanding product work.

The user confirmed the proposal has already been submitted and cannot be changed. Modest
implementation deviations are acceptable to the user when documented. The approval/review field
printed in the PDF is a placeholder; no separate faculty-review document is available in this
checkout. Submission should not be confused with independently verified faculty approval.

## Exact proposal details in compact form

The PDF lists Anupam A Nair (1MS24CS033), Abhay G Warrier (1MS24CS005), and Chinmay Panghri
(1MS24CS006). Its original responsibility assignments and dates are preserved in the verbatim
appendix below as historical proposal content. **They are not the current build workflow.** The user
will use coding assistants and explicitly rejected dividing implementation into teammate workstreams.

The proposal describes manual categorization, duplicate checking and requests for missing logs or
reproduction steps as repetitive maintainer work. Maintainers retain final decisions.

Its three objectives are: compare TF-IDF/class-weighted logistic regression with sentence-embedding
logistic regression (DistilBERT optional); compare BM25, embedding/FAISS and rank-fused retrieval;
and compare the full pipeline against classical and LLM-only baselines through ablations.

The proposed dataset is 2,000–4,000 filtered public issues from two or three open-source repositories.
The four targets are bug, feature, question/documentation and other. PRs and unlabelled issues are
excluded from supervised data. Repository labels are audited; author identity is excluded from features.

The proposal combines Markdown cleaning preserving technical tokens, chronological splits,
duplicate grouping, Platt calibration, abstention, SHAP, hybrid retrieval through RRF and a rule
checklist. Its reported metrics include macro/per-class F1, calibration, accuracy/F1 versus coverage,
coverage at fixed F1, Recall@5, MRR, missing-information precision/recall/F1, unsupported-suggestion
rate and median latency. It proposes about 100 manually checked retrieval cases, bootstrap
intervals and a train-on-one/test-on-another transfer check.

The expected UI shows category/confidence or deferral, SHAP tokens, five related earlier issues,
missing information and a draft question. The Google sign-in crash example in the PDF is explicitly
illustrative, not a measured result. Nothing is posted automatically.

The original schedule was November 10, November 24 and December 8, 2026. Those dates have been
superseded for implementation by the user's five-day deadline. The original stack already includes
Streamlit and says CPU compute suffices; Colab is mentioned only for optional DistilBERT.

The appendix reproduces all extracted proposal text, including references, identifiers, original
assignments and schedule. PDF layout is not reproduced; the PDF remains authoritative for layout.

## Finalized approach and practical simplifications

The final plan is a CPU-first local Python application with Streamlit as the browser GUI. There is
no React frontend, FastAPI service, PostgreSQL deployment, queue, multi-agent product or automatic
GitHub write. Clustering, emerging trends and transformer fine-tuning are outside the five-day scope.

The target execution machine is a **16 GB RAM MacBook with a reported 10-core GPU**. The exact chip
has not been supplied. Development/scaffold checks here run on Windows. CPU-only is the agreed
initial implementation. Apple MPS/GPU support is neither implemented nor required; it can be
considered only if measured CPU embedding performance is insufficient. Historical embeddings and
trained models are cached, not regenerated on every Streamlit rerun. No local generative LLM is planned.

The five retained architectural decisions are:

1. Embedding classification is a comparison, not a predetermined winner. TF-IDF classification plus
   hybrid retrieval is a valid final configuration.
2. Calibration/confidence applies only to the category prediction, not the report or retrieval score.
3. TF-IDF SHAP/linear contributions explain that specific model; embedding coordinates are not word explanations.
4. Citation validity is distinct from semantic support and correctness. A small manual audit supplements reference checks.
5. Streamlit is the delivery interface; full web infrastructure and repository analytics are deferred.

The practical evaluation uses repository A with chronological 60/20/20 Train/Validation/Test and B
for a small transfer check. Validation is divided into calibration/policy portions when feasible;
reuse of Validation for both is an explicitly disclosed fallback for small data. Model fitting and
calibration remain disjoint, and Test is not used for selection. The fixed classifier starts with
`C=1`, balanced class weights and a seed. Policy thresholds range over a small declared grid and
target 90% accepted validation accuracy with at least 20 accepted examples; unmet policy means
defer all, not an invented guarantee.

Current GitHub snapshots are frozen and ordered by creation time. Historical text edits are not
reconstructed; results are retrospective, not a faithful deployment replay. Same-repository,
strictly earlier candidates are selected before BM25 statistics and dense ranking. Retrieval uses
up to 50 candidates per method, RRF constant 60 and up to five displayed candidates. Similarity does
not establish duplicate status. Evaluation reports Recall@5, Hit@5 and explicitly truncated MRR@5.

Manual workload was reduced to roughly 40 label-audit examples, 30–50 retrieval queries, 30–40
checklist reports, 20 suggestion checks and 20–30 LLM-only comparison examples where possible.
These are planned sample sizes, not completed annotations. Model-generated labels are not
independent human ground truth. Optional bootstrap/SHAP/LLM shortfalls must be named if unfinished.

The build sequence is: first real TF-IDF/BM25 path; complete embedding/checklist report; add
calibration/evidence/review; evaluate; package and verify. Five-day checkpoints have no named owners
and do not require multiple coding agents. Detailed gates and fallback budgets are in the final plan.

## UI preference

The user explicitly wants the final UI to be **aesthetic, clean and sleek**. The current shell starts
with a warm off-white background, restrained green accent, generous spacing, simple typography,
sidebar navigation and two-column issue/report workspace. The desired final UI feels coherent and
polished rather than like a raw notebook or a dense collection of charts. Evidence, uncertainty,
loading/empty/error states and primary actions remain easy to understand.

The existing appearance is a starting point, not a finalized product design. Scaffold-only status
messaging is deliberate now; the eventual product flow should focus on the issue and its useful
results, without exposing irrelevant implementation detail. No fake confidence, hardcoded metrics
or made-up historical issues are present in the UI.

## Technical scaffold currently present

### Packaging and configuration

The repository uses a `src/` Python package, `pyproject.toml`, an executable `knotify` CLI and a
committed `uv.lock`. `.python-version` selects 3.12. `uv sync --locked --python 3.12` creates an
editable environment. Base dependencies are Streamlit, Pydantic and python-dotenv; development tools
are pytest and Ruff. Optional extras separate baseline ML/importing (`ml`), sentence encoders
(`embeddings`), native CPU search (`faiss`) and explanations (`explain`). Resolving extras in the
lock is not proof that they have been executed on the target Mac.

`configs/default.toml` contains taxonomy, split ratios, seed, policy grid, encoder identity,
CPU device, batch size, retrieval depth/RRF settings and disabled optional LLM configuration.
`config.py` validates these and resolves data/artifact/run paths relative to the editable repository.
The CPU field intentionally accepts only `cpu` at this stage. No GPU auto-detection is implemented.
`configs/repositories.example.toml` is a schema illustration, not audited repository selection.

`.env.example` contains blank credential placeholders; `.env` and `.streamlit/secrets.toml` are
ignored. The CLI explicitly loads `.env`; no external integration currently consumes a token.
The application makes no API calls, downloads no models and performs no training at startup.

### Domain records

`contracts.py` defines strict Pydantic records with forbidden extra fields:

- `Issue`: stable ID, repository, number, original text, timezone-aware timestamps, source URL,
  source labels, optional mapped target and imported/pasted origin. Imported URLs must match the
  GitHub repository and issue number. Pasted input needs no GitHub URL.
- `ClassificationResult`: optional four-class raw/calibrated distributions, predicted category,
  deferral/reason and model/calibration provenance. An accepted category requires calibrated
  probabilities. Missing predictions are null; probability distributions must sum to one.
- `RetrievedIssue`: source issue, rank, method scores and related-candidate relationship.
- `ChecklistItem`: one of six item IDs, present/missing/uncertain/not-applicable, explanation and
  optional matched text/rule version.
- `Suggestion`: text, evidence IDs, reference validity, source and semantic-support status.
- `AnalysisReport`: UUID, input/config hashes, original issue, component results/statuses and timing.
- `Review`: UUID, saved analysis reference, decision, optional correction/note and timestamp.

The report schema is version 1. `knotify schema` prints its JSON Schema. These contracts are a
starting API between local modules, not a remote HTTP API. They can evolve with explicit schema
changes when real implementation reveals a need; no rigid database schema has been introduced.

### Module boundaries and implemented helpers

| File | Present behaviour | Still absent |
|---|---|---|
| `data.py` | Title/body-only feature text and stable hashes | Label audit/mapping, preprocessing refinements, split generation |
| `ingest.py` | Reserved import function that raises `ComponentUnavailable` | GitHub client, pagination, PR filtering, frozen dataset export |
| `classifier.py` | `Classifier` protocol and explicit unavailable adapter | TF-IDF/embedding training, calibration, artifact loader, SHAP |
| `retrieval.py` | Same-repository earlier-only filtering, deduplication/conflict check, `Retriever` protocol | BM25, embedding cache, FAISS/NumPy search, RRF |
| `checklist.py` | Checklist protocol and unavailable adapter | Rules, applicability and completeness evaluation |
| `llm.py` | Optional drafting protocol | Provider selection, prompts, API client and response validation |
| `triage.py` | `ModelBundle` injection and shared `analyze()` orchestration | Question templates, evidence registry/validator, LLM wiring |
| `evaluate.py` | Recall/hit/reciprocal-rank and selective-accuracy primitives | Dataset-level experiment runner, classification metrics/plots, transfer and ablations |
| `storage.py` | Append-only analysis/review JSON files; reviews require saved reports | UI save/review integration and richer dataset manifests |
| `cli.py` | `doctor`, `schema`, version/help | Import/train/calibrate/evaluate actions; reserved names exit 2 |
| `ui/app.py` | Input form, unavailable report preview, JSON download and readiness view | Imported-issue selection, real model results, review controls and final visual polish |

`analyze(issue, repository_history, model_bundle, config, dataset_version=...)` is the shared
orchestrator. Default adapters report unavailable rather than return simulated results. A component
failure is isolated, allowing other components to proceed. Retrieved outputs are rechecked against
eligible immutable input records so a buggy retriever cannot expose future/foreign/self matches.
The classifier receives only the title/body view. The checklist can see classification uncertainty.
No suggestions are produced while checklist/drafting logic is absent.

`data/`, `artifacts/` and `runs/` contain tracked README placeholders; actual local contents are
ignored. JSON files are named using validated UUIDs and saved with exclusive creation, so existing
analyses are not overwritten. This is deliberately simple local persistence, not a concurrent or
crash-transactional production datastore. Model files are not yet loaded or deserialized anywhere.

### Verification and continuation state

Tests exercise invalid configurations, probability/source validation, feature leakage, temporal
eligibility, conflicting snapshots, metrics, component failure isolation, report/review persistence,
CLI status and Streamlit interaction. `.github/workflows/ci.yml` defines default-environment checks
on Linux, Windows and macOS. The initial local verification results are recorded in
`IMPLEMENTATION_STATUS.md`; CI configuration alone is not evidence of a passing hosted run.

The initial scaffold does not complete Day 1 of the product plan: there is no real imported dataset,
trained classifier, retrieval implementation, checklist, annotation set or measured model result.
The next integration milestone remains a real imported issue displayed with TF-IDF and BM25.

The user specifically raised dataset readiness during scaffold creation. Data will come from
existing public open-source GitHub repositories, not the new Knotify repository. No actual source
repositories have been selected or audited yet. Dataset feasibility (class counts, temporal support
and explicit duplicate/related targets) is therefore the first implementation gate, before model
tuning. The audit will distinguish classification training labels, retrieval relevance pairs and
small manually reviewed checklist samples. The 2,000–4,000 issue target is not a claim of existing data.

The GitHub destination authorized by the user is `https://github.com/Anupam-hash926/Knotify`, branch
`main`. The remote was empty when inspected before scaffold creation. The user authorized pushing
the scaffold, proposal, final plan and this context together. This contextual fact is not blanket
authorization for unrelated future repository actions.

## Appendix — complete submitted proposal text

The following is extracted from the supplied PDF, with page boundaries retained. PDF text extraction
changes line wrapping and table layout; no proposal wording has been intentionally rewritten here.

### Proposal page 1

```text
RAMAIAH INSTITUTE OF TECHNOLOGY
Department of Computer Science and Engineering
AIML Mini Project Plan
Project title: Knotify: A Confidence-Aware, Evidence-Linked Triage Pipeline for GitHub Issues
Faculty guide: Prof.Uzma Sulthana
Team member details:
Student name
USN
Main responsibility
Anupam A Nair
1MS24CS033
Dataset collection, label mapping,
classifiers
Abhay G Warrier
1MS24CS005
Retrieval, hybrid search and triage logic
Chinmay Panghri
1MS24CS006
Evaluation, explainability and interface
1. Problem statement
Maintainers of open-source repositories must label each new issue, check it against earlier reports and often ask the reporter for missing
details such as logs or reproduction steps. This manual triage is slow, labels are applied inconsistently, and duplicate reports waste effort. We
study whether a trained classifier, hybrid retrieval of similar past issues and a confidence-based deferral rule can support maintainers, who
keep the final decision.
2. Project objectives
1. Develop and compare classifiers (class-weighted TF-IDF + Logistic Regression, sentence-embedding + Logistic Regression; fine-tuned
DistilBERT optional), reporting macro-F1 and per-class F1 on a time-based test set, calibration and accuracy-vs-coverage curve for
abstention.
2. Develop and compare BM25, embedding (FAISS) and hybrid rank-fusion retrieval of earlier similar issues, evaluated with Recall@5 and MRR
against duplicate-linked issues and a manually checked sample.
3. Evaluate the full pipeline against a classical baseline (TF-IDF + BM25) and an LLM-only variant through ablations, measuring
missing-information F1 (rule checklist vs LLM baseline), unsupported-suggestion rate and latency.
3. Existing work and research gap
GitHub's AI issue-intake action suggests marking an issue actionable or requesting more information [1]. Many classification studies, surveyed
across 46 papers, target bug versus non-bug and call for real-world evaluation and checks on the reliability of ground-truth labels [2]. We
therefore measure label noise, abstention and cross-repository transfer in one pipeline.
4. Proposed novelty
Beyond a basic classifier, we add calibrated confidence with abstention, hybrid BM25 + embedding retrieval and a missing-information
checklist. Baseline: TF-IDF + Logistic Regression with BM25 search, plus an LLM-only variant. Ablations on identical held-out issues use
macro-F1, coverage at fixed F1, Recall@5 and unsupported-suggestion rate to show which component helps.
5. Dataset and feasibility
Public issues from two or three open-source GitHub repositories [3], about 2,000-4,000 after filtering, collected with a personal access token
(public data, no special permission). Inputs: title, body, labels, dates. Targets: bug, feature, question/documentation, other, mapped from
repository labels. Pull requests and unlabelled issues are excluded; repositories are fixed after a label audit.
6. Methodology and ML pipeline
Clean Markdown while keeping code tokens; map labels to four classes. Baseline: class-weighted TF-IDF + Logistic Regression with BM25
retrieval. Proposed: sentence-embedding classifier with Platt calibration, SHAP attributions and abstention below a threshold; hybrid BM25 +
FAISS retrieval via Reciprocal Rank Fusion; rule-based checklist. Split by time into train, validation and test; group duplicates for evaluation;
the retrieval index holds only earlier issues.
AIML Project Proposal  |  Page 1
```

### Proposal page 2

```text
RAMAIAH INSTITUTE OF TECHNOLOGY
Department of Computer Science and Engineering

7. Expected output and sample result
The maintainer sees a predicted category with a confidence score and SHAP tokens (or a defer-to-human flag), five related earlier issues,
missing details and a drafted follow-up question. Nothing is posted automatically.
Sample input: App crashes whenever I log in with Google sign-in.
Expected output: Illustrative, not a measured result: category bug (with confidence); similar earlier issues listed; missing: OS, app version,
error log; suggested follow-up question.
8. Evaluation plan
Metrics: macro-F1 and per-class F1; F1 vs coverage across confidence thresholds; Recall@5 and MRR; missing-information precision, recall
and F1; unsupported-suggestion rate (a suggestion citing no retrieved issue or checklist item); median latency.
Baseline comparison: TF-IDF + Logistic Regression with BM25, and an LLM-only variant, against the proposed variants, using ablations
(embeddings, hybrid retrieval, abstention, checklist).
Test setup: time-based held-out test set; retrieval relevance from duplicate-linked issues plus about 100 manually checked cases; bootstrap
confidence intervals; train-on-one-repository, test-on-another transfer check.
9. Tools resources and responsible use
Tools and hardware: Python, pandas, scikit-learn, rank_bm25, sentence-transformers, FAISS, SHAP, Streamlit. A CPU suffices; free Colab GPU
only if DistilBERT is attempted. An optional LLM drafts follow-up questions.
Risk and safeguards: repository labels may be noisy, so we document the label mapping, audit a sample and report class-wise metrics. Public
data only; author identity excluded from features; maintainers approve every action; nothing is auto-posted.
Syllabus concepts: Unit II uncertainty and probabilities (calibration, abstention); Unit III classification, validation, metrics; Unit IV similarity
measures; Unit V embeddings, vector databases (FAISS), retrieval, SHAP, responsible and human-in-the-loop AI.
10. Team schedule
Milestone
Owner
Target date
Data and baseline ready
Anupam A Nair
10 Nov 2026
Proposed retrieval and triage logic ready
Abhay G Warrier
24 Nov 2026
Evaluation, explainability, UI demo ready
Chinmay Panghri
8 Dec 2026
11. References
[1] GitHub Docs, "Triaging an issue with AI", accessed Oct 2026.
https://docs.github.com/en/issues/tracking-your-work-with-issues/administering-issues/triaging-an-issue-with-ai
[2] M. Laiq and F. Dobslaw, "Automatic techniques for issue report classification: A systematic mapping study", Automated Software Engineering, vol. 33, 2026.
https://doi.org/10.1007/s10515-026-00616-x
[3] GitHub Docs, "REST API endpoints for issues", accessed Oct 2026. https://docs.github.com/en/rest/issues/issues
Team confirmation: We have identified the data and resources needed and will report only measured results.
​
Team representative: Anupam A Nair                                                   Faculty review: [Approved / Revise / Pending]
AIML Project Proposal  |  Page 2
```
