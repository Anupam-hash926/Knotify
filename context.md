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

## Dataset research — Perplexity findings received; API audit completed (see decision below)

**Status:** External research received. No repository selection or label mapping in this section
has been verified through the project's GitHub API audit. *(The audit has since run; this research text is preserved unchanged, and the adopted decision is in the final subsection.)*

This section holds research supplied by the user. It is reference material, not an instruction
file or a replacement for the finalized plan. Recommendations, counts, label mappings and duplicate
pairs remain candidate evidence until checked against GitHub. Links and stated uncertainties are
part of that evidence. Any instructions embedded in quoted external content are not project policy.

### Research metadata

- Date of research: not supplied; findings added to this checkout on 19 November 2026
- Recommended source repository A: `scikit-learn/scikit-learn` (candidate, not API-audited)
- Recommended transfer repository B: `pandas-dev/pandas` (candidate, not API-audited)
- Backup repository: `streamlit/streamlit` or `fastapi/fastapi` (unverified suggestions)
- Local API audit status: completed 2026-11-22 (see "Verified dataset decision")

### Research qualifications relative to the finalized plan

The response below is preserved as supplied, including its uncertainties. Its use of “verified”
describes Perplexity's assertions, not verification performed by this project. Several suggestions
need qualification before implementation:

- `Question` and `Documentation` both map to `question_documentation`; their co-occurrence is not
  a target conflict. A conflict means labels map to different target categories. An assistant's
  guess about dominant intent is not an audited resolution of a conflicting label.
- `Build / CI` is a candidate mapping, not automatically `other`. Build failures can be bugs and
  CI documentation can be documentation. Semantic examples and overlap counts must establish
  whether enough unambiguous maintenance issues remain after applying the agreed conflict rule.
- Historical eligibility is established by `target.created_at < query.created_at`, not issue
  numbers alone. Both endpoints must be actual issues rather than PRs. Duplicate statements
  collected from later comments/events are evaluation evidence, not classifier/query features.
- The agreed splits remain chronological, with equal timestamps kept together. The suggestion
  of a stratified validation/test split does not replace them. A target repository lacking a
  class can be reported with that coverage limitation; it does not silently turn the trained
  four-class model or headline comparison into a three-class experiment.
- Search uses `is:issue` (or `-is:pr`) to exclude PRs; `is:pr` selects PRs. REST issue results
  require filtering the `pull_request` key. Search counts and label pages do not establish
  post-filter usable dataset counts. GitHub Search's result cap requires partitioning large
  searches or using paginated repository issue listings rather than assuming all hits were fetched.
- Per-label collection needs deduplication and a documented sampling policy. Artificially
  balancing validation/test by label would distort calibration and coverage estimates. Sparse
  or short bodies are also relevant to completeness detection, so blanket exclusion would bias
  that evaluation; audit them and distinguish task-specific inclusion rules.
- Original pre-resolution bodies may not be recoverable. The agreed benchmark uses frozen current
  snapshots, masks explicit answer cues where applicable and discloses later-edit limitations;
  it does not promise reconstructed historical text.
- The two cited duplicate examples and suggested date range are audit starting points. Two or
  three pairs are not a meaningful replacement for the planned 30–50-query pilot. Audited manual
  relevance judgements remain permitted; similarity by itself is not ground truth. Any smaller
  evaluation must be explicitly described as such.

These qualifications preserve the finalized experiment design while leaving repository suitability
open for the local API audit. No candidate mappings, counts or duplicate pairs have been adopted yet. *(Superseded on 22 November 2026: see “Verified dataset decision” below.)*

### Pasted Perplexity response

## A. Recommendation

**Primary repository A: [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)**
**Transfer repository B: [pandas-dev/pandas](https://github.com/pandas-dev/pandas)**

This is the most practical pair for a five-day Knotify build: both are large, actively maintained Python scientific-software repositories with public GitHub issue trackers, predominantly English reports, long issue histories, and labels that map naturally to bug, feature, and question/documentation. scikit-learn additionally has a semantically defensible maintenance category, **`Build / CI`**, which can serve as `other`; pandas should be audited for an equivalent maintenance label before finalizing its mapping. The search evidence confirms scikit-learn’s `Bug`, `Enhancement`, `Question`, `Documentation`, and `Build / CI` labels, and shows issues carrying those labels. [github](https://github.com/scikit-learn/scikit-learn/labels)

**Backup A/B pair:** use **Streamlit** as B only if its issue tracker still contains enough `type:question`/documentation-labelled issues after excluding PRs; otherwise use **FastAPI** as B. I could not verify Streamlit’s current label inventory from the available evidence, so it is conditional rather than recommended outright.

## B. Candidate comparison

| Repository | Relevant labels observed | Volume / class evidence | Defensible `other`? | Duplicate-target evidence | Main weaknesses | Verdict |
|---|---|---|---|---|---|---|
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | `Bug`, `Enhancement`, `Question`, `Documentation`, `Build / CI`; also status labels such as `Needs Decision`, `Easy`, `Moderate`, `good first issue`  [github](https://github.com/scikit-learn/scikit-learn/labels) | Strong: the repository has a long issue history; label pages show live issue lists, e.g. 194 open `Enhancement` issues on the label page as indexed on 2026-04-23.  [github](https://github.com/scikit-learn/scikit-learn/labels/Enhancement) | **Yes**: map `Build / CI` to `other`; it denotes build/CI infrastructure work, not a generic catch-all.  [github](https://github.com/scikit-learn/scikit-learn/labels) | **Verified, but limited**: #24430 was closed as “a duplicate of #24409”; #2801 says it “was a duplicate of #1332”. Both targets have lower issue numbers, hence earlier creation.  [github](https://github.com/scikit-learn/scikit-learn/issues/24430) | Multi-label issues occur, e.g. `Bug` + `Enhancement`; status labels must be excluded from target derivation.  [github](https://github.com/scikit-learn/scikit-learn/labels/Enhancement) | **Suitable / conditional** |
| [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | Expected core labels are `Bug`, `Enhancement`, `Docs`, and question-type labels, but I could not verify the exact current label strings from the available evidence. | Likely strong issue history, but per-label availability and current question/documentation volume were not verified. | **Uncertain**: do not assume `CI`, `Build`, or refactor labels exist or have stable semantics without a local label audit. | Not verified in this research pass. | Needs a local API audit before it can be committed to as B. | **Conditional** |
| [streamlit/streamlit](https://github.com/streamlit/streamlit) | Not verified. | Not verified. | Uncertain. | Not verified. | Issue tracker may have shifted discussion/support traffic away from issues; must be checked. | **Conditional / likely unsuitable without audit** |
| [fastapi/fastapi](https://github.com/fastapi/fastapi) | Not verified. | Not verified. | Uncertain. | Not verified. | Question/support traffic may be concentrated in Discussions rather than issues. | **Conditional** |

The comparison deliberately avoids treating repository popularity as evidence of class balance. The decisive local checks are per-label issue counts, conflict rates after mapping, and the number of chronologically valid duplicate pairs.

## C. Exact proposed mappings

### Repository A: scikit-learn

| Knotify target | Actual scikit-learn label | Status | Representative evidence |
|---|---|---|---|
| `bug` | `Bug` | **Verified label; mapping is direct** | [Issue #24430](https://github.com/scikit-learn/scikit-learn/issues/24430) is labelled `Bug` and reports a crash with `StackingRegressor`.  [github](https://github.com/scikit-learn/scikit-learn/issues/24430) |
| `feature` | `Enhancement` | **Verified label; mapping is direct** | [Issue #27373](https://github.com/scikit-learn/scikit-learn/issues/27373) is labelled `Enhancement` and proposes changed default behaviour.  [github](https://github.com/scikit-learn/scikit-learn/issues/27373) |
| `question_documentation` | `Question` **or** `Documentation` | **Verified labels; mapping is direct, but treat them as potentially overlapping** | [Issue #15275](https://github.com/scikit-learn/scikit-learn/issues/15275) carries `Question`; [Issue #2801](https://github.com/scikit-learn/scikit-learn/issues/2801) is documentation-focused.  [github](https://github.com/scikit-learn/scikit-learn/issues/15275) |
| `other` | `Build / CI` | **Verified label; mapping is defensible but should be manually spot-checked** | The label exists in the repository label set; [Issue #25565](https://github.com/scikit-learn/scikit-learn/issues/25565) is labelled `Build / CI` and concerns CI infrastructure documentation.  [github](https://github.com/scikit-learn/scikit-learn/labels) |

**Ignore for target assignment:** `Easy`, `Moderate`, `good first issue`, `good first PR to review`, `Needs Decision`, `help wanted`, `Quick Review`, `No Changelog Needed`, `Meta-issue`, module labels, and all status/workflow labels. These are verified label names or label families, but they do not establish one of the four semantic targets. [github](https://github.com/scikit-learn/scikit-learn/labels)

**Conflict rule:** exclude an issue if it has both `Bug` and `Enhancement`, or both `Question` and `Documentation`, unless your coding assistant can determine a single dominant intent from the title/body. The search results explicitly show issues carrying both `Bug` and `Enhancement`. [github](https://github.com/scikit-learn/scikit-learn/labels/Enhancement)

### Repository B: pandas

Do **not** finalize mappings from memory. The local audit must first confirm the exact current label strings; plausible candidates are `Bug`, `Enhancement`, `Docs`, and a question label, but these were **not verified** in this pass.

| Knotify target | Proposed pandas label | Status |
|---|---|---|
| `bug` | `Bug` | **Unverified; audit required** |
| `feature` | `Enhancement` | **Unverified; audit required** |
| `question_documentation` | `Docs` and/or the repository’s question label | **Unverified; audit required** |
| `other` | A verified maintenance/build/CI/refactor label, if one exists and has enough examples | **Unverified; do not force a mapping** |

If pandas lacks a meaningful, sufficiently populated `other` category, retain B for the three-class transfer evaluation or replace it with another audited repository; do **not** convert unmapped or unlabelled issues into `other`.

## D. Verified duplicate examples

### scikit-learn

| Query issue | Earlier target | Where stated | Predates query? |
|---|---|---|---|
| [#24430 — `cross_val_score` crashes with `StackingRegressor`](https://github.com/scikit-learn/scikit-learn/issues/24430) | [#24409](https://github.com/scikit-learn/scikit-learn/issues/24409) | Issue text: “Closing since this a duplicate of #24409.”  [github](https://github.com/scikit-learn/scikit-learn/issues/24430) | **Yes**: #24409 < #24430. |
| [#2801 — documentation of model/data persistence](https://github.com/scikit-learn/scikit-learn/issues/2801) | [#1332](https://github.com/scikit-learn/scikit-learn/issues/1332) | Issue text: “it turns out this issue was a duplicate of #1332.”  [github](https://github.com/scikit-learn/scikit-learn/issues/2801) | **Yes**: #1332 < #2801. |
| **Third pair: not yet verified** | — | — | The search result for [#6452](https://github.com/scikit-learn/scikit-learn/issues/6452) says it was marked as a duplicate of #9330, but #9330 has a higher issue number and therefore was created later; it must **not** be used as an earlier retrieval target.  [github](https://github.com/scikit-learn/scikit-learn/issues/6452) |

**Shortfall:** I verified only **two** chronologically valid explicit duplicate pairs for scikit-learn in this pass. That is below your requested minimum of three, so the local audit must find at least one more before retrieval evaluation is committed. A practical API/search check is:

```text
repo:scikit-learn/scikit-learn is:issue "duplicate of #"
```

Then retain only pairs where the referenced target issue number is lower than the query issue number.

### pandas

**No duplicate pairs were verified in this research pass.** Run the equivalent audit before selecting pandas as B:

```text
repo:pandas-dev/pandas is:issue "duplicate of #"
```

Retain only explicit target references, exclude PRs, and require `target.number < query.number`.

## E. Collection strategy

Use a **frozen snapshot collected on one day**, while recording `created_at`, `updated_at`, labels, state, and issue number for every retained issue.

| Purpose | Repository | Approximate allocation |
|---|---|---:|
| Classifier train/validation/test and same-repo retrieval | scikit-learn | 2,400–3,200 labelled issues |
| Cross-repository transfer evaluation | pandas | 600–1,000 labelled issues |
| Retrieval evaluation queries | Both, but primarily scikit-learn | 30–50 manually checked query–target pairs |

**Recommended date range for A:** collect issues created from **2019-01-01 through the snapshot date**, rather than only recently updated issues. This preserves enough chronological history for a 60/20/20 split while avoiding obsolete issue-tracker conventions.

**Recommended date range for B:** use the same creation-date window if volume permits; otherwise widen it only after confirming that label semantics remained stable.

To avoid class and recency bias:

- Query each target label separately, rather than taking the first N issues from a general issue search.
- Sort each per-label result by `created_at` ascending, then sample across the full window.
- Exclude PRs with `is:pr` or by checking the GitHub API issue/pull-request endpoint.
- Exclude issues with empty or trivially short bodies; retain title plus body as model input.
- Exclude issues whose mapped labels conflict semantically.
- Keep unlabelled historical issues in a separate retrieval pool; never assign them `other`.
- For retrieval, retain duplicate-marked issues even if their original class label is weak, but use only the **query issue’s pre-resolution text** as model input.

The chronological split should be by issue creation date: oldest 60% train, next 20% validation, newest 20% test. This prevents future issues from leaking into training through temporal ordering.

## F. Small API audit

Give your coding assistant this checklist:

1. **Confirm non-PR issue counts**
   - GitHub search: `repo:OWNER/REPO is:issue label:"LABEL"`
   - API: `GET /repos/{owner}/{repo}/issues?state=all&labels=LABEL&per_page=100&page=N`
   - Filter out every item with a `pull_request` key.
   - Record the query, filter, and audit date beside every count.

2. **Compute mapped target counts**
   - For scikit-learn, count unique non-PR issues for `Bug`, `Enhancement`, `Question`, `Documentation`, and `Build / CI`.
   - Compute overlap matrices: `Bug ∩ Enhancement`, `Question ∩ Documentation`, and every conflicting pair.
   - Report both raw label counts and post-conflict-exclusion usable counts; do not sum overlapping counts.

3. **Check class availability over time**
   - Group retained issues by creation year or quarter.
   - Produce a table of `bug`, `feature`, `question_documentation`, and `other` counts per period.
   - Reject a period/class only if it prevents a usable chronological split.

4. **Check body quality**
   - Count issues with empty, whitespace-only, or very short bodies.
   - For bugs, separately count issues containing likely reproduction signals: `repro`, `reproduce`, `steps`, `expected`, `actual`, `version`, `traceback`, or code fences.
   - Do not use these heuristics as labels; use them only for inclusion filtering.

5. **Find valid duplicate pairs**
   - Search issue bodies, titles, comments, and timeline events for `duplicate of #`, `dup of #`, and GitHub’s duplicate-marking event.
   - API timeline: `GET /repos/{owner}/{repo}/issues/{issue_number}/timeline`
   - Keep a pair only if the relationship explicitly names the target and `target.number < query.number`.
   - Do not infer duplication from ordinary cross-references, similar titles, or embeddings.

6. **Assess four-class feasibility**
   - Require enough post-filter examples in every class for at least a small stratified validation/test set.
   - If `other` is too sparse, do not silently merge it; either restrict the experiment to three classes or manually annotate a small, clearly defined `other` sample.

## G. Honest fallback

The evidence supports **scikit-learn as repository A**, but does **not yet** fully establish a production-ready B or a complete duplicate-pair set. The verified strengths are scikit-learn’s four relevant label families, a defensible `Build / CI` mapping for `other`, and two chronologically valid explicit duplicate pairs. [github](https://github.com/scikit-learn/scikit-learn/labels)

The least disruptive fallback is:

1. Begin the local audit with **scikit-learn as A** and **pandas as B**.
2. If pandas lacks a verified `other` mapping or sufficient question/documentation issues, retain it for a **three-class transfer evaluation** rather than forcing `other`.
3. If fewer than three valid duplicate pairs are found in either repository, reduce the retrieval evaluation to the verified pairs and explicitly report the shortfall; do not manufacture pairs from similar-issue retrieval results.
4. If `other` remains too small in A, manually annotate a small sample of clearly maintenance/build/CI issues, preserving the original label evidence and excluding unlabelled issues.

**Start the local audit with A and B because scikit-learn already provides verified semantic labels, a defensible `other` category, and chronologically valid duplicate evidence, while pandas is the most interpretable closely related Python-data-software transfer target—provided its exact labels, class volumes, and duplicate pairs pass the API audit.** [github](https://github.com/scikit-learn/scikit-learn/labels)

### Verified dataset decision — separate from the research response

**Status:** decided on 22 November 2026 from the project's own read-only GitHub audit and frozen
snapshots. Perplexity's claims were treated as unverified; corrections are listed below.

- **Audit date and saved output:** 2026-11-22, unauthenticated GitHub issue search with boolean
  operators (203 queries, resumable). `docs/audit/search_audit_20261122T113553Z.md` (+ `.json`
  with every query string and count). Command: `knotify audit` using
  `configs/audit_candidates.toml` (candidate mappings taken from each repository's live label list).
- **Selected repositories:** A = `scikit-learn/scikit-learn` (source: training, calibration,
  retrieval); B = `pandas-dev/pandas` (transfer check). Streamlit and FastAPI were rejected:
  Streamlit has only 129 single-target question/docs issues and no maintenance label; FastAPI's
  issues are 87% `question` (support moved to Discussions; `question-migrate` label), with 113 bug
  and 114 feature issues.
- **Collection windows (frozen):** A retrieval history = all 12,387 non-PR issues created
  2010-08-31 → 2026-11-21 (`data/raw/scikit-learn__scikit-learn/2010-01-01_2026-11-21_20261122T115450Z`,
  issues SHA-256 `6249a98c…`). A supervised window = created ≥ 2019-01-01. B = 1,912 issues created
  2024-09-11 → 2026-11-21, i.e. after A's validation cutoff
  (`data/raw/pandas-dev__pandas/2024-09-11_2026-11-21_20261122T120313Z`). Both snapshots were
  collected by date-partitioned issue search (≤1,000 results per partition); every partition
  returned exactly its reported count and the totals equal a fresh whole-window count
  (`partial: false` in each manifest).
- **Adopted mapping:** `configs/repositories.toml`. A: bug = `Bug`, `Regression`; feature =
  `Enhancement`, `New Feature`; question_documentation = `Question`, `Documentation`; other =
  `Build / CI`, `Packaging`, `Refactor`. B: bug = `Bug`, `Regression`; feature = `Enhancement`;
  question_documentation = `Docs`, `Usage Question`; other = `CI`, `Build`, `Testing`, `Clean`,
  `Code Style`, `Refactor`. `other` means developer-facing maintenance, never "unlabelled".
  Labels mapping to two different targets are conflicts and excluded. Labels of the same
  target (e.g. `Question` + `Documentation`) are not conflicts.
- **Label-provenance finding (verified from the repositories' `.github/ISSUE_TEMPLATE` files):**
  both repositories' templates auto-apply the type label together with `Needs Triage`
  (scikit-learn: `Bug`, `New Feature`, `Documentation`; pandas: `Bug`, `Enhancement`, `Docs`,
  `Usage Question`, `Build`). pandas templates also prefill titles with `BUG:`, `ENH:`, `DOC:`,
  `QST:`, `BUILD:` (92% of B's usable titles). Consequences adopted: (1) issues still carrying
  `Needs Triage` (label not yet maintainer-reviewed) are excluded from supervised data and counted;
  (2) the primary classifier input masks template cues (title type prefixes, Markdown headings,
  checkbox boilerplate; `knotify.features.mask_template_cues`), with an unmasked variant reported
  for comparison. Both are documented deviations/refinements, not proposal changes.
- **Counts — A (search audit, all years):** 12,388 issues, 21,770 PRs; single-target usable
  6,008 (bug 2,288 / feature 1,916 / question_documentation 1,523 / other 281); 260 multi-target
  conflicts; 6,120 unmapped. **A supervised set (snapshot, ≥ 2019):** 6,633 issues in window →
  3,850 single-target, 134 conflicts, 2,649 unmapped, 577 excluded `Needs Triage`, 2 exact-text
  duplicates removed → **3,271 examples** (bug 1,132 / feature 1,064 / question_documentation 866 /
  other 211, before dedup). Search and snapshot counts agree (3,851 vs 3,850).
- **Counts — B transfer set (snapshot):** 1,272 usable (bug 777 / feature 192 /
  question_documentation 221 / other 82); 55 conflicts, 331 unmapped, 254 `Needs Triage`.
  All four classes are present, so the transfer check stays four-class.
- **Chronological class coverage (frozen split, `data/splits/scikit-learn__scikit-learn/…/manifest.json`):**
  Train 1,963 (2019-01-02 → 2022-11-30; 556/718/575/114), Validation 654 (→ 2024-09-10;
  278/205/129/42), Test 654 (2024-09-11 → 2026-11-17; 298/140/161/55), in bug/feature/
  question_documentation/other order. Validation supports disjoint chronological halves:
  V-cal 327 (139/110/61/17) and V-policy 327 (139/95/68/25). `other` is the thinnest class
  everywhere; per-class results for it will carry wide uncertainty.
- **Explicit earlier duplicate targets:** search hits for "duplicate of": A 373, B 1,131
  (+1,134 `Duplicate Report` labels in B). Audit sample of 100 A hits: 12 verified pairs (target is
  an issue created earlier), 1 rejected (#14607 → #17170, target later). Statements are almost all
  in comments (only 2 in A issue bodies). Full collection for queries created ≥ 2022-11-30
  (start of A's validation period; 60 candidates, all comments checked, `knotify duplicates`):
  55 explicit pairs, **52 valid** (3 rejected: target created later), one target per query, so
  52 queries, 38 created in the validation period and 14 in the test period. 50 of the 52 statements
  were made by repository members. Frozen before any method was scored as
  `data/annotations/scikit-learn__scikit-learn/…/retrieval_queries.json`: 5 debug queries
  (earliest) + **47 evaluation queries**. This meets the plan's 30–50 target from explicit evidence,
  but only 14 queries fall in the classifier's test period. Retrieval methods have no fitted
  parameters, so validation-period queries remain held-out for retrieval. GitHub
  `MarkedAsDuplicateEvent` (GraphQL, needs a token) was not collected.
- **Feasibility outcome:** four-class experiment feasible on A and B; 3,271 supervised examples
  is inside the proposal's 2,000–4,000 range. Deviations to report: `Needs Triage` exclusion,
  template-cue masking, a thin `other` class (211 in A), and a retrieval set that may fall below
  30–50 queries if fewer explicit pairs are found (shortfall to be stated, not padded).
- **Corrections to the Perplexity research:** pandas does have labels for all four targets
  (`Docs`, not `Documentation`; `Usage Question`; maintenance labels `CI`, `Build`, `Testing`,
  `Clean`, `Code Style`, `Refactor`). Its scikit-learn mapping omitted `New Feature`, the most
  frequent feature label (1,253 issues vs 743 `Enhancement`). Its "stratified validation/test"
  and "exclude short bodies" suggestions were not adopted (chronological splits; only 8 short
  bodies among A's mapped issues). Its two cited pairs (#24430 → #24409, #2801 → #1332) have real
  issue endpoints with earlier targets, but neither statement is in the issue body as it
  claimed; comment evidence for them is unchecked.
- **Human label audit (23 Nov 2026):** one reviewer judged 40 stratified Train examples without AI
  suggestions: 28 agree, 3 disagree, 9 unclear. 7 of the 9 unclear cases are `other`; bug,
  feature and question_documentation agree in 26 of 30. The mapping was kept, and `other` is
  reported as the least certain class. Details are in `IMPLEMENTATION_STATUS.md`.
- **Frozen dataset and split manifest paths:** snapshots above (each with `manifest.json`);
  split IDs `data/splits/scikit-learn__scikit-learn/2010-01-01_2026-11-21_20261122T115450Z/`
  (`split_ids.json`, SHA-256 `1a09757a…`). `data/` is gitignored; recreate with the commands in
  `IMPLEMENTATION_STATUS.md`.
