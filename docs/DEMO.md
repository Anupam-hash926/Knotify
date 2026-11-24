# Knotify — demo guide (about 5 minutes)

Launch from the repository root (needs the shared `data/` and `artifacts/` folders; see README):

```sh
uv run streamlit run ui/app.py
```

The first analysis takes about 10–15 s while the search index warms up; later ones take about
2–3 s. Everything runs locally; nothing is posted to GitHub. The examples below were picked
**after** evaluation, purely to demonstrate features; they are not evaluation evidence.

## 1. A confident category and a confirmed duplicate: issue **32072**

Type `32072` in *Open an imported issue* → **Analyze issue**.

- **Category:** `bug`, accepted, calibrated confidence 0.95, which meets the 0.95 review
  threshold chosen on validation data. Point out that only about 5% of Test issues get an
  automatic category; everything else is deferred to a maintainer.
- **Related earlier issues:** #31859 is ranked **1st**. A maintainer explicitly closed 32072 as a
  duplicate of 31859. Each result shows its BM25 rank and embedding rank; hybrid search fuses
  them. Stress that similarity is *not* proof of a duplicate.
- **Missing information:** everything present, so no questions are drafted.

## 2. Deferral with an explanation: issue **25859**

"How to early stop in GradientBoostingClassifer?" (labelled `New Feature`).

- **Category:** *Needs a maintainer's decision*, candidate `feature` at 0.60.
- **Why this candidate?** opens the SHAP word contributions: "currently" and "implementation"
  support `feature`; "how to" argues against it (it sounds like a question). These explain the
  TF-IDF model's raw score, not the calibrated confidence or whether the answer is right.

## 3. Missing information and a drafted reply: issue **29019**

"TunedThreasholdClassifierCV failing inside a SearchCV object".

- **Checklist:** expected behaviour, environment and version are **missing** (a human reviewer
  confirmed all three). Reproduction code, actual behaviour and the error are present.
- **Suggested follow-up:** three questions in an editable draft. Each cites the checklist item
  it comes from. Because the category is deferred, the wording is conditional ("If this is a
  bug report…").
- **Maintainer review:** click **Save report**, choose *accepted / rejected / edited*, add a note
  and **Record review**. The saved report itself is never modified.

## 4. Hybrid search beating keyword search: issue **28218**

"StratifiedGroupKFold not ensuring Stratified splits". Its confirmed duplicate **#24656** is
ranked **2nd** by hybrid search. Keyword search alone ranked it 27th; the embedding ranking
brings it up. (On the 47 evaluation queries, embedding and hybrid search found the duplicate in the
top 5 for 45% of queries vs 36% for BM25. The difference is not statistically conclusive.)

## 5. A brand-new pasted report

In *Or paste a new report*, keep the repository and paste:

- Title: `PCA crashes when n_components is a float`
- Description: `When I call PCA(n_components=0.5).fit(X) it crashes. Please fix.`

Expect a `bug` candidate (deferred), several missing items (code, expected/actual result,
environment, version, error message) and a drafted reply. History cutoff for pasted reports is
the analysis time. Without the optional embeddings extra installed, pasted text uses BM25 only, and
the app says so.

## Talking points

- **Deliberate limits:** confidence applies to the category only; search results and questions
  carry no confidence; nothing is automated.
- **An honest finding from the explanations:** issue **21291**'s top words are `_no response_`,
  GitHub's placeholder for empty form fields. Template masking removed headings and
  title prefixes but not this placeholder, so some form-structure signal remains. It was found
  after the freeze and is reported as a limitation rather than "fixed" after seeing Test.
- **Results summary:** see `docs/results/results.md` (Test macro-F1 0.785; checklist missing-item
  F1 0.789; 0/56 questions without valid evidence).
