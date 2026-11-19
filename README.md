# Knotify

**Confidence-aware, evidence-linked GitHub issue triage.**

An academic AIML project with a local Streamlit interface and a CPU-first Python pipeline.
This commit is a **runnable scaffold**, not a completed ML application. It includes no collected
dataset, trained model, invented recommendations or measured performance.

## Project references

- [Final five-day plan](KNOTIFY_FINAL_PLAN.md): agreed scope and implementation sequence.
- [Context](context.md): factual background, submitted proposal details, decisions and code map.
- [Implementation status](IMPLEMENTATION_STATUS.md): current completion and next milestone.
- [Submitted proposal](Knotify_AIML_Project_Proposal%20%281%29%20%281%29.pdf): unchanged original.

## Quick start — macOS or Windows

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed. For example,
with an existing Python installation: `python -m pip install --user uv`. If the executable is
not on PATH, use `python -m uv` in place of `uv` in the following commands.

From this repository root:

```sh
uv sync --locked --python 3.12
uv run knotify doctor
uv run streamlit run ui/app.py
```

Open the localhost address printed by Streamlit. The shell has an issue-input form, an honest
unavailable-state report preview, and a project-readiness view. No GitHub token or LLM key is
required. Initial dependency/Python installation requires internet; opening the installed shell
does not. Python 3.12 is the development target for both Windows and the 16 GB MacBook.

## Dependencies grow with implementation

The default environment installs the app/contracts and development tools. Heavy components are
optional so the UI and tests do not need PyTorch, FAISS, models or GPU drivers.

```sh
# When implementing the baseline/importer:
uv sync --locked --extra ml
# When implementing semantic embeddings:
uv sync --locked --extra ml --extra embeddings
# Optional native search/explanation dependencies:
uv sync --locked --extra ml --extra embeddings --extra faiss --extra explain
```

Use the same extras on later `uv sync` commands; syncing without them removes optional packages.
Installing an extra does not implement the corresponding feature. The lock records resolved
versions; only the default scaffold environment has been locally exercised at this stage.
Native FAISS and ML extras still need validation on the target MacBook. The CPU configuration is
explicit; optional MPS/GPU support is not implemented or required. PyTorch wheel selection for
other platforms can be refined when the embeddings extra is first installed.

## Configuration and local files

`configs/default.toml` is validated by `knotify.config`; paths resolve relative to this editable
checkout, not the shell's current directory. `knotify --config /path/config.toml doctor` validates
an alternate config. Repository mapping is an unaudited example, not a chosen dataset.

Copy `.env.example` to `.env` only when credentials are needed. The scaffold does not read secrets
into the UI or make external API calls. The CLI loads `.env`; future API adapters can use the same
explicit bootstrap. `.env`, Streamlit secrets, datasets, artifacts and run outputs are ignored.

`data/`, `artifacts/` and `runs/` each explain their intended contents. Model/config hashes will
identify future runs; source labels never enter `data.model_text()`.

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run knotify schema
```

Tests cover contracts, probability validation, temporal eligibility, metric edge cases,
component failure isolation, append-only report/review files, CLI honesty and Streamlit flows.
CI runs the lightweight suite on Linux, Windows and macOS. There is no ML accuracy claim yet.

## Code map

`src/knotify` contains shared contracts, config, adapter protocols, an `analyze()` orchestration
shell, eligibility/metric helpers, storage and CLI. `ui/app.py` is a thin Streamlit view.
`ModelBundle` accepts classifier/retriever/checklist implementations; defaults report unavailable.
No service container, queue, database, frontend build system or web API is needed for this stage.

The next product milestone is a real imported issue with TF-IDF classification and BM25 retrieval,
following the final plan. The UI direction is aesthetic, clean and sleek: restrained colour,
clear typography, generous spacing, good empty/error states and inspectable evidence.

**Dataset selection is the first implementation gate.** Public issues will come from audited
existing open-source repositories, not this Knotify repository. No source repositories or usable
class counts have been confirmed yet. The example label mapping is not training ground truth.
The dataset audit and collection must precede model tuning; see the implementation status file.
