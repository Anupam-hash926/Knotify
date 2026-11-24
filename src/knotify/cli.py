"""Small diagnostics entry point. Future commands fail clearly rather than report fake work."""

import argparse
import json
from importlib.util import find_spec
from pathlib import Path

from dotenv import load_dotenv

from knotify import __version__
from knotify.config import PROJECT_ROOT, load_config

PENDING = ()
DEFAULT_MAPPING = PROJECT_ROOT / "configs" / "repositories.toml"
DEFAULT_AUDIT_PLAN = PROJECT_ROOT / "configs" / "audit_candidates.toml"
AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knotify offline tools (read-only GitHub access)")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Validate config and show local capabilities; no downloads")
    commands.add_parser("schema", help="Print the report JSON schema")
    audit = commands.add_parser("audit", help="Read-only GitHub dataset feasibility audit")
    audit.add_argument("--plan", type=Path, default=DEFAULT_AUDIT_PLAN)
    audit.add_argument("--out", type=Path, default=AUDIT_DIR)
    audit.add_argument("--repo", action="append", help="Limit to these repositories")
    audit.add_argument("--verify-limit", type=int, default=20)
    audit.add_argument("--time-budget", type=float, default=150, help="Seconds per run")
    imp = commands.add_parser("import", help="Read-only GitHub import into a frozen snapshot")
    imp.add_argument("repository")
    imp.add_argument("--method", choices=("search", "rest"), default="search")
    imp.add_argument("--start", required=True, help="Earliest creation date (YYYY-MM-DD)")
    imp.add_argument("--end", help="Latest creation date (default: today, UTC)")
    imp.add_argument("--time-budget", type=float, default=110, help="Seconds per run (search)")
    split = commands.add_parser("split", help="Freeze chronological splits for a snapshot")
    split.add_argument("repository")
    split.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    train = commands.add_parser("train", help="Train TF-IDF baselines; score validation only")
    train.add_argument("repository")
    dups = commands.add_parser("duplicates", help="Collect explicit duplicate evidence")
    dups.add_argument("repository")
    dups.add_argument("--created-from", required=True, help="Query issues created on/after")
    dups.add_argument("--time-budget", type=float, default=110)
    sheet = commands.add_parser("label-sheet", help="Write the ~40-example label-audit sheet")
    sheet.add_argument("repository")
    embed = commands.add_parser("embed", help="Encode the frozen snapshot once (MiniLM, CPU)")
    embed.add_argument("repository")
    train.add_argument(
        "--model", choices=("tfidf", "embedding"), default="tfidf", help="Which baseline to train"
    )
    calib = commands.add_parser(
        "calibrate", help="Calibrate on V-cal; choose classifier and threshold on V-policy"
    )
    calib.add_argument("repository")
    shap_setup = commands.add_parser("explain-setup", help="Save the Train SHAP background")
    shap_setup.add_argument("repository")
    sheets = commands.add_parser("review-sheets", help="Freeze evaluation samples; write sheets")
    sheets.add_argument("repository")
    freeze = commands.add_parser("freeze", help="Write or verify the Day 3 freeze manifest")
    freeze.add_argument("repository")
    freeze.add_argument("--verify", action="store_true")
    ev = commands.add_parser("evaluate", help="Day 4: evaluate frozen files (reads Test)")
    ev.add_argument("repository")
    ev.add_argument("--transfer", default="pandas-dev/pandas")
    ev.add_argument("--report-only", action="store_true", help="Only re-render results.md")
    lat = commands.add_parser("latency", help="Median warm analysis time on this machine")
    lat.add_argument("repository")
    lat.add_argument("--issues", type=int, default=12)
    lat.add_argument("--label", default="local")
    for name in PENDING:
        commands.add_parser(name, help="Reserved: not implemented yet")
    args = parser.parse_args(argv)
    if args.command in PENDING:
        parser.exit(2, f"{args.command}: not implemented in the scaffold; no work was performed.\n")
    if args.command == "schema":
        from knotify.contracts import AnalysisReport

        print(json.dumps(AnalysisReport.model_json_schema(), indent=2))
        return 0
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if args.command == "audit":
        return _audit(args)
    if args.command == "import":
        return _import(args)
    if args.command == "duplicates":
        return _duplicates(args)
    if args.command == "label-sheet":
        from knotify.experiment import label_audit_sheet

        print(label_audit_sheet(load_config(args.config), args.repository))
        return 0
    if args.command in ("evaluate", "latency"):
        from knotify import evaluation

        config = load_config(args.config)
        manifest = evaluation.check_freeze()
        run_dir = config.paths.runs / "evaluation" / manifest["serving"]["split"]
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "latency":
            result = evaluation.measure_latency(config, args.repository, issues=args.issues)
            (run_dir / f"latency_{args.label}.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({k: result[k] for k in ("runs", "median_ms", "p90_ms")}, indent=2))
            return 0
        if not args.report_only:
            result = evaluation.evaluate_classification(config, args.repository)
            preds = result.pop("_predictions")
            (run_dir / "test_predictions.json").write_text(json.dumps(preds))
            (run_dir / "classification.json").write_text(json.dumps(result, indent=2))
            transfer = evaluation.evaluate_transfer(config, args.repository, args.transfer)
            (run_dir / "transfer.json").write_text(json.dumps(transfer, indent=2))
            evidence = evaluation.evidence_mechanical(config, args.repository)
            (run_dir / "evidence_mechanical.json").write_text(json.dumps(evidence, indent=2))
            from knotify.embeddings import load_encoder
            from knotify.errors import ComponentUnavailable

            try:
                encoder = load_encoder(config.embeddings.local_path)
            except ComponentUnavailable as exc:
                print(f"Retrieval skipped ({exc}); existing retrieval.json kept if present.")
            else:
                retrieval = evaluation.evaluate_retrieval(config, args.repository, encoder)
                (run_dir / "retrieval.json").write_text(json.dumps(retrieval, indent=2))
        try:
            scored = evaluation.human_sections(config, args.repository, run_dir)
            print("Human sheets scored:", scored)
        except ImportError as exc:
            print(
                f"Human sheets not re-scored ({exc.name} missing). Run: uv sync --locked "
                "--python 3.12 --extra ml --extra explain. Existing results are kept."
            )
        (run_dir / "results.md").write_text(evaluation.render_results(run_dir), encoding="utf-8")
        print(run_dir / "results.md")
        return 0
    if args.command == "freeze":
        from knotify.freeze import verify_freeze, write_freeze

        path = PROJECT_ROOT / "docs" / "freeze" / "day3_freeze.json"
        if args.verify:
            problems = verify_freeze(path)
            print("intact" if not problems else f"CHANGED: {problems}")
            return 0 if not problems else 1
        manifest = write_freeze(load_config(args.config), args.repository, path)
        print(json.dumps({k: manifest[k] for k in ("serving", "checklist_rule_version")}, indent=2))
        print(f"{len(manifest['files'])} files hashed -> {path}")
        return 0
    if args.command == "review-sheets":
        from knotify.annotation import prepare_sheets

        print(json.dumps(prepare_sheets(load_config(args.config), args.repository), indent=2))
        return 0
    if args.command == "explain-setup":
        from knotify.explain import build_background

        print(build_background(load_config(args.config), args.repository))
        return 0
    if args.command == "calibrate":
        from knotify.selection import calibrate_and_select

        report = calibrate_and_select(load_config(args.config), args.repository)
        summary = {
            "selection": report["selection"],
            "policy": report["policy"],
            "log_loss": {
                k: {"raw": v["raw"]["log_loss"], "calibrated": v["calibrated"]["log_loss"]}
                for k, v in report["candidates"].items()
            },
        }
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "embed":
        from knotify.experiment import build_embeddings

        path = build_embeddings(load_config(args.config), args.repository)
        print((path / "meta.json").read_text(encoding="utf-8"))
        return 0
    if args.command == "train" and args.model == "embedding":
        from knotify.experiment import train_embedding_lr

        result = train_embedding_lr(load_config(args.config), args.repository)
        keys = ("evaluated_on", "macro_f1", "accuracy", "per_class_f1", "truncated_fraction")
        print(json.dumps({k: result[k] for k in keys}, indent=2))
        return 0
    if args.command in ("split", "train"):
        from knotify import experiment

        config = load_config(args.config)
        if args.command == "split":
            path = experiment.prepare_splits(config, args.mapping, args.repository)
            print((path / "manifest.json").read_text(encoding="utf-8"))
        else:
            result = experiment.train_baseline(config, args.repository)
            summary = {
                name: {k: m.get(k) for k in ("macro_f1", "accuracy", "per_class_f1")}
                for name, m in result["models"].items()
            }
            print(json.dumps({"evaluated_on": result["evaluated_on"], **summary}, indent=2))
        return 0
    config = load_config(args.config)
    packages = ("sklearn", "rank_bm25", "sentence_transformers", "faiss", "shap")
    print(
        json.dumps(
            {
                "version": __version__,
                "frozen_snapshots": sorted(
                    str(p.relative_to(config.paths.data))
                    for p in config.paths.data.glob("raw/*/*/manifest.json")
                ),
                "trained_models": sorted(
                    str(p.relative_to(config.paths.artifacts))
                    for p in config.paths.artifacts.glob("*/*/*.joblib")
                ),
                "device": config.embeddings.device,
                "llm_enabled": config.llm.enabled,
                "paths": config.paths.model_dump(mode="json"),
                "optional_packages": {name: find_spec(name) is not None for name in packages},
            },
            indent=2,
        )
    )
    return 0


def _import(args) -> int:
    from knotify.github import GitHubClient
    from knotify.ingest import CollectionPaused, snapshot_repository, snapshot_via_search

    data_dir = load_config(args.config).paths.data
    with GitHubClient.from_env(max_wait=65) as client:
        try:
            if args.method == "search":
                path = snapshot_via_search(
                    client,
                    args.repository,
                    data_dir,
                    start=args.start,
                    end=args.end,
                    time_budget=args.time_budget,
                )
            else:
                path = snapshot_repository(
                    client, args.repository, data_dir, created_floor=args.start
                )
        except CollectionPaused as exc:
            print(f"PAUSED: {exc}")
            return 3
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    summary = {k: manifest.get(k) for k in ("issue_count", "partial", "created_range")}
    print(json.dumps({"snapshot": str(path), **summary}, indent=2))
    return 1 if manifest.get("partial") else 0


def _duplicates(args) -> int:
    from knotify.duplicates import collect
    from knotify.experiment import latest_snapshot, slug
    from knotify.github import GitHubClient
    from knotify.ingest import load_snapshot

    config = load_config(args.config)
    snapshot = latest_snapshot(config.paths.data, args.repository)
    _, records, _ = load_snapshot(snapshot)
    out = config.paths.data / "duplicates" / slug(args.repository) / snapshot.name
    out.mkdir(parents=True, exist_ok=True)
    with GitHubClient.from_env(max_wait=65) as client:
        summary = collect(
            client,
            args.repository,
            records,
            out,
            created_from=args.created_from,
            time_budget=args.time_budget,
        )
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete"] else 3


def _audit(args) -> int:
    from dataclasses import replace
    from datetime import UTC, datetime

    from knotify.audit import AuditPaused, load_plan, render_markdown, run_audit
    from knotify.github import GitHubClient

    plan = load_plan(args.plan)
    if args.repo:
        selected = tuple(r for r in plan.repositories if r.name in args.repo)
        plan = replace(plan, repositories=selected)
    cache = PROJECT_ROOT / "runs" / "audit_cache.json"
    with GitHubClient.from_env(max_wait=65) as client:
        try:
            report = run_audit(
                client,
                plan,
                verify_limit=args.verify_limit,
                cache_path=cache,
                time_budget=args.time_budget,
            )
        except AuditPaused as exc:
            print(f"PAUSED: {exc}")
            return 3
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.out / f"search_audit_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
