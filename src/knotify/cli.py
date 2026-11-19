"""Small diagnostics entry point. Future commands fail clearly rather than report fake work."""

import argparse
import json
from importlib.util import find_spec
from pathlib import Path

from dotenv import load_dotenv

from knotify import __version__
from knotify.config import PROJECT_ROOT, load_config

PENDING = ("import", "train", "calibrate", "evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knotify scaffold tools")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Validate config and show local capabilities; no downloads")
    commands.add_parser("schema", help="Print the report JSON schema")
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
    config = load_config(args.config)
    packages = ("sklearn", "sentence_transformers", "faiss", "shap")
    print(
        json.dumps(
            {
                "version": __version__,
                "stage": "scaffold — no trained models or dataset",
                "device": config.embeddings.device,
                "llm_enabled": config.llm.enabled,
                "paths": config.paths.model_dump(mode="json"),
                "optional_packages": {name: find_spec(name) is not None for name in packages},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
