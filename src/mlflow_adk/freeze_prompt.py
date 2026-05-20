"""Resolve a prompt from MLflow and write a committed frozen artifact.

The recommended production pattern is to keep MLflow off the runtime hot
path: a developer (or CI) resolves the current ``@production`` (or any
pinned version) once at build time and writes the resulting (template,
version) pair to a JSON file committed to git. The running agent then
reads from that file via ``PromptSource.FROZEN``, with no MLflow client
involvement.

This script does the build-time resolution. The output file is whatever
``settings.prompt_frozen_path`` points at, defaulting to the per-agent
canonical location (e.g. ``simple_agent/prompt_frozen.json``). The diff
between freezes is exactly the "what's about to ship" review signal.

    uv run python -m mlflow_adk.freeze_prompt
    git diff src/mlflow_adk/agents/simple_agent/prompt_frozen.json
    git commit -am "Freeze simple_agent prompt v4"

Pass ``--name`` to target a different prompt, ``--alias`` / ``--version``
to pick the revision, or ``--output`` to override the destination.
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow

from mlflow_adk.agents.simple_agent.prompts import (
    FROZEN_PATH,
    PROMPT_NAME,
    PROMPT_PROD_ALIAS,
    FrozenPromptFile,
)
from mlflow_adk.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--name", default=PROMPT_NAME, help=f"Prompt name (default: {PROMPT_NAME})"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--alias",
        default=PROMPT_PROD_ALIAS,
        help=f"Alias to resolve (default: {PROMPT_PROD_ALIAS})",
    )
    group.add_argument(
        "--version",
        type=int,
        help="Pin to a specific version number instead of an alias.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.prompt_frozen_path or FROZEN_PATH,
        help=f"Destination JSON file (default: {FROZEN_PATH}).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    selector = str(args.version) if args.version is not None else f"@{args.alias}"
    sep = "/" if args.version is not None else ""
    prompt = mlflow.genai.load_prompt(f"prompts:/{args.name}{sep}{selector}")

    frozen = FrozenPromptFile(
        name=args.name, version=prompt.version, template=prompt.template
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen.model_dump(), indent=2) + "\n")

    print(
        f"Wrote {args.output} ({args.name} v{prompt.version}) "
        f"from {settings.mlflow_tracking_uri}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
