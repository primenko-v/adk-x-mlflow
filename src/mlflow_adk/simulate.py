import argparse
import asyncio
import importlib
import logging
from pathlib import Path
from typing import Any

import mlflow
import yaml
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import EvalCase, Invocation, SessionInput
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulatorConfig,
)
from google.adk.telemetry.setup import maybe_set_otel_providers
from google.genai import types
from mlflow.entities.model_registry import PromptVersion
from pydantic import BaseModel

from mlflow_adk.tracing import (
    add_file_sink,
    configure_tracing,
    flush_and_apply_tags,
    git_info,
    link_prompt_to_traces,
    setup_otlp_export,
    trace_tags,
)

AGENT_MODULE = "mlflow_adk.agents.simple_agent"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "simulations/conversations"
USER_SIMULATOR_CONFIG = PROJECT_ROOT / "simulations/user_simulator.yaml"

# Cosmetic identifiers for the in-memory session a static conversation runs in.
# The agent under test is stateless beyond session state, so these only show up
# as the session's app_name/user_id in traces.
STATIC_APP_NAME = "simple_agent"
STATIC_USER_ID = "static_user"

logger = logging.getLogger(__name__)


class StaticConversationFile(BaseModel):
    """Schema of a fixed-input conversation file.

    ``messages`` are the verbatim user turns, replayed in order by ADK's
    StaticUserSimulator (no LLM). ``state`` optionally seeds the session before
    turn 1 — the "evaluate a turn given pre-established context" path.
    """

    messages: list[str]
    state: dict[str, Any] | None = None

    def to_eval_case(self, eval_id: str) -> EvalCase:
        """Convert to an ADK EvalCase that the StaticUserSimulator will replay."""
        conversation = [
            Invocation(
                user_content=types.Content(
                    role="user", parts=[types.Part(text=message)]
                )
            )
            for message in self.messages
        ]
        session_input = (
            SessionInput(
                app_name=STATIC_APP_NAME, user_id=STATIC_USER_ID, state=self.state
            )
            if self.state
            else None
        )
        return EvalCase(
            eval_id=eval_id, conversation=conversation, session_input=session_input
        )


def load_user_simulator_config(path: Path) -> LlmBackedUserSimulatorConfig | None:
    if not path.exists():
        return None
    return LlmBackedUserSimulatorConfig.model_validate(yaml.safe_load(path.read_text()))


def _build_eval_case(path: Path) -> EvalCase:
    """Build an EvalCase from one YAML file, choosing the mode by its shape.

    A file with a ``messages`` list is a fixed-input conversation (ADK selects
    the StaticUserSimulator); anything else is an LLM-driven
    ConversationScenario. The two are mutually exclusive on EvalCase, so the
    file's content alone decides which simulator runs it.
    """
    data = yaml.safe_load(path.read_text())
    if "messages" in data:
        # StaticUserSimulator path
        return StaticConversationFile.model_validate(data).to_eval_case(path.stem)

    elif "conversation_plan" in data:
        # LlmBackedUserSimulator path
        scenario = ConversationScenario.model_validate(data)
        return EvalCase(eval_id=path.stem, conversation_scenario=scenario)

    raise ValueError(
        f"{path.name}: expected a 'messages' (static) or 'conversation_plan' "
        f"(scenario) key, found neither."
    )


def load_eval_set(input_dir: Path) -> EvalSet:
    """Load every ``*.yaml`` under ``input_dir`` (recursively) as an EvalCase.

    Each file is classified by its shape via ``_build_eval_case``, so scenario
    and static files can be organised into whatever subdirectories you like
    (e.g. ``scenarios/`` and ``static/``) without any extra configuration.
    """
    cases = [_build_eval_case(path) for path in sorted(input_dir.rglob("*.yaml"))]
    logger.info("Loaded %d eval case(s) from %s", len(cases), input_dir)
    return EvalSet(eval_set_id="adk-x-mlflow", eval_cases=cases)


def _setup_simulation_tracing(
    experiment: str | None, output_traces: Path | None
) -> None:
    """Configure the OTel pipeline for the simulation run.

    ``experiment is None`` disables the MLflow export path entirely (spans
    still flow to the file sink if one is configured). Order matters: OTLP
    env vars must be set before the provider is created, and the span
    processor must be registered after. See ``tracing.py`` for why each step
    is its own primitive.
    """
    if experiment is not None:
        setup_otlp_export(experiment)
    maybe_set_otel_providers()
    if output_traces is not None:
        add_file_sink(output_traces)
        logger.info("Writing spans to %s", output_traces)
    configure_tracing()


async def run_simulation(
    input_dir: Path = DEFAULT_INPUT_DIR,
    experiment: str | None = None,
    agent_module: str = AGENT_MODULE,
    output_traces: Path | None = None,
    write_trace_ids: Path | None = None,
) -> list[str]:
    """Run every eval case found under ``input_dir`` (recursively).

    The directory holds both LLM-driven scenario files and fixed-input
    ``messages`` files (typically in ``scenarios/`` and ``static/``
    subdirectories); ADK picks the user simulator per case, so both kinds run
    in one batch (see ``load_eval_set``).

    ``experiment`` is the MLflow experiment name. Default ``None`` disables
    MLflow export — only useful in combination with ``output_traces`` to
    capture spans to a file for offline inspection. At least one of
    ``experiment`` or ``output_traces`` must be set.

    ``write_trace_ids``, if provided, is the path to write the MLflow trace
    IDs of every scenario in this batch (one per line) once the batch
    finishes. The output is the explicit handoff that
    ``evaluate.py --trace-ids <path>`` reads — scoring exactly the traces
    that the simulation produced, with no tag-filter guesswork.

    Returns the list of trace IDs produced (same as the file contents),
    so programmatic callers can chain simulate → evaluate in-process.
    """
    if experiment is None and output_traces is None:
        raise ValueError(
            "Refusing to run: no trace sink configured. "
            "Provide an experiment name, an output_traces path, or both."
        )

    _setup_simulation_tracing(experiment, output_traces)

    eval_set = load_eval_set(input_dir)
    if not eval_set.eval_cases:
        raise ValueError(f"No eval cases found under {input_dir}. Nothing to run.")
    user_simulator_config = load_user_simulator_config(USER_SIMULATOR_CONFIG)
    logger.info(
        "Starting simulation — experiment=%s agent=%s scenarios=%d simulator=%s",
        experiment,
        agent_module,
        len(eval_set.eval_cases),
        user_simulator_config.model if user_simulator_config else "default",
    )

    # Provenance shared across every scenario in this batch. Computed once;
    # spread into the per-scenario tag dict below. ``prompt_version`` is the
    # MLflow Prompt Registry version pinned by the agent — lets us filter the
    # trace list by which revision of the instruction produced each scenario.
    base_tags = {
        "source": "simulation",
        "agent_module": agent_module,
        **git_info(),
    }
    agent_pkg = importlib.import_module(f"{agent_module}.agent")
    prompt_version = getattr(agent_pkg, "PROMPT_VERSION", None)
    prompt_name = getattr(agent_pkg, "PROMPT_NAME", None)
    if prompt_version is not None:
        base_tags["prompt_version"] = str(prompt_version)
    # Fetch the PromptVersion handle once so we can populate the trace's
    # "Linked prompts" sidebar in the MLflow UI. Independent of the
    # ``prompt_version`` tag above — the tag is a string for filtering; the
    # linker creates a clickable association between trace and registry.
    prompt_version_obj: PromptVersion | None = None
    if experiment is not None and prompt_name and prompt_version is not None:
        try:
            prompt_version_obj = mlflow.genai.load_prompt(
                f"prompts:/{prompt_name}/{prompt_version}"
            )
        except Exception:
            logger.exception(
                "Could not load prompt %s v%d for trace linking — traces will "
                "still be tagged but the UI sidebar will be empty.",
                prompt_name,
                prompt_version,
            )

    all_trace_ids: list[str] = []
    total = 0
    for eval_case in eval_set.eval_cases:
        single_case = EvalSet(eval_set_id=eval_set.eval_set_id, eval_cases=[eval_case])

        # ContextVar.set returns a token capturing the previous value;
        # reset(token) restores it. The try/finally guarantees the restore
        # even on error — without it, a failed scenario would leak its tags
        # into the next iteration (or into any outer caller's context). No
        # except clause: we want failures to propagate and stop the batch.
        token = trace_tags.set(
            {
                **base_tags,
                "scenario": eval_case.eval_id,
                # "static" = fixed-input replay, "scenario" = LLM-driven. Lets
                # the MLflow trace list be filtered by input mode.
                "conversation_mode": (
                    "static" if eval_case.conversation is not None else "scenario"
                ),
                # Overrides the root-span-derived default name in the UI's
                # trace list so scenarios are scannable without opening each.
                "mlflow.traceName": eval_case.eval_id,
            }
        )
        try:
            result = await EvaluationGenerator.generate_responses(
                eval_set=single_case,
                agent_module_path=agent_module,
                repeat_num=1,
                user_simulator_config=user_simulator_config,
            )
        finally:
            trace_tags.reset(token)

        total += len(result)

        if experiment is not None:
            request_ids = flush_and_apply_tags()
            all_trace_ids.extend(request_ids)
            if prompt_version_obj is not None and request_ids:
                link_prompt_to_traces(prompt_version_obj, request_ids)

    if write_trace_ids is not None and all_trace_ids:
        write_trace_ids.write_text("\n".join(all_trace_ids) + "\n")
        logger.info("Wrote %d trace ID(s) to %s", len(all_trace_ids), write_trace_ids)

    logger.info("Simulation complete — %d case(s) processed", total)
    return all_trace_ids


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        dest="input_dir",
        help="Directory of conversation YAML files, read recursively.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment name. Omit to disable MLflow export.",
    )
    parser.add_argument("--agent", default=AGENT_MODULE)
    parser.add_argument(
        "--output-traces",
        type=Path,
        default=None,
        dest="output_traces",
        help="Also write spans as JSON to this path.",
    )
    parser.add_argument(
        "--write-trace-ids",
        type=Path,
        default=None,
        dest="write_trace_ids",
        help=(
            "Write the MLflow trace ID of each scenario (one per line) to "
            "this path. Pair with `evaluate.py --trace-ids <path>`."
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        run_simulation(
            args.input_dir,
            experiment=args.experiment,
            agent_module=args.agent,
            output_traces=args.output_traces,
            write_trace_ids=args.write_trace_ids,
        )
    )
