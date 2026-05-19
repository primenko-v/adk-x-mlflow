import argparse
import asyncio
import logging
from pathlib import Path

import yaml
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulatorConfig,
)
from google.adk.telemetry.setup import maybe_set_otel_providers

from mlflow_adk.tracing import (
    add_file_sink,
    configure_tracing,
    flush_and_apply_tags,
    git_info,
    setup_otlp_export,
    trace_tags,
)

AGENT_MODULE = "mlflow_adk.agents.simple_agent"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENARIOS_DIR = PROJECT_ROOT / "simulations/scenarios"
USER_SIMULATOR_CONFIG = PROJECT_ROOT / "simulations/user_simulator.yaml"

logger = logging.getLogger(__name__)


def load_user_simulator_config(path: Path) -> LlmBackedUserSimulatorConfig | None:
    if not path.exists():
        return None
    return LlmBackedUserSimulatorConfig.model_validate(yaml.safe_load(path.read_text()))


def load_eval_set(scenarios_dir: Path) -> EvalSet:
    cases = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        scenario = ConversationScenario.model_validate(yaml.safe_load(path.read_text()))
        cases.append(EvalCase(eval_id=path.stem, conversation_scenario=scenario))
    logger.info("Loaded %d scenario(s) from %s", len(cases), scenarios_dir)
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
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    experiment: str | None = None,
    agent_module: str = AGENT_MODULE,
    output_traces: Path | None = None,
) -> None:
    """Run all scenarios in ``scenarios_dir``.

    ``experiment`` is the MLflow experiment name. Default ``None`` disables
    MLflow export — only useful in combination with ``output_traces`` to
    capture spans to a file for offline inspection. At least one of
    ``experiment`` or ``output_traces`` must be set.
    """
    if experiment is None and output_traces is None:
        raise ValueError(
            "Refusing to run: no trace sink configured. "
            "Provide an experiment name, an output_traces path, or both."
        )

    _setup_simulation_tracing(experiment, output_traces)

    eval_set = load_eval_set(scenarios_dir)
    user_simulator_config = load_user_simulator_config(USER_SIMULATOR_CONFIG)
    logger.info(
        "Starting simulation — experiment=%s agent=%s scenarios=%d simulator=%s",
        experiment,
        agent_module,
        len(eval_set.eval_cases),
        user_simulator_config.model if user_simulator_config else "default",
    )

    # Provenance shared across every scenario in this batch. Computed once;
    # spread into the per-scenario tag dict below.
    base_tags = {
        "source": "simulation",
        "agent_module": agent_module,
        **git_info(),
    }

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
            flush_and_apply_tags()

    logger.info("Simulation complete — %d case(s) processed", total)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_DIR)
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
    args = parser.parse_args()
    asyncio.run(
        run_simulation(
            args.scenarios,
            experiment=args.experiment,
            agent_module=args.agent,
            output_traces=args.output_traces,
        )
    )
