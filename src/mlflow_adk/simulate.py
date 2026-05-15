import argparse
import asyncio
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from google.adk.telemetry.setup import maybe_set_otel_providers

from mlflow_adk.settings import settings
from mlflow_adk.tracing import configure_tracing, setup_otlp_export

AGENT_MODULE = "mlflow_adk.agents.simple_agent"

logger = logging.getLogger(__name__)


def _load_eval_set(scenarios_dir: Path) -> EvalSet:
    cases = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        scenario = ConversationScenario.model_validate(yaml.safe_load(path.read_text()))
        cases.append(EvalCase(eval_id=path.stem, conversation_scenario=scenario))
    logger.info("Loaded %d scenario(s) from %s", len(cases), scenarios_dir)
    return EvalSet(eval_set_id="adk-x-mlflow", eval_cases=cases)


async def main(scenarios_dir: Path, agent_module: str = AGENT_MODULE) -> None:
    load_dotenv()
    setup_otlp_export(settings.mlflow_simulation_experiment)
    maybe_set_otel_providers()
    configure_tracing()

    eval_set = _load_eval_set(scenarios_dir)
    logger.info(
        "Starting simulation — experiment: %s | agent: %s | scenarios: %d",
        settings.mlflow_simulation_experiment,
        agent_module,
        len(eval_set.eval_cases),
    )

    # TODO(once PR lands): pass user_simulator_config=LlmBackedUserSimulatorConfig(...)
    # loaded from simulations/user_simulator.yaml.
    result = await EvaluationGenerator.generate_responses(
        eval_set=eval_set,
        agent_module_path=agent_module,
        repeat_num=1,
    )

    logger.info("Simulation complete — %d case(s) processed", len(result))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("simulations/scenarios"))
    parser.add_argument("--agent", default=AGENT_MODULE)
    args = parser.parse_args()
    asyncio.run(main(args.scenarios, agent_module=args.agent))
