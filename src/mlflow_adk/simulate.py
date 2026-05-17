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

from mlflow_adk.tracing import configure_tracing, setup_otlp_export

AGENT_MODULE = "mlflow_adk.agents.simple_agent"
DEFAULT_EXPERIMENT = "adk-simulation"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENARIOS_DIR = PROJECT_ROOT / "simulations/scenarios"
DEFAULT_USER_SIMULATOR_CONFIG = PROJECT_ROOT / "simulations/user_simulator.yaml"

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


async def main(
    scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    experiment: str = DEFAULT_EXPERIMENT,
    agent_module: str = AGENT_MODULE,
    user_simulator_config_path: Path = DEFAULT_USER_SIMULATOR_CONFIG,
) -> None:
    setup_otlp_export(experiment)
    maybe_set_otel_providers()
    configure_tracing()

    eval_set = load_eval_set(scenarios_dir)
    user_simulator_config = load_user_simulator_config(user_simulator_config_path)
    logger.info(
        "Starting simulation — experiment=%s agent=%s scenarios=%d simulator=%s",
        experiment,
        agent_module,
        len(eval_set.eval_cases),
        user_simulator_config.model if user_simulator_config else "default",
    )

    result = await EvaluationGenerator.generate_responses(
        eval_set=eval_set,
        agent_module_path=agent_module,
        repeat_num=1,
        user_simulator_config=user_simulator_config,
    )

    logger.info("Simulation complete — %d case(s) processed", len(result))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_DIR)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--agent", default=AGENT_MODULE)
    parser.add_argument(
        "--user-simulator",
        type=Path,
        default=DEFAULT_USER_SIMULATOR_CONFIG,
        dest="user_simulator",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.scenarios,
            experiment=args.experiment,
            agent_module=args.agent,
            user_simulator_config_path=args.user_simulator,
        )
    )
