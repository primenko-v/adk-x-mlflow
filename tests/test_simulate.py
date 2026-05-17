import pytest
from google.adk.evaluation.evaluation_generator import EvaluationGenerator

from mlflow_adk.simulate import load_eval_set, main


@pytest.mark.unit
async def test_generate_responses_uses_user_simulator_config_from_yaml(
    tmp_path, monkeypatch, no_tracing
):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "s.yaml").write_text(
        "starting_prompt: 'Hi'\nconversation_plan: 'Ask once.\n'\n"
    )
    (tmp_path / "user_simulator.yaml").write_text(
        "model: gemini-test\nmaxAllowedInvocations: 7\n"
    )

    captured = {}

    async def fake_process_query(
        module_name, user_simulator, agent_name=None, initial_session=None
    ):
        captured["user_simulator"] = user_simulator
        return []

    monkeypatch.setattr(EvaluationGenerator, "_process_query", fake_process_query)

    await main(
        scenarios_dir=scenarios,
        user_simulator_config_path=tmp_path / "user_simulator.yaml",
    )

    sim = captured["user_simulator"]
    assert sim._config.model == "gemini-test"
    assert sim._config.max_allowed_invocations == 7


@pytest.mark.unit
def test_load_eval_set_builds_cases_from_yaml(tmp_path):
    (tmp_path / "weather.yaml").write_text(
        "starting_prompt: 'What is the weather?'\n"
        "conversation_plan: 'Ask about temperature.\n'\n"
    )

    eval_set = load_eval_set(tmp_path)

    assert eval_set.eval_set_id == "adk-x-mlflow"
    assert len(eval_set.eval_cases) == 1
    case = eval_set.eval_cases[0]
    assert case.eval_id == "weather"
    assert case.conversation_scenario.starting_prompt == "What is the weather?"
