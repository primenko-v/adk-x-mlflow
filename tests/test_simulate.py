
import pytest

from mlflow_adk.simulate import _load_eval_set


@pytest.mark.unit
def test_load_eval_set_builds_cases_from_yaml(tmp_path):
    (tmp_path / "weather.yaml").write_text(
        "starting_prompt: 'What is the weather?'\n"
        "conversation_plan: 'Ask about temperature.\n'\n"
    )

    eval_set = _load_eval_set(tmp_path)

    assert eval_set.eval_set_id == "adk-x-mlflow"
    assert len(eval_set.eval_cases) == 1
    case = eval_set.eval_cases[0]
    assert case.eval_id == "weather"
    assert case.conversation_scenario.starting_prompt == "What is the weather?"
