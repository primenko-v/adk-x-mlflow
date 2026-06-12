import json
import os

import pytest
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

from mlflow_adk.simulate import (
    load_eval_set,
    load_simulation_list,
    load_user_simulator_config,
    run_simulation,
)


@pytest.mark.unit
def test_load_user_simulator_config_maps_yaml_camelcase_to_snake_case(tmp_path):
    path = tmp_path / "user_simulator.yaml"
    path.write_text("model: gemini-test\nmaxAllowedInvocations: 7\n")

    config = load_user_simulator_config(path)

    assert config.model == "gemini-test"
    assert config.max_allowed_invocations == 7


@pytest.mark.unit
def test_load_user_simulator_config_returns_none_when_file_missing(tmp_path):
    assert load_user_simulator_config(tmp_path / "missing.yaml") is None


@pytest.mark.integration
async def test_simulation_writes_spans_to_file_sink(tmp_path, monkeypatch):
    """End-to-end demo for the ADK PR.

    Drives the simple_agent with an LLM-backed user simulator configured from
    YAML (this PR's feature), routes spans to a JSONL file sink instead of
    MLflow, and verifies the agent actually ran.

    Reviewers can run just this test (requires GOOGLE_CLOUD_PROJECT and ADC
    via ``gcloud auth application-default login``):

        uv run pytest tests/test_simulate.py -v -m integration
    """
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        pytest.skip("Requires GOOGLE_CLOUD_PROJECT")

    scenarios = tmp_path / "conversations"
    scenarios.mkdir()
    (scenarios / "weather.yaml").write_text(
        'starting_prompt: "What\'s the weather in London?"\n'
        "conversation_plan: 'Ask once and stop.'\n"
    )
    config_path = tmp_path / "user_simulator.yaml"
    config_path.write_text("model: gemini-3.1-flash-lite\nmaxAllowedInvocations: 1\n")
    monkeypatch.setattr("mlflow_adk.simulate.USER_SIMULATOR_CONFIG", config_path)
    traces_path = tmp_path / "traces.jsonl"

    await run_simulation(
        input_dir=scenarios,
        experiment=None,
        output_traces=traces_path,
    )

    # In production, BatchSpanProcessor flushes via atexit on process exit; in
    # this in-process test we must flush explicitly before reading the file.
    provider = trace.get_tracer_provider()
    assert isinstance(provider, SDKTracerProvider)
    provider.force_flush()

    assert traces_path.exists(), "expected spans file to be written"
    spans = [
        json.loads(line)
        for line in traces_path.read_text().splitlines()
        if line.strip()
    ]
    assert spans, "expected at least one span to be captured"

    span_names = {s.get("name") for s in spans}
    assert any(n and n.startswith("invoke_agent") for n in span_names), (
        f"expected an invoke_agent span; got: {sorted(span_names)}"
    )


@pytest.mark.unit
async def test_write_trace_ids_writes_one_id_per_line(
    tmp_path, monkeypatch, no_tracing
):
    scenarios = tmp_path / "conversations"
    scenarios.mkdir()
    (scenarios / "a.yaml").write_text(
        "starting_prompt: 'Hi'\nconversation_plan: 'Ask once.\n'\n"
    )
    (scenarios / "b.yaml").write_text(
        "starting_prompt: 'Hello'\nconversation_plan: 'Ask once.\n'\n"
    )

    async def fake_process_query(
        module_name, user_simulator, agent_name=None, initial_session=None
    ):
        return []

    monkeypatch.setattr(EvaluationGenerator, "_process_query", fake_process_query)

    # Stub the drain so it pretends a trace was flushed per scenario; we just
    # want to verify the IDs flow from there into the output file.
    fake_ids = iter([["tr-aaa"], ["tr-bbb"]])
    monkeypatch.setattr(
        "mlflow_adk.simulate.flush_and_apply_tags",
        lambda: next(fake_ids),
    )
    out = tmp_path / "ids.txt"

    returned = await run_simulation(
        input_dir=scenarios,
        experiment="test-exp",
        write_trace_ids=out,
    )

    assert returned == ["tr-aaa", "tr-bbb"]
    assert out.read_text() == "tr-aaa\ntr-bbb\n"


@pytest.mark.unit
async def test_run_simulation_runs_scenario_and_static_cases(
    tmp_path, monkeypatch, no_tracing
):
    # Subdirectories under one input dir, found by the recursive read.
    convos = tmp_path / "conversations"
    (convos / "scenarios").mkdir(parents=True)
    (convos / "static").mkdir()
    (convos / "scenarios" / "scen.yaml").write_text(
        "starting_prompt: 'Hi'\nconversation_plan: 'Ask once.\n'\n"
    )
    (convos / "static" / "static.yaml").write_text(
        "messages:\n  - 'What is the temp?'\n"
    )

    # ADK selects the user simulator per case; capturing its type per call
    # proves recursion finds both subdirs and dispatches scenario→LLM,
    # static→static.
    seen_simulators = []

    async def fake_process_query(
        module_name, user_simulator, agent_name=None, initial_session=None
    ):
        seen_simulators.append(type(user_simulator).__name__)
        return []

    monkeypatch.setattr(EvaluationGenerator, "_process_query", fake_process_query)

    monkeypatch.setattr("mlflow_adk.simulate.flush_and_apply_tags", lambda: [])

    await run_simulation(input_dir=convos, experiment="test-exp")

    assert seen_simulators == ["LlmBackedUserSimulator", "StaticUserSimulator"]


@pytest.mark.unit
def test_load_eval_set_builds_static_case_from_messages(tmp_path):
    (tmp_path / "seeded.yaml").write_text(
        "state:\n  temperature_unit: fahrenheit\n"
        "messages:\n  - 'What is the temperature in Berlin?'\n  - 'And Tokyo?'\n"
    )

    case = load_eval_set(tmp_path).eval_cases[0]

    assert case.eval_id == "seeded"
    assert case.conversation_scenario is None
    messages = [inv.user_content.parts[0].text for inv in case.conversation]
    assert messages == ["What is the temperature in Berlin?", "And Tokyo?"]
    assert case.session_input.state == {"temperature_unit": "fahrenheit"}


@pytest.mark.unit
def test_load_eval_set_rejects_file_with_neither_key(tmp_path):
    (tmp_path / "bogus.yaml").write_text("something_else: 1\n")

    with pytest.raises(ValueError, match="bogus.yaml"):
        load_eval_set(tmp_path)


@pytest.mark.unit
def test_load_eval_set_static_without_state_has_no_session_input(tmp_path):
    (tmp_path / "plain.yaml").write_text("messages:\n  - 'Hi'\n")

    assert load_eval_set(tmp_path).eval_cases[0].session_input is None


@pytest.mark.unit
def test_load_simulation_list_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "default.txt"
    path.write_text(
        "# a comment\n\ncurious_traveler\n  temperature_*  \n# trailing comment\n"
    )

    assert load_simulation_list(path) == ["curious_traveler", "temperature_*"]


@pytest.mark.unit
def test_load_eval_set_select_filters_by_stem_across_subdirs(tmp_path):
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "static").mkdir()
    (tmp_path / "scenarios" / "curious.yaml").write_text(
        "starting_prompt: 'Hi'\nconversation_plan: 'Ask once.\n'\n"
    )
    (tmp_path / "static" / "seeded.yaml").write_text("messages:\n  - 'Hi'\n")
    (tmp_path / "static" / "other.yaml").write_text("messages:\n  - 'Bye'\n")

    eval_set = load_eval_set(tmp_path, select=["curious", "seeded"])

    assert {c.eval_id for c in eval_set.eval_cases} == {"curious", "seeded"}


@pytest.mark.unit
def test_load_eval_set_select_supports_glob(tmp_path):
    (tmp_path / "temperature_basic.yaml").write_text(
        "starting_prompt: 'Hi'\nconversation_plan: 'Ask once.\n'\n"
    )
    (tmp_path / "temperature_units.yaml").write_text("messages:\n  - 'Hi'\n")
    (tmp_path / "unrelated.yaml").write_text("messages:\n  - 'Bye'\n")

    eval_set = load_eval_set(tmp_path, select=["temperature_*"])

    assert {c.eval_id for c in eval_set.eval_cases} == {
        "temperature_basic",
        "temperature_units",
    }


@pytest.mark.unit
def test_load_eval_set_select_by_path_under_subdir(tmp_path):
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "static").mkdir()
    (tmp_path / "scenarios" / "curious.yaml").write_text(
        "starting_prompt: 'Hi'\nconversation_plan: 'Ask once.\n'\n"
    )
    (tmp_path / "static" / "seeded.yaml").write_text("messages:\n  - 'Hi'\n")

    eval_set = load_eval_set(tmp_path, select=["static/seeded"])

    assert {c.eval_id for c in eval_set.eval_cases} == {"seeded"}


@pytest.mark.unit
def test_load_eval_set_select_folder_prefix_runs_whole_subdir(tmp_path):
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "static").mkdir()
    (tmp_path / "scenarios" / "curious.yaml").write_text(
        "starting_prompt: 'Hi'\nconversation_plan: 'Ask once.\n'\n"
    )
    (tmp_path / "static" / "seeded.yaml").write_text("messages:\n  - 'Hi'\n")
    (tmp_path / "static" / "other.yaml").write_text("messages:\n  - 'Bye'\n")

    eval_set = load_eval_set(tmp_path, select=["static/"])

    assert {c.eval_id for c in eval_set.eval_cases} == {"seeded", "other"}


@pytest.mark.unit
def test_load_eval_set_select_unmatched_pattern_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("messages:\n  - 'Hi'\n")

    with pytest.raises(ValueError, match="nope"):
        load_eval_set(tmp_path, select=["nope"])


@pytest.mark.unit
def test_load_eval_set_builds_scenario_case_from_yaml(tmp_path):
    (tmp_path / "weather.yaml").write_text(
        "starting_prompt: 'What is the weather?'\n"
        "conversation_plan: 'Ask about temperature.\n'\n"
    )

    case = load_eval_set(tmp_path).eval_cases[0]

    assert case.eval_id == "weather"
    assert case.conversation is None
    assert case.conversation_scenario.starting_prompt == "What is the weather?"
