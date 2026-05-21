import json
import os

import pytest
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

from mlflow_adk.simulate import (
    load_eval_set,
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

    scenarios = tmp_path / "scenarios"
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
        scenarios_dir=scenarios,
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
    scenarios = tmp_path / "scenarios"
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
        scenarios_dir=scenarios, experiment="test-exp", write_trace_ids=out
    )

    assert returned == ["tr-aaa", "tr-bbb"]
    assert out.read_text() == "tr-aaa\ntr-bbb\n"


@pytest.mark.unit
def test_load_eval_set_builds_cases_from_yaml(tmp_path):
    (tmp_path / "weather.yaml").write_text(
        "starting_prompt: 'What is the weather?'\n"
        "conversation_plan: 'Ask about temperature.\n'\n"
    )

    eval_set = load_eval_set(tmp_path)

    assert len(eval_set.eval_cases) == 1
    case = eval_set.eval_cases[0]
    assert case.eval_id == "weather"
    assert case.conversation_scenario.starting_prompt == "What is the weather?"
