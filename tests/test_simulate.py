import json
import os

import pytest
from google.adk.evaluation.evaluation_generator import EvaluationGenerator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

from mlflow_adk.simulate import load_eval_set, run_simulation


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

    await run_simulation(
        scenarios_dir=scenarios,
        user_simulator_config_path=tmp_path / "user_simulator.yaml",
    )

    sim = captured["user_simulator"]
    assert sim._config.model == "gemini-test"
    assert sim._config.max_allowed_invocations == 7


@pytest.mark.integration
async def test_simulation_writes_spans_to_file_sink(tmp_path):
    """End-to-end demo for the ADK PR.

    Drives the simple_agent with an LLM-backed user simulator configured from
    YAML (this PR's feature), routes spans to a JSONL file sink instead of
    MLflow, and verifies the agent actually ran.

    Reviewers can run just this test (requires GOOGLE_CLOUD_PROJECT + Vertex
    AI credentials):

        uv run pytest tests/test_simulate.py -v -m integration
    """
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        pytest.skip("Requires GOOGLE_CLOUD_PROJECT for Vertex AI calls")

    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "weather.yaml").write_text(
        'starting_prompt: "What\'s the weather in London?"\n'
        "conversation_plan: 'Ask once and stop.'\n"
    )
    config_path = tmp_path / "user_simulator.yaml"
    config_path.write_text("model: gemini-3.1-flash-lite\nmaxAllowedInvocations: 1\n")
    traces_path = tmp_path / "traces.jsonl"

    await run_simulation(
        scenarios_dir=scenarios,
        user_simulator_config_path=config_path,
        mlflow_enabled=False,
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
