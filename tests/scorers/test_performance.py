"""Tests for performance scorers."""

from types import SimpleNamespace

import pytest

from mlflow_adk.scorers.performance import turn_latency_ms, turn_tokens


@pytest.mark.unit
def test_turn_latency_returns_execution_duration():
    trace = SimpleNamespace(
        info=SimpleNamespace(execution_duration=850.5),
        data=SimpleNamespace(spans=[]),
    )
    fb = turn_latency_ms(trace=trace)
    assert fb.value == 850.5
    assert fb.name == "turn_latency_ms"


@pytest.mark.unit
def test_turn_latency_records_error_when_duration_missing():
    """Missing duration should not silently become 0 — that would pollute aggregates."""
    trace = SimpleNamespace(
        info=SimpleNamespace(execution_duration=None),
        data=SimpleNamespace(spans=[]),
    )
    fb = turn_latency_ms(trace=trace)
    assert fb.value is None
    assert fb.error is not None


@pytest.mark.unit
def test_turn_tokens_returns_total_tokens_from_trace_metadata():
    trace = SimpleNamespace(
        info=SimpleNamespace(
            token_usage={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
        ),
        data=SimpleNamespace(spans=[]),
    )
    fb = turn_tokens(trace=trace)
    assert fb.value == 150.0
    assert fb.name == "turn_tokens"


@pytest.mark.unit
def test_turn_tokens_records_error_when_usage_missing():
    """Tool-only / routing turns may have no LLM usage — record None, not 0."""
    trace = SimpleNamespace(
        info=SimpleNamespace(token_usage=None),
        data=SimpleNamespace(spans=[]),
    )
    fb = turn_tokens(trace=trace)
    assert fb.value is None
    assert fb.error is not None
