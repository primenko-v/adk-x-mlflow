from types import SimpleNamespace

import pytest

from mlflow_adk.agents.simple_agent.agent import (
    _from_celsius,
    _init_state,
    _lookup_temp_c,
    get_temperature,
    set_temperature_unit,
)


def _ctx(state: dict) -> SimpleNamespace:
    """Minimal stand-in for ToolContext/CallbackContext.

    The tools and callback only ever touch ``.state``; a plain dict gives the
    same get/setdefault/__setitem__ behaviour as ADK's State object.
    """
    return SimpleNamespace(state=state)


@pytest.mark.unit
def test_lookup_temp_c_normalizes_case_and_whitespace():
    assert _lookup_temp_c("  Berlin  ") == 8.5


@pytest.mark.unit
def test_from_celsius_celsius_is_passthrough():
    assert _from_celsius(12.0, "celsius") == 12.0


@pytest.mark.unit
def test_from_celsius_fahrenheit_converts():
    assert _from_celsius(0.0, "fahrenheit") == 32.0


@pytest.mark.unit
def test_get_temperature_uses_unit_from_state():
    result = get_temperature("berlin", _ctx({"temperature_unit": "fahrenheit"}))
    assert result == {"city": "berlin", "temp": 47.3, "unit": "fahrenheit"}


@pytest.mark.unit
def test_get_temperature_unknown_city_returns_error():
    result = get_temperature("atlantis", _ctx({}))
    assert "error" in result


@pytest.mark.unit
def test_set_temperature_unit_persists_choice():
    state = {}
    result = set_temperature_unit("Fahrenheit", _ctx(state))
    assert result == {"temperature_unit": "fahrenheit"}
    assert state == {"temperature_unit": "fahrenheit"}


@pytest.mark.unit
def test_set_temperature_unit_rejects_unknown_unit_without_writing():
    state = {}
    result = set_temperature_unit("kelvin", _ctx(state))
    assert "error" in result
    assert state == {}


@pytest.mark.unit
def test_init_state_leaves_seeded_value_intact():
    state = {"temperature_unit": "fahrenheit"}
    _init_state(_ctx(state))
    assert state == {"temperature_unit": "fahrenheit"}
