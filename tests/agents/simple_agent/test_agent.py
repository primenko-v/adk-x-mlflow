import pytest

from mlflow_adk.agents.simple_agent.agent import _CITIES, get_cities, get_temperature


@pytest.mark.unit
def test_get_temperature_known_city_returns_temp():
    assert get_temperature("london") == {"city": "london", "temp_c": 12.0}


@pytest.mark.unit
def test_get_temperature_unknown_city_returns_error():
    assert get_temperature("atlantis") == {
        "error": "No temperature data for 'atlantis'"
    }


@pytest.mark.unit
def test_get_temperature_mixed_case_is_normalized():
    assert get_temperature("Berlin") == {"city": "Berlin", "temp_c": 8.5}


@pytest.mark.unit
def test_get_temperature_extra_whitespace_is_stripped():
    assert get_temperature("  tokyo  ") == {"city": "  tokyo  ", "temp_c": 22.0}


@pytest.mark.unit
def test_get_cities_returns_all_supported_cities():
    assert get_cities() == list(_CITIES.keys())
