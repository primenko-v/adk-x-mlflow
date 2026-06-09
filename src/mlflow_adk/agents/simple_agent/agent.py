import logging

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from mlflow_adk.agents.simple_agent import prompts
from mlflow_adk.settings import settings

logger = logging.getLogger(__name__)

_prompt = prompts.load()
PROMPT_NAME: str = prompts.PROMPT_NAME
PROMPT_VERSION: int = _prompt.version

_CITIES: dict[str, float] = {
    "london": 12.0,
    "berlin": 8.5,
    "paris": 14.0,
    "tokyo": 22.0,
    "new york": 18.0,
    "sydney": 20.0,
    "helsinki": 5.0,
    "stockholm": 6.0,
    "lisbon": 19.0,
}

_UNITS = ("celsius", "fahrenheit")
_DEFAULT_UNIT = "celsius"
_UNIT_STATE_KEY = "temperature_unit"


def _lookup_temp_c(city: str) -> float | None:
    """Celsius temperature for a city, or None if unknown. Pure lookup."""
    return _CITIES.get(city.lower().strip())


def _from_celsius(temp_c: float, unit: str) -> float:
    """Convert a Celsius temperature into the requested unit (1 decimal)."""
    if unit == "fahrenheit":
        return round(temp_c * 9 / 5 + 32, 1)
    return round(temp_c, 1)


def get_temperature(city: str, tool_context: ToolContext) -> dict:
    """Return the current temperature for a city.

    The value is reported in the user's preferred unit, read from session
    state (defaults to Celsius). Set the preference with set_temperature_unit.

    Args:
        city: Name of the city.

    Returns:
        A dict with 'city', 'temp' and 'unit' keys, or an error message.
    """
    unit = tool_context.state.get(_UNIT_STATE_KEY, _DEFAULT_UNIT)
    temp_c = _lookup_temp_c(city)
    if temp_c is None:
        logger.info("get_temperature: '%s' not found", city)
        return {"error": f"No temperature data for '{city}'"}
    temp = _from_celsius(temp_c, unit)
    logger.info("get_temperature: %s → %.1f (%s)", city, temp, unit)
    return {"city": city, "temp": temp, "unit": unit}


def get_cities() -> list[str]:
    """Return a list of the cities supported by the `get_temperature` tool

    Args: None

    Returns:
        A list of the keys supported by the `get_temperature` tool
    """
    cities = list(_CITIES.keys())
    logger.info("get_cities: %s", ", ".join(cities))
    return cities


def set_temperature_unit(unit: str, tool_context: ToolContext) -> dict:
    """Remember the user's preferred temperature unit for future answers.

    Call this whenever the user expresses a preference for Celsius or
    Fahrenheit. The choice persists for the rest of the conversation and
    governs how get_temperature reports values.

    Args:
        unit: Either 'celsius' or 'fahrenheit'.

    Returns:
        A dict echoing the stored unit, or an error message.
    """
    normalized = unit.lower().strip()
    if normalized not in _UNITS:
        logger.info("set_temperature_unit: rejected '%s'", unit)
        return {"error": f"Unsupported unit '{unit}'. Use 'celsius' or 'fahrenheit'."}
    tool_context.state[_UNIT_STATE_KEY] = normalized
    logger.info("set_temperature_unit: %s", normalized)
    return {_UNIT_STATE_KEY: normalized}


def _init_state(callback_context: CallbackContext) -> None:
    """Seed the default temperature unit if the session hasn't set one.

    setdefault leaves a pre-seeded value (e.g. from an eval's session_input)
    intact, so seeded state always wins over this default.
    """
    callback_context.state.setdefault(_UNIT_STATE_KEY, _DEFAULT_UNIT)


root_agent = Agent(
    name="simple_agent",
    model=settings.agent_model,
    description="A simple weather assistant that answers temperature questions.",
    instruction=_prompt.template,
    tools=[
        FunctionTool(get_temperature),
        FunctionTool(get_cities),
        FunctionTool(set_temperature_unit),
    ],
    before_agent_callback=_init_state,
)
