import logging

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

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
}


def get_temperature(city: str) -> dict:
    """Return the current temperature for a city.

    Args:
        city: Name of the city.

    Returns:
        A dict with 'city' and 'temp_c' keys, or an error message.
    """
    temp = _CITIES.get(city.lower().strip())
    if temp is None:
        logger.info("get_temperature: '%s' not found", city)
        return {"error": f"No temperature data for '{city}'"}
    logger.info("get_temperature: %s → %.1f°C", city, temp)
    return {"city": city, "temp_c": temp}


def get_cities() -> list[str]:
    """Return a list of the cities supported by the `get_temperature` tool

    Args: None

    Returns:
        A list of the keys supported by the `get_temperature` tool
    """
    cities = list(_CITIES.keys())
    logger.info("get_cities: %s", ", ".join(cities))
    return cities


root_agent = Agent(
    name="simple_agent",
    model=settings.agent_model,
    description="A simple weather assistant that answers temperature questions.",
    instruction=_prompt.template,
    tools=[FunctionTool(get_temperature), FunctionTool(get_cities)],
)
