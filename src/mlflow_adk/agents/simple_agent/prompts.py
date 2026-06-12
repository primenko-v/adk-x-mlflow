"""Source-of-truth for the simple_agent instruction."""

import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.entities.model_registry import PromptVersion
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

from mlflow_adk.settings import PromptSource, settings

logger = logging.getLogger(__name__)

PROMPT_NAME = "simple_agent.instruction"
# The ONE shared alias. All other aliases are per-developer or per-feature
# and never appear as constants in code.
PROMPT_PROD_ALIAS = "production"

# Canonical location of the committed frozen artifact for this agent. The
# ``freeze_prompt`` CLI writes here by default; the FROZEN path reads from
# here unless ``settings.prompt_frozen_path`` overrides.
FROZEN_PATH = Path(__file__).parent / "prompt_frozen.json"

_TEMPLATE = (
    "You are a helpful weather assistant. "
    "When asked about the temperature in a city, use the get_temperature tool. "
    "Respond concisely."
)


@dataclass(frozen=True)
class ResolvedPrompt:
    """Source-agnostic prompt payload returned by ``load()``."""

    template: str
    version: int


class FrozenPromptFile(BaseModel):
    """Schema of the committed ``prompt_frozen.json`` artifact."""

    name: str
    version: int
    template: str


def ref_to_uri(name: str, ref: str) -> str:
    """Convert ``@<alias>`` or ``<version>`` into a ``prompts:/...`` URI.

    Raises ``ValueError`` on malformed refs so misconfigurations fail fast.
    """
    if ref.startswith("@"):
        alias = ref[1:]
        if not alias:
            raise ValueError(f"Invalid prompt ref {ref!r}: empty alias after '@'.")
        return f"prompts:/{name}@{alias}"
    if not ref.isdigit():
        raise ValueError(
            f"Invalid prompt ref {ref!r}: must be '@<alias>' or a "
            f"positive integer version number."
        )
    return f"prompts:/{name}/{ref}"


def load() -> ResolvedPrompt:
    """Resolve the agent's instruction from the configured source."""
    if settings.prompt_source is PromptSource.FROZEN:
        return _load_frozen()
    return _load_from_registry()


def _load_frozen() -> ResolvedPrompt:
    path = settings.prompt_frozen_path or FROZEN_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"prompt_source={PromptSource.FROZEN.value} but no frozen artifact "
            f"found at {path}. Produce one with "
            f"`uv run python -m mlflow_adk.freeze_prompt`."
        )
    frozen = FrozenPromptFile.model_validate_json(path.read_text())
    return ResolvedPrompt(template=frozen.template, version=frozen.version)


def _load_from_registry() -> ResolvedPrompt:
    """Strict read of ``settings.prompt_ref`` from MLflow. Never writes."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = ref_to_uri(PROMPT_NAME, settings.prompt_ref)
    current = _try_load_uri(uri)
    if current is None:
        raise RuntimeError(
            f"prompt_source={PromptSource.REGISTRY.value} but `{uri}` does "
            f"not exist at {settings.mlflow_tracking_uri}. Run "
            f"`make register_prompt` and assign the alias in the UI, or "
            f"set PROMPT_REF to an existing ref."
        )
    return ResolvedPrompt(template=current.template, version=current.version)


def _try_load_uri(uri: str) -> PromptVersion | None:
    try:
        return mlflow.genai.load_prompt(uri)
    except MlflowException:
        return None


def register() -> int:
    """Publish ``_TEMPLATE`` as a new version. No alias work.

    Aliases are managed in the MLflow UI; after this returns, go assign
    your alias (e.g. ``@alice-tool-tweak``) to the new version there.
    Returns the new version number.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    new = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=_TEMPLATE,
        commit_message="Published from source via make register_prompt.",
        tags={"agent": "simple_agent"},
    )
    print(
        f"Registered {PROMPT_NAME} v{new.version}. "
        f"Assign an alias in the MLflow UI to refer to it by name."
    )
    return new.version
