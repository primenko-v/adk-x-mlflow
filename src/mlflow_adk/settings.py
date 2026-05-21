import os
from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PromptSource(StrEnum):
    """Where ``prompts.load()`` reads the agent's instruction from.

    - ``REGISTRY``: hit the MLflow Prompt Registry at import time. The
      dev/iteration workflow — fast feedback, history in MLflow.
    - ``FROZEN``: read a template + version from a JSON file that CI (or a
      developer) resolved from MLflow at build time and committed to git.
      The production workflow — no MLflow on the runtime hot path; the
      shipped prompt text is visible in PR review.

    See ``docs/mlflow-runs-and-traces.md`` and the ``freeze_prompt`` CLI.
    """

    REGISTRY = "registry"
    FROZEN = "frozen"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    google_cloud_project: str
    google_cloud_location: str = "global"
    google_genai_use_vertexai: bool = True

    agent_model: str = "gemini-3.1-flash-lite"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Judge model for MLflow conversation scorers (Correctness, Safety, etc.).
    # Format follows MLflow's convention: ``<provider>:/<model>`` — e.g.
    # ``openai:/gpt-4.1-mini``, ``gemini/gemini-2.5-flash`` (LiteLLM-routed).
    # ``None`` defers to MLflow's own default (openai:/gpt-4.1-mini on
    # non-Databricks tracking servers, which would require OPENAI_API_KEY).
    judge_model: str | None = None

    prompt_source: PromptSource = PromptSource.REGISTRY
    # What to load in REGISTRY mode. Format: ``@<alias>`` to load by alias,
    # or ``<version>`` (an integer-as-string) to pin a specific revision.
    # Default ``@production`` — if you don't override, you get the deployed
    # version. Devs override (e.g. ``PROMPT_REF=@primenko``) to read their
    # own iteration alias.
    prompt_ref: str = "@production"
    # Override the per-agent default frozen-artifact path. Usually unset —
    # each prompt module declares its own canonical location (see
    # ``simple_agent.prompts.FROZEN_PATH``).
    prompt_frozen_path: Path | None = None

    def model_post_init(self, _ctx) -> None:
        # google-adk / google-genai read these from os.environ directly, so
        # propagate the Settings values out for downstream libraries.
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", self.google_cloud_project)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", self.google_cloud_location)
        os.environ.setdefault(
            "GOOGLE_GENAI_USE_VERTEXAI", str(self.google_genai_use_vertexai).lower()
        )


settings = Settings()
