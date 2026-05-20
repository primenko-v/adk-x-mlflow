"""Shared pytest fixtures and global test environment.

The module-level env setup below runs before any test module is collected,
which is how it can take effect before `from ...agent import ...` triggers
`prompts.load()` at import time. Without it, unit tests would require a
live MLflow server with the `@candidate` alias populated.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

_TEST_FROZEN_FILE = Path(tempfile.gettempdir()) / "mlflow_adk_test_prompt_frozen.json"
_TEST_FROZEN_FILE.write_text(
    json.dumps(
        {
            "name": "simple_agent.instruction",
            "version": 0,
            "template": "test instruction",
        }
    )
)
os.environ.setdefault("PROMPT_SOURCE", "frozen")
os.environ.setdefault("PROMPT_FROZEN_PATH", str(_TEST_FROZEN_FILE))


@pytest.fixture
def no_tracing(monkeypatch):
    """Stub out MLflow/OTel tracing setup so tests don't require a live server."""
    monkeypatch.setattr("mlflow_adk.simulate.setup_otlp_export", lambda *a: None)
    monkeypatch.setattr("mlflow_adk.simulate.maybe_set_otel_providers", lambda: None)
    monkeypatch.setattr("mlflow_adk.simulate.configure_tracing", lambda: None)
