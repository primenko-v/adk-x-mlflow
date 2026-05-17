"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def no_tracing(monkeypatch):
    """Stub out MLflow/OTel tracing setup so tests don't require a live server."""
    monkeypatch.setattr("mlflow_adk.simulate.setup_otlp_export", lambda *a: None)
    monkeypatch.setattr("mlflow_adk.simulate.maybe_set_otel_providers", lambda: None)
    monkeypatch.setattr("mlflow_adk.simulate.configure_tracing", lambda: None)
