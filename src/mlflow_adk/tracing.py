"""MLflow + ADK tracing bridge.

Two functions, called in order around the OTel provider setup:

    setup_otlp_export()          # before the provider exists
    get_fast_api_app(...)        # creates the provider
    configure_tracing()          # after the provider exists

``setup_otlp_export`` sets the MLflow experiment and wires the OTLP env vars
that ADK's ``_setup_telemetry()`` reads when building its ``BatchSpanProcessor``.

``configure_tracing`` registers ``_SessionIdSpanProcessor`` on the now-existing
provider so MLflow receives session-grouped traces with populated inputs/outputs.
"""

import json
import os
import threading
from collections import defaultdict

import mlflow
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

from mlflow_adk.settings import settings

_configured_providers: set[int] = set()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Span attribute extractors
# ---------------------------------------------------------------------------


def _extract_user_text(llm_request_json: str) -> str | None:
    """Return the last user-role text from a gcp.vertex.agent.llm_request blob."""
    try:
        request = json.loads(llm_request_json)
        for content in reversed(request.get("contents", [])):
            if content.get("role") == "user":
                parts = content.get("parts", [])
                text = " ".join(p["text"] for p in parts if p.get("text"))
                if text:
                    return text
    except Exception:
        pass
    return None


def _extract_assistant_text(llm_response_json: str) -> str | None:
    """Return the model-role text from a gcp.vertex.agent.llm_response blob."""
    try:
        response = json.loads(llm_response_json)
        content = response.get("content") or {}
        parts = content.get("parts", [])
        text = " ".join(p["text"] for p in parts if p.get("text"))
        return text or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Span processor
# ---------------------------------------------------------------------------


class _SessionIdSpanProcessor(SpanProcessor):
    """Bridges ADK spans to MLflow's session/chat-session view.

    1. Copies ``gen_ai.conversation.id`` → ``session.id`` so MLflow groups
       traces into sessions.
    2. Collects user input and assistant output from ``call_llm`` child spans
       and injects them onto the root ``invoke_agent`` span as
       ``mlflow.spanInputs`` / ``mlflow.spanOutputs`` so the Sessions view
       shows non-empty turns.

    See docs/mlflow-session-view-bridge.md for a full explanation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # trace_id (int) → {"input": str | None, "output": str | None}
        self._pending: dict[int, dict] = defaultdict(
            lambda: {"input": None, "output": None}
        )

    def on_end(self, span: ReadableSpan) -> None:
        attrs = span.attributes or {}
        trace_id = span.context.trace_id

        # 1. Copy gen_ai.conversation.id → session.id
        session_id = attrs.get("gen_ai.conversation.id")
        if session_id and "session.id" not in attrs:
            # BoundedAttributes is mutable after span.end(); BatchSpanProcessor
            # queues a reference so this write is visible to the export thread.
            span._attributes["session.id"] = session_id

        # 2. Accumulate content from call_llm child spans.
        #    ADK always sets gcp.vertex.agent.llm_request ('{}' when content
        #    capture is disabled), so we check for a non-empty value.
        llm_req = attrs.get("gcp.vertex.agent.llm_request", "{}")
        llm_res = attrs.get("gcp.vertex.agent.llm_response", "{}")
        if llm_req and llm_req != "{}":
            with self._lock:
                entry = self._pending[trace_id]
                if entry["input"] is None:
                    entry["input"] = _extract_user_text(llm_req)
                out = _extract_assistant_text(llm_res)
                if out:
                    entry["output"] = out

        # 3. Root span: inject accumulated inputs/outputs.
        if span.parent is None:
            with self._lock:
                entry = self._pending.pop(trace_id, None)
            if entry:
                if entry["input"] and "mlflow.spanInputs" not in attrs:
                    span._attributes["mlflow.spanInputs"] = json.dumps(
                        [{"role": "user", "content": entry["input"]}]
                    )
                if entry["output"] and "mlflow.spanOutputs" not in attrs:
                    span._attributes["mlflow.spanOutputs"] = json.dumps(
                        [{"role": "assistant", "content": entry["output"]}]
                    )

    def on_start(self, span, parent_context=None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_otlp_export(experiment_name: str | None = None) -> None:
    """Configure MLflow as the OTLP destination for ADK traces.

    Sets the MLflow experiment and writes the OTLP env vars that ADK's
    ``_setup_telemetry()`` reads when it builds the ``BatchSpanProcessor``.
    Must be called **before** ``get_fast_api_app`` (or any other call that
    creates the OTel provider).

    Args:
        experiment_name: MLflow experiment to create or activate.  Defaults to
            ``settings.mlflow_experiment``.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(experiment_name or settings.mlflow_experiment)
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"{settings.mlflow_tracking_uri}/v1/traces",
    )
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_HEADERS",
        f"x-mlflow-experiment-id={experiment.experiment_id}",
    )


def configure_tracing() -> bool:
    """Register the ADK → MLflow span processor on the current OTel provider.

    Must be called **after** the ``SDKTracerProvider`` has been created (i.e.
    after ``get_fast_api_app`` or equivalent).  Safe to call multiple times —
    the processor is added at most once per provider instance.

    Returns:
        ``True`` if the processor was registered, ``False`` if no
        ``SDKTracerProvider`` is available (no-op when running without OTel).
    """
    provider = trace.get_tracer_provider()
    if not isinstance(provider, SDKTracerProvider):
        return False

    with _lock:
        provider_id = id(provider)
        if provider_id in _configured_providers:
            return True
        _configured_providers.add(provider_id)

    provider.add_span_processor(_SessionIdSpanProcessor())
    return True
