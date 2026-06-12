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
import logging
import os
import threading
from collections import defaultdict
from contextvars import ContextVar
from pathlib import Path

import git
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import PromptVersion
from mlflow.tracing.utils import generate_mlflow_trace_id_from_otel_trace_id
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from mlflow_adk.settings import settings

logger = logging.getLogger(__name__)

_configured_providers: set[int] = set()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Trace tagging
# ---------------------------------------------------------------------------

# Ambient tags applied to every trace whose root span ends while this var is
# set. Callers set it before driving the agent (per-scenario in the simulation
# loop, once at startup for the interactive server) and the span processor
# buffers (request_id, tags) for each root span seen. The caller then drains
# the buffer and applies tags via mlflow.set_trace_tag once spans have been
# flushed.
trace_tags: ContextVar[dict[str, str] | None] = ContextVar("trace_tags", default=None)

_tagged_traces_buffer: list[tuple[str, dict[str, str]]] = []
_tagged_traces_lock = threading.Lock()


def drain_tagged_trace_ids() -> list[tuple[str, dict[str, str]]]:
    """Pop and return all buffered (request_id, tags) entries."""
    with _tagged_traces_lock:
        entries = _tagged_traces_buffer[:]
        _tagged_traces_buffer.clear()
    return entries


def git_info() -> dict[str, str]:
    """Best-effort git provenance tags for the current checkout.

    Returns a dict with ``git_commit``, ``git_branch`` (omitted on detached
    HEAD), ``git_dirty``. Empty dict if git is unavailable or the cwd is not
    a git checkout — callers can splat unconditionally:

        tags = {"source": "simulation", **git_info()}

    Keys deliberately do **not** use MLflow's ``mlflow.source.git.*``
    convention. The MLflow UI hides ``mlflow.*``-prefixed tags from the
    trace-detail chip display (treats them as system tags), so unprefixed
    keys are used for visibility. ``dirty`` still matches MLflow's semantic
    — tracked changes only, untracked files ignored — via
    ``Repo.is_dirty(untracked_files=False)``.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return {}
    tags = {
        "git_commit": repo.head.commit.hexsha,
        "git_dirty": str(repo.is_dirty(untracked_files=False)).lower(),
    }
    if not repo.head.is_detached:
        tags["git_branch"] = repo.active_branch.name
    return tags


def flush_and_apply_tags() -> list[str]:
    """Flush queued spans, then apply any buffered tags via MLflow.

    Calls ``provider.force_flush()`` so the ``BatchSpanProcessor`` has shipped
    all queued spans to MLflow — ``set_trace_tag`` needs the ``TraceInfo``
    record to exist before tags can be attached. A per-tag try/except prevents
    one failed call from skipping the rest.

    Returns the list of request_ids that received tags, so callers can do
    follow-up per-trace work (e.g. linking prompts via
    ``link_prompt_to_traces``) without duplicating the buffer drain.
    """
    provider = otel_trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    request_ids: list[str] = []
    for request_id, tags in drain_tagged_trace_ids():
        request_ids.append(request_id)
        for key, value in tags.items():
            try:
                mlflow.set_trace_tag(trace_id=request_id, key=key, value=value)
            except Exception:
                logger.exception("Failed to set tag %s on trace %s", key, request_id)
    return request_ids


def link_prompt_to_traces(prompt_version: PromptVersion, trace_ids: list[str]) -> None:
    """Attach a prompt version to each trace via the MLflow registry linker.

    Populates the "Linked prompts" sidebar in the MLflow UI's trace view.
    The trace tag ``prompt_version`` (set elsewhere) makes traces filterable
    by version; this linker makes them clickable from the prompt's page and
    vice versa. Both are useful — they're independent features.

    Per-trace try/except so one failed link doesn't skip the rest.
    """
    client = MlflowClient()
    for trace_id in trace_ids:
        try:
            client.link_prompt_versions_to_trace(
                prompt_versions=[prompt_version], trace_id=trace_id
            )
        except Exception:
            logger.exception(
                "Failed to link prompt %s v%d to trace %s",
                prompt_version.name,
                prompt_version.version,
                trace_id,
            )


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

    See docs/internals/mlflow-session-view-bridge.md for a full explanation.
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

        # 1b. Mirror ADK tool spans to MLflow's TOOL conventions so the UI and
        #     judges that rely on ``include_tool_calls_in_conversation`` (e.g.
        #     our session-level groundedness judge) see tool I/O. ADK writes
        #     gcp.vertex.agent.tool_call_args/tool_response; MLflow looks for
        #     mlflow.spanInputs/Outputs plus mlflow.spanType == "TOOL".
        if attrs.get("gen_ai.operation.name") == "execute_tool":
            if "mlflow.spanType" not in attrs:
                span._attributes["mlflow.spanType"] = "TOOL"
            if (args := attrs.get("gcp.vertex.agent.tool_call_args")) and (
                "mlflow.spanInputs" not in attrs
            ):
                span._attributes["mlflow.spanInputs"] = args
            if (resp := attrs.get("gcp.vertex.agent.tool_response")) and (
                "mlflow.spanOutputs" not in attrs
            ):
                span._attributes["mlflow.spanOutputs"] = resp

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

        # 3. Root span: inject accumulated inputs/outputs and buffer tags.
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

            tags = trace_tags.get()
            if tags:
                # Use MLflow's canonical function rather than the literal
                # "tr-<hex>" formula so we adapt automatically if MLflow ever
                # flips to a different schema (a v4 form already exists at
                # mlflow.tracing.utils.generate_trace_id_v4_from_otel_trace_id).
                request_id = generate_mlflow_trace_id_from_otel_trace_id(trace_id)
                with _tagged_traces_lock:
                    _tagged_traces_buffer.append((request_id, dict(tags)))

    def on_start(self, span, parent_context=None) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_otlp_export(experiment_name: str) -> None:
    """Configure MLflow as the OTLP destination for ADK traces.

    Sets the MLflow experiment and writes the OTLP env vars that ADK's
    ``_setup_telemetry()`` reads when it builds the ``BatchSpanProcessor``.
    Must be called **before** ``get_fast_api_app`` (or any other call that
    creates the OTel provider).

    Args:
        experiment_name: MLflow experiment to create or activate.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
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
    provider = otel_trace.get_tracer_provider()
    if not isinstance(provider, SDKTracerProvider):
        return False

    with _lock:
        provider_id = id(provider)
        if provider_id in _configured_providers:
            return True
        _configured_providers.add(provider_id)

    provider.add_span_processor(_SessionIdSpanProcessor())
    return True


def add_file_sink(path: Path) -> None:
    """Register a span exporter that writes JSONL spans to ``path``.

    Each line is one compact JSON span — greppable, line-streamable.
    Independent of the MLflow OTLP exporter; both can be active at once.
    Creates an ``SDKTracerProvider`` if one isn't already installed (e.g.
    when running with ``--no-mlflow`` so no OTLP env vars were set).
    """
    provider = otel_trace.get_tracer_provider()
    if not isinstance(provider, SDKTracerProvider):
        provider = SDKTracerProvider()
        otel_trace.set_tracer_provider(provider)

    provider.add_span_processor(
        BatchSpanProcessor(
            ConsoleSpanExporter(
                out=path.open("w"),
                formatter=lambda span: span.to_json(indent=None) + "\n",
            )
        )
    )
