import json
import threading
from collections import defaultdict

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

from .agent import root_agent

__all__ = ["root_agent"]


def _extract_user_text(llm_request_json: str) -> str | None:
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
    try:
        response = json.loads(llm_response_json)
        content = response.get("content") or {}
        parts = content.get("parts", [])
        text = " ".join(p["text"] for p in parts if p.get("text"))
        return text or None
    except Exception:
        pass
    return None


class _SessionIdSpanProcessor(SpanProcessor):
    """Bridges ADK spans to MLflow's session/chat-session view.

    Does two things per span.end():
    1. Copies gen_ai.conversation.id → session.id so MLflow groups traces by session.
    2. Collects user input and assistant output from call_llm child spans and injects
       them onto the root invoke_agent span as mlflow.spanInputs / mlflow.spanOutputs
       so the Sessions view shows non-empty turns.
    """

    def __init__(self):
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
            # BoundedAttributes is mutable after span.end(); BatchSpanProcessor queues
            # a reference so this mutation is visible to the background export thread.
            span._attributes["session.id"] = session_id

        # 2. Accumulate user input / assistant output from call_llm child spans.
        #    ADK always sets gcp.vertex.agent.llm_request (may be '{}' if content
        #    capture is disabled), so we check for a non-empty value.
        llm_req = attrs.get("gcp.vertex.agent.llm_request", "{}")
        llm_res = attrs.get("gcp.vertex.agent.llm_response", "{}")
        if llm_req and llm_req != "{}":
            with self._lock:
                entry = self._pending[trace_id]
                # Take input from the first LLM call (initial user message).
                if entry["input"] is None:
                    entry["input"] = _extract_user_text(llm_req)
                # Always overwrite output to capture the final LLM response.
                out = _extract_assistant_text(llm_res)
                if out:
                    entry["output"] = out

        # 3. Root invoke_agent span: inject accumulated inputs/outputs.
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


# When `adk web` is used, the agent module is imported lazily on the first
# /list-apps or chat request — both of which happen after ADK's _setup_telemetry()
# has already set the global SDKTracerProvider (in get_fastapi_app → _setup_telemetry).
# add_span_processor appends to the existing chain, so ADK's own processors
# (debug-UI exporter, in-memory session tracker, OTLP exporter) are preserved.
# When running without OTLP export (make agent), no SDKTracerProvider is set and
# the isinstance guard makes this a no-op.
_provider = trace.get_tracer_provider()
if isinstance(_provider, SDKTracerProvider):
    _provider.add_span_processor(_SessionIdSpanProcessor())
