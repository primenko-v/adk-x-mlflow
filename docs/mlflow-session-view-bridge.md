# MLflow Session View Bridge

`src/mlflow_adk/tracing.py` contains a custom OTel `SpanProcessor` that
bridges the gap between ADK's span schema and what MLflow needs to render the
**Chat Sessions** view correctly, plus two setup functions that wire the whole
pipeline together.

---

## The two problems it solves

### 1. Traces not grouped into sessions

MLflow's `log_spans` extracts a session ID from the OTel standard attribute
`session.id` ([OTel session semconv][otel-session]):

```python
# mlflow/store/tracking/sqlalchemy_store.py
span_session_id := span_attributes.get("session.id")
```

ADK emits the session ID under a different key: `gen_ai.conversation.id`. The
two names never match, so every trace lands as a standalone item — no grouping.

### 2. Session turns show empty Inputs / Outputs

MLflow's session view reads each turn's user text and agent reply from
`mlflow.spanInputs` / `mlflow.spanOutputs` on the **root span** of each trace.
Those attributes are populated by MLflow's translation layer, which maps
`gcp.vertex.agent.llm_request` → `mlflow.spanInputs` and
`gcp.vertex.agent.llm_response` → `mlflow.spanOutputs`.

The problem: ADK only sets `gcp.vertex.agent.llm_request/response` on the
**`call_llm` child spans**, not on the root `invoke_agent` span. So the root
span has no content, and the turn rows show blank.

---

## The span hierarchy (per user turn)

```
invoke_agent simple_agent          ← root span; only has session/agent metadata
  └── call_llm (LLM #1)           ← has llm_request (full history) + llm_response
        └── execute_tool foo       ← has tool_call_args + tool_response
  └── call_llm (LLM #2)           ← final reply after tool result
```

OTel guarantees children end before parents, so `call_llm` spans always fire
`on_end` before `invoke_agent` does.

---

## Setup: two functions with distinct responsibilities

`tracing.py` exposes two functions that must be called in order around the
moment the OTel provider is created:

### `setup_otlp_export(experiment_name?)`

Call this **before** anything that creates the OTel provider.  It:
- Sets the MLflow tracking URI and creates/activates the experiment.
- Writes `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` and
  `OTEL_EXPORTER_OTLP_HEADERS` (with the experiment ID) using `setdefault`,
  so the first caller wins and subsequent calls are safe.

ADK's `_setup_telemetry()` reads these env vars when it builds its
`BatchSpanProcessor`, so they must exist before that call.

### `configure_tracing()`

Call this **after** the OTel provider exists.  It appends
`_SessionIdSpanProcessor` to the provider's processor chain, and is
idempotent — calling it more than once on the same provider is safe.

### Startup sequence in `server.py`

```python
setup_otlp_export()                        # env vars written; no provider yet
app = get_fast_api_app(agents_dir=..., web=True)  # _setup_telemetry() runs here
configure_tracing()                        # provider exists; processor registered
uvicorn.run(app, ...)
```

The same pattern applies to any script (simulation runner, custom app):
`setup_otlp_export()` → create provider → `configure_tracing()`.

### Fallback for `adk web` CLI

`simple_agent/__init__.py` calls `configure_tracing()` at import time as a
fallback for the plain `adk web` CLI path.  In that case `setup_otlp_export()`
is not called from Python — the env vars come from the Makefile instead.  The
idempotency guard ensures the processor is never registered twice.

---

## How `_SessionIdSpanProcessor` works

Every `span.end()` call fans out synchronously through all registered
processors. Our `on_end` runs three steps:

**Step 1 — fix session grouping**
```python
span._attributes["session.id"] = session_id   # copy from gen_ai.conversation.id
```
`BoundedAttributes` (the underlying dict) is mutable after `span.end()`.
`BatchSpanProcessor` queues a *reference*, so this write is visible when the
background export thread flushes the batch.

**Step 2 — collect content from `call_llm` spans**

Any span with a non-empty `gcp.vertex.agent.llm_request` is a `call_llm` span.
We extract:
- **input**: the last `user`-role message from the first `call_llm` request
  (i.e. what the user just typed, not the full conversation history).
- **output**: the `model`-role text from the *most recent* `call_llm` response
  (always overwritten, so tool-use turns capture the final reply, not the
  intermediate "call this tool" message).

Both are stored in `self._pending[trace_id]`.

**Step 3 — inject onto the root span**

When a span with `span.parent is None` ends (the root `invoke_agent` span), we
pop the stored data and write:
```python
span._attributes["mlflow.spanInputs"]  = '[{"role":"user","content":"..."}]'
span._attributes["mlflow.spanOutputs"] = '[{"role":"assistant","content":"..."}]'
```
The `[{"role", "content"}]` format matches what MLflow's chat UI expects.  The
`_pending` entry is removed on pop, so there is no memory leak across sessions.

---

## Why not a custom runner?

A custom runner would require reimplementing ADK's session management, HTTP
API, and web UI — a large surface. Injecting a single `SpanProcessor` into the
existing `SDKTracerProvider` achieves the same observability goal with ~100
lines of code and zero changes to ADK internals.

[otel-session]: https://opentelemetry.io/docs/specs/semconv/registry/attributes/session/
