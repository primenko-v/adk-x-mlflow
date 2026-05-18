# MLflow trace tags

## The entity hierarchy

MLflow has two separate hierarchies that live under an experiment.

**Experiment tracking** (the classic path):
```
experiment
  └── run  ← has metrics, params, tags
```

**GenAI tracing** (the path used in this project):
```
experiment
  └── trace  ← has spans, request_metadata, tags
        └── span  ← has attributes (key-value pairs)
```

These two hierarchies are independent. Traces are not children of runs, and runs
have no visibility into traces. The two share an experiment, but that is the only
connection.

---

## Tags, metadata, and span attributes are three different things

| Concept | Where it lives | Mutable after creation? | Search filter prefix |
|---|---|---|---|
| **Trace tag** | `trace.info.tags` | Yes | `tags.key = 'value'` |
| **Trace request_metadata** | `trace.info.request_metadata` | No | `metadata.key = 'value'` |
| **Span attribute** | `span.attributes` | No (after export) | `span.<name> = 'value'` (only `name`, `type`, `status` are queryable) |

Prefixes are taken from `mlflow.utils.search_utils.SearchTraceUtils`
(`_ALTERNATE_IDENTIFIERS` and `_SUPPORTED_SPAN_ATTRIBUTES`). `metadata.*` is the
alias for `request_metadata.*`; `tags.*` is the alias for `tag.*`; `attributes.*`
and `trace.*` both resolve to trace-level attributes (`name`, `status`,
`start_time_ms`, …), not span attributes.

Tags are the only mutable, trace-level labels. They are the right place to attach
things like a scenario name, an environment, or a quality label that you might
want to update later.

MLflow does **not** map span attributes to trace tags automatically. Setting
`span._attributes["scenario"] = "temperature_basic"` makes that value available
as a span attribute in the trace detail view, but it is not a trace tag and will
not show up in the Tags panel or under `tags.*` filter syntax. Arbitrary span
attributes are also not directly queryable via `search_traces` — only the three
supported span attributes (`name`, `type`, `status`) are.

---

## Sessions

Sessions are **not a first-class entity in MLflow**. There is no `Session`
object, no `set_session_tag` API, and no session-level storage. A session is
purely a UI concept: the Chat Sessions view groups traces that share the same
`session.id` span attribute.

If you want session-level context on every trace in a session, set it as a trace
tag on each trace individually. Because tags are mutable, you can back-fill all
traces in a session at any time using `search_traces` + `set_trace_tag`.

---

## Runs

`mlflow.set_tag(key, value)` sets a tag on the **active run** (experiment
tracking). Runs have nothing to do with traces. Do not use run tags to annotate
tracing data.

---

> **Status — design proposal.** The sections below describe how trace tagging
> *should* be wired in this project. As of writing, `src/mlflow_adk/tracing.py`,
> `src/mlflow_adk/simulate.py`, and `src/mlflow_adk/server.py` do not yet
> implement this. The sections above (entity hierarchy, tags vs. metadata vs.
> attributes, sessions, runs) describe MLflow as it is today.

## How to set trace tags in this project

### Why `mlflow.update_current_trace()` does not apply here

The MLflow docs show:

```python
@mlflow.trace
def my_func(x):
    mlflow.update_current_trace(tags={"scenario": "temperature_basic"})
```

This works when MLflow's own tracer creates the trace. Here ADK drives the OTel
provider and exports spans via OTLP. MLflow's trace context is never activated,
so `update_current_trace()` has nothing to attach to.

### The OTLP path

ADK exports spans to MLflow's `/v1/traces` OTLP endpoint via a
`BatchSpanProcessor`. The server ingests each span and stores it. When the root
span arrives, MLflow creates the `TraceInfo` record with `tags={}`. Tags can
only be added after that record exists.

### Approach: a single `trace_tags` ContextVar

A `ContextVar[dict[str, str] | None]` named `trace_tags` carries an ambient set
of tags through any call stack. The span processor reads it as each root span
ends and buffers `(request_id, tags)`; the caller drains the buffer and applies
the tags via `mlflow.set_trace_tag` once spans have been flushed.

The same primitive serves both callers: the simulation loop sets
`{"scenario": eval_case.eval_id, "source": "simulation"}` per scenario; the
interactive server sets `{"source": "interactive"}` once for its lifetime.

**Why ContextVar works here**

`ContextVar` carries ambient state through a call stack without threading it
through every function signature. In asyncio, each `Task` created via
`asyncio.create_task()` receives a copy of the context that existed at creation
time — so child tasks inherit the value. A direct `await` (no new task) runs in
the same context as the caller.

`_SessionIdSpanProcessor.on_end` is called synchronously during `span.end()`,
which fires from within the ADK runner. Because the runner runs inside the same
task as the simulation loop (or in a child task that inherits the context), the
ContextVar value is always visible in `on_end`.

The `try/finally` reset pattern is essential: `set()` returns a token that
records the previous value; `reset(token)` restores it. This guarantees the
ContextVar is cleaned up even if the body raises, preventing stale tags from
leaking into the next iteration.

**Why this is safe for the live path**

The live path creates an internal task via
`asyncio.create_task(self._consume_events())` inside `_LiveSession.__aenter__`
(`vendor/google-adk/src/google/adk/evaluation/evaluation_generator.py:106`).
That task is created while the ContextVar is already set, so it inherits the
correct value into its own copy. Critically, `_LiveSession.__aexit__` awaits
the consume-task before returning
(`vendor/google-adk/src/google/adk/evaluation/evaluation_generator.py:230-240`,
specifically `await asyncio.wait_for(self.consume_task, timeout=30)` at line
234) — so the `finally` reset only runs after all spans have been produced.

**The OTel trace ID → MLflow request ID mapping**

MLflow derives its `request_id` from the OTel `trace_id` when ingesting spans
via OTLP. The formula lives at
`mlflow/tracing/utils/__init__.py:455-465`:

```python
def generate_mlflow_trace_id_from_otel_trace_id(otel_trace_id: int) -> str:
    return TRACE_REQUEST_ID_PREFIX + encode_trace_id(otel_trace_id)
```

where `TRACE_REQUEST_ID_PREFIX = "tr-"` (`mlflow/tracing/constant.py:165`) and
`encode_trace_id` is a cached wrapper around
`opentelemetry.trace.format_trace_id` — the standard 32-character lowercase hex
representation of the 128-bit trace ID. The MLflow `request_id` for any span
seen in `on_end` can therefore be computed directly from
`span.context.trace_id` as `f"tr-{trace_id:032x}"`, with no search or timestamp
required.

> **Caveat —** this formula is an internal MLflow implementation detail, not a
> public API. The practical risk of it changing is low (it is built on the OTel
> standard trace ID format and has been stable since MLflow tracing was
> introduced), but it should be noted as a coupling point.

### Putting it together

`tracing.py` exposes the ContextVar and a buffer-drain function:

```python
# tracing.py additions (sketch)
from contextvars import ContextVar

trace_tags: ContextVar[dict[str, str] | None] = ContextVar(
    "trace_tags", default=None
)

# Inside _SessionIdSpanProcessor.on_end, after the existing root-span block:
if span.parent is None:
    tags = trace_tags.get()
    if tags:
        request_id = f"tr-{span.context.trace_id:032x}"
        # store (request_id, dict(tags)) in a thread-safe buffer

# New public function:
def drain_tagged_trace_ids() -> list[tuple[str, dict[str, str]]]:
    # pop and return all buffered entries
```

The simulation loop wraps each scenario:

```python
# simulate.py loop
from mlflow_adk.tracing import trace_tags, drain_tagged_trace_ids
from opentelemetry import trace as otel_trace
import mlflow

for eval_case in eval_set.eval_cases:
    single_case = EvalSet(
        eval_set_id=eval_set.eval_set_id, eval_cases=[eval_case]
    )

    token = trace_tags.set(
        {"scenario": eval_case.eval_id, "source": "simulation"}
    )
    try:
        await EvaluationGenerator.generate_responses(
            eval_set=single_case,
            agent_module_path=agent_module,
            repeat_num=1,
            user_simulator_config=user_simulator_config,
        )
    finally:
        trace_tags.reset(token)

    if mlflow_enabled:
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()

        for request_id, tags in drain_tagged_trace_ids():
            for key, value in tags.items():
                mlflow.set_trace_tag(
                    trace_id=request_id, key=key, value=value
                )
```

`force_flush()` is still required: it drains the `BatchSpanProcessor` queue so
all spans have arrived at the MLflow server before `set_trace_tag` is called.
(The relevant flush is `TracerProvider.force_flush()` propagating to the
`BatchSpanProcessor`, not `_SessionIdSpanProcessor.force_flush` —
the latter is a no-op stub at `src/mlflow_adk/tracing.py:137`.) The trace ID
correlation, however, is now exact — the IDs were captured directly from the
spans as they ended, not inferred from a query.

`eval_case.eval_id` is the YAML filename stem (`temperature_basic`, etc.) —
already set in `load_eval_set` (`src/mlflow_adk/simulate.py:38`) as
`EvalCase(eval_id=path.stem, ...)`.

### Server case

`server.py` runs as a long-lived process that handles interactive chat requests.
Because the server is always in one mode for its entire lifetime, the setup is
a single line:

```python
# server.py — set once before uvicorn starts, never reset
trace_tags.set({"source": "interactive"})
uvicorn.run(app, host="127.0.0.1", port=port)
```

**Propagation through uvicorn.** `uvicorn.run()` calls `asyncio.run()`
internally, which launches the server coroutine as a `Task` copied from the
calling context — the one where `trace_tags` is already set. Uvicorn then
creates a new `Task` per connection/request inside that server task via
`asyncio.create_task()`. Each request task inherits the context, including the
`trace_tags` value. `on_end` runs synchronously inside `span.end()`, which is
called from within the request task, so every interactive trace correctly picks
up `{"source": "interactive"}` without any per-request setup.

**No `reset()` needed.** The server sets `trace_tags` once and leaves it. Each
request task has its own copy of the context (by asyncio's copy-on-create
rule), so mutations inside one request task don't affect other tasks or the
root value. The tag is effectively immutable for the server's lifetime.

The server still needs the same flush-then-tag step as the simulation loop, but
typically driven from a background task or a per-request hook — exact placement
is left for the implementation.

---

## Filtering by tag

```python
import mlflow

# All interactive traces
mlflow.search_traces(filter_string="tags.source = 'interactive'")

# All simulation traces
mlflow.search_traces(filter_string="tags.source = 'simulation'")

# A specific scenario
mlflow.search_traces(filter_string="tags.scenario = 'temperature_basic'")

# Scoped to an experiment
mlflow.search_traces(
    experiment_names=["adk-simulation"],
    filter_string="tags.scenario = 'temperature_basic'",
)
```

### Setting multiple tags at once

`mlflow.set_trace_tag` (and `MlflowClient.set_trace_tag`) set one key at a
time; there is no `set_trace_tags` plural method in MLflow 3.x. Loop over the
dict:

```python
from mlflow import MlflowClient

client = MlflowClient()
for request_id, tags in drain_tagged_trace_ids():
    for key, value in tags.items():
        client.set_trace_tag(request_id, key, value)
```
