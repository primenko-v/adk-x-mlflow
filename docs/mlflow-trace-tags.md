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

| Concept | Where it lives | Mutable after creation? | Searchable via |
|---|---|---|---|
| **Trace tag** | `trace.info.tags` | Yes | `tags.key = 'value'` |
| **Trace request_metadata** | `trace.info.request_metadata` | No | `attributes.key = 'value'` |
| **Span attribute** | `span.attributes` | No (after export) | `attributes.key = 'value'` |

Tags are the only mutable, trace-level labels. They are the right place to attach
things like a scenario name, an environment, or a quality label that you might
want to update later.

MLflow does **not** map span attributes to trace tags automatically. Setting
`span._attributes["scenario"] = "temperature_basic"` makes that value available
as a span attribute in the trace detail view and queryable via
`attributes.scenario = 'temperature_basic'`, but it is not a trace tag and will
not show up in the Tags panel or under `tags.*` filter syntax.

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

### Approach: ContextVar + span processor

The clean approach uses a `ContextVar` to signal the current scenario to
`_SessionIdSpanProcessor`, which captures the exact OTel trace IDs as they are
produced. This avoids querying MLflow by timestamp entirely.

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
ContextVar is cleaned up even if `generate_responses` raises, preventing a stale
scenario name from leaking into the next iteration.

**Why this is safe for the live path**

The live path creates an internal task via `asyncio.create_task(_consume_events)`
inside `_LiveSession.__aenter__`. That task is created while the ContextVar is
already set, so it inherits the correct value into its own copy. Critically,
`_LiveSession.__aexit__` calls `await asyncio.wait_for(consume_task, ...)` —
the task is fully awaited before `generate_responses` returns, so the `finally`
reset only runs after all spans have been produced.

**The OTel trace ID → MLflow request ID mapping**

MLflow derives its `request_id` from the OTel `trace_id` when ingesting spans
via OTLP:

```
# mlflow/entities/span.py
return "tr-" + encode_trace_id(otel_trace_id)
```

where `encode_trace_id` is `opentelemetry.trace.format_trace_id` — the standard
32-character lowercase hex representation of the 128-bit trace ID. This means
the MLflow `request_id` for any span seen in `on_end` can be computed directly
from `span.context.trace_id`, with no search or timestamp required.

This formula is an internal MLflow implementation detail, not a public API.
The practical risk of it changing is low (it is built on the OTel standard
trace ID format and has been stable since MLflow tracing was introduced), but
it should be noted as a coupling point.

**Putting it together**

`tracing.py` exposes a `ContextVar` and a buffer-drain function. The simulation
loop sets the var, runs the scenario, resets the var, then flushes and tags:

```python
# tracing.py additions (sketch)
from contextvars import ContextVar

current_scenario: ContextVar[str | None] = ContextVar("current_scenario", default=None)

# Inside _SessionIdSpanProcessor.on_end, after the existing root-span block:
if span.parent is None:
    scenario = current_scenario.get()
    if scenario is not None:
        request_id = f"tr-{span.context.trace_id:032x}"
        # store (request_id → scenario) in a thread-safe buffer

# New public function:
def drain_scenario_trace_ids(scenario: str) -> list[str]:
    # pop and return all request_ids collected for this scenario
```

```python
# simulate.py loop
from mlflow_adk.tracing import current_scenario, drain_scenario_trace_ids
from opentelemetry import trace as otel_trace
import mlflow

for eval_case in eval_set.eval_cases:
    single_case = EvalSet(eval_set_id=eval_set.eval_set_id, eval_cases=[eval_case])

    token = current_scenario.set(eval_case.eval_id)
    try:
        result = await EvaluationGenerator.generate_responses(
            eval_set=single_case,
            agent_module_path=agent_module,
            repeat_num=1,
            user_simulator_config=user_simulator_config,
        )
    finally:
        current_scenario.reset(token)

    if mlflow_enabled:
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()

        for request_id in drain_scenario_trace_ids(eval_case.eval_id):
            mlflow.set_trace_tag(trace_id=request_id, key="scenario", value=eval_case.eval_id)
```

`force_flush()` is still required: it drains the `BatchSpanProcessor` queue so
all spans have arrived at the MLflow server before `set_trace_tag` is called.
The trace ID correlation, however, is now exact — the IDs were captured directly
from the spans as they ended, not inferred from a query.

`eval_case.eval_id` is the YAML filename stem (`temperature_basic`, etc.) —
already set in `load_eval_set` as `EvalCase(eval_id=path.stem, ...)`.

---

## Tagging server traces (interactive mode)

`server.py` runs as a long-lived process that handles interactive chat requests.
Its traces can be tagged to distinguish them from simulation traces using the
same ContextVar mechanism — but with a simpler setup, because the server is
always in one mode for its entire lifetime.

### Generalising the ContextVar

`current_scenario` is too narrow a name once both callers are involved. The
right abstraction is a `trace_tags: ContextVar[dict[str, str] | None]` that
holds an arbitrary set of tags. Both callers write to it with different keys:

```python
# tracing.py
trace_tags: ContextVar[dict[str, str] | None] = ContextVar("trace_tags", default=None)
```

```python
# server.py — set once before uvicorn starts, never reset
trace_tags.set({"source": "interactive"})
uvicorn.run(app, host="127.0.0.1", port=port)
```

```python
# simulate.py — set and reset per scenario
token = trace_tags.set({"scenario": eval_case.eval_id, "source": "simulation"})
try:
    result = await EvaluationGenerator.generate_responses(...)
finally:
    trace_tags.reset(token)
```

The span processor reads the dict from the ContextVar in `on_end` and applies
all keys when building the buffer entry. No other changes are needed.

### How propagation works through uvicorn

`uvicorn.run()` calls `asyncio.run()` internally, which launches the server
coroutine as a `Task` copied from the calling context — the one where
`trace_tags` is already set. Uvicorn then creates a new `Task` per
connection/request inside that server task via `asyncio.create_task()`. Each
request task inherits the context, including the `trace_tags` value.

`on_end` runs synchronously inside `span.end()`, which is called from within
the request task. So every interactive trace correctly picks up
`{"source": "interactive"}` without any per-request setup.

### No `reset()` needed for the server

The server sets `trace_tags` once and leaves it. Each request task has its own
copy of the context (by asyncio's copy-on-create rule), so mutations inside one
request task don't affect other tasks or the root value. The tag is effectively
immutable for the server's lifetime.

### Filtering by source after the fact

```python
# All interactive traces
mlflow.search_traces(filter_string="tags.source = 'interactive'")

# All simulation traces
mlflow.search_traces(filter_string="tags.source = 'simulation'")

# A specific scenario
mlflow.search_traces(filter_string="tags.scenario = 'temperature_basic'")
```

---

## Filtering by tag after the fact

```python
traces = mlflow.search_traces(
    experiment_names=["adk-simulation"],
    filter_string="tags.scenario = 'temperature_basic'",
)
```

### Setting multiple tags at once

`mlflow.set_trace_tag` sets one key at a time. To set several, use the
underlying client method:

```python
from mlflow import MlflowClient

client = MlflowClient()
for request_id in drain_scenario_trace_ids(eval_case.eval_id):
    client.set_trace_tags(request_id, {"scenario": eval_case.eval_id, "env": "local"})
```

`set_trace_tags` is a thin wrapper that calls `set_trace_tag` per key — see
`mlflow/tracking/_tracking_service/client.py`.
