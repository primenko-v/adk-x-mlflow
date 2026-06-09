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
tracking). Runs and traces live in independent hierarchies — a Run is not a
parent of any trace.

That said, MLflow supports a *correlation* between Runs and traces. The
evaluation pipeline (`evaluate.py`) opens a Run per invocation and links it to
the traces it scored via an `eval_run_id` trace tag (it deliberately avoids the
`mlflow.sourceRun` metadata key — that path can't fire on OTLP ingest). See
[mlflow-runs-and-traces.md](mlflow-runs-and-traces.md) for the full account.

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

### Approach: a single `trace_tags` ContextVar

A `ContextVar[dict[str, str] | None]` named `trace_tags` carries an ambient set
of tags through any call stack. The span processor reads it as each root span
ends and buffers `(request_id, tags)`; the caller drains the buffer and applies
the tags via `mlflow.set_trace_tag` once spans have been flushed.

The same primitive serves both callers. The simulation loop sets per
scenario:

```python
{
    "source": "simulation",
    "agent_module": agent_module,
    "scenario": eval_case.eval_id,
    "mlflow.traceName": eval_case.eval_id,
    # plus git_commit, git_branch (when not detached), git_dirty when
    # the cwd is a git checkout
    **git_info(),
}
```

The interactive server sets, once for its lifetime:

```python
{
    "source": "interactive",
    **git_info(),  # same git tags as above
}
```

The server omits ``agent_module`` because a single ADK web process serves
multiple agents (chosen per request via URL), so a process-level tag would
be misleading. The simulation always runs against exactly one ``agent_module``
per ``run_simulation()`` invocation.

### A note on key names

MLflow's standard tag namespace (`mlflow.source.name`, `mlflow.source.git.commit`,
etc. — see `mlflow.utils.mlflow_tags`) is **deliberately not used here**.
The MLflow UI hides `mlflow.*`-prefixed tags from the trace-detail chip
display (treats them as system tags), so adopting that convention would make
our provenance invisible to anyone scanning a trace in the UI. The
namespaced versions are still set on Runs by MLflow's own machinery; we just
diverge for the trace-tag path where UI visibility matters.

One exception: ``mlflow.traceName`` is kept namespaced. It's not filtered as
a system tag — it's a *consumed* tag that MLflow uses to override the trace
name displayed in lists/sessions views. Renaming it would lose that UI
behavior.

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
via OTLP. The canonical mapping is a single function,
`mlflow.tracing.utils.generate_mlflow_trace_id_from_otel_trace_id`
(`mlflow/tracing/utils/__init__.py:455-465`):

```python
def generate_mlflow_trace_id_from_otel_trace_id(otel_trace_id: int) -> str:
    return TRACE_REQUEST_ID_PREFIX + encode_trace_id(otel_trace_id)
```

where `TRACE_REQUEST_ID_PREFIX = "tr-"` (`mlflow/tracing/constant.py:165`) and
`encode_trace_id` is a cached wrapper around
`opentelemetry.trace.format_trace_id` — the standard 32-character lowercase hex
representation of the 128-bit trace ID. The function is also the single
source of this mapping inside MLflow: every other site that needs it
(`mlflow/entities/span.py:418`, `mlflow/tracing/distributed/__init__.py:167`,
`generate_trace_id_v3`) calls it rather than re-implementing the formula.

We call this function directly rather than hard-coding `f"tr-{trace_id:032x}"`.
The function is public (no leading underscore, has a docstring), and MLflow
already ships a v4-schema sibling
(`generate_trace_id_v4_from_otel_trace_id`, same file, line 468) that produces
a different format (`trace:/<location>/<hex>`) — so a future MLflow release
may flip defaults. Calling the canonical function lets us inherit whichever
schema MLflow considers correct without our code needing to know about it.

### Putting it together

`tracing.py` exposes the ContextVar (`trace_tags`, `tracing.py:52`), a
thread-safe buffer, the low-level drain (`drain_tagged_trace_ids`,
`tracing.py:58`), and the `flush_and_apply_tags` wrapper (`tracing.py:95`)
that callers actually use. Inside `_SessionIdSpanProcessor.on_end`,
the root-span branch reads the ContextVar and, if tags are set, computes the
MLflow `request_id` via `generate_mlflow_trace_id_from_otel_trace_id(trace_id)`
and appends `(request_id, dict(tags))` to the buffer. A `dict(tags)` snapshot
is stored so later caller-side mutations cannot affect the buffered value.

The simulation loop wraps each scenario (`simulate.py:224-266`):

```python
# simulate.py loop
from mlflow_adk.tracing import flush_and_apply_tags, trace_tags

for eval_case in eval_set.eval_cases:
    single_case = EvalSet(eval_set_id=eval_set.eval_set_id, eval_cases=[eval_case])

    token = trace_tags.set(
        {
            **base_tags,                       # source="simulation", agent_module, git_*
            "scenario": eval_case.eval_id,
            "conversation_mode": "static" if eval_case.conversation else "scenario",
            "mlflow.traceName": eval_case.eval_id,
        }
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

    if experiment is not None:
        request_ids = flush_and_apply_tags()
        # request_ids is then used to link the prompt version to each trace
```

`flush_and_apply_tags()` (`tracing.py:95`) is the single helper that both the
simulation loop and the server use. It calls `provider.force_flush()` — which
drains the `BatchSpanProcessor` queue so every span has reached the MLflow
server before `set_trace_tag` is called (the `TraceInfo` record must exist
first) — then drains the tag buffer and applies each tag, returning the list
of tagged `request_id`s so the caller can do follow-up per-trace work (the
simulation uses them for `link_prompt_to_traces`). The trace-ID correlation is
exact: the IDs were captured directly from the spans as they ended, not
inferred from a query.

(The relevant flush is `TracerProvider.force_flush()` propagating to the
`BatchSpanProcessor`, not `_SessionIdSpanProcessor.force_flush`, which is a
no-op stub.)

`eval_case.eval_id` is the YAML filename stem (`temperature_basic`, etc.),
set in `_build_eval_case` (`simulate.py:97`/`:102`) as `eval_id=path.stem`.

### Server case

`server.py` runs as a long-lived process that handles interactive chat
requests. Because the server is always in one mode for its entire lifetime,
the ContextVar is set once before `uvicorn.run` and never reset
(`src/mlflow_adk/server.py:79-83`).

**Propagation through uvicorn.** `uvicorn.run()` calls `asyncio.run()`
internally, which launches the server coroutine as a `Task` copied from the
calling context — the one where `trace_tags` is already set. Uvicorn then
creates a new `Task` per connection/request inside that server task via
`asyncio.create_task()`. Each request task inherits the context, including the
`trace_tags` value. `on_end` runs synchronously inside `span.end()`, which is
called from within the request task, so every interactive trace correctly picks
up `{"mlflow.source.name": "interactive", ...}` without any per-request setup.

**No `reset()` needed.** The server sets `trace_tags` once and leaves it. Each
request task has its own copy of the context (by asyncio's copy-on-create
rule), so mutations inside one request task don't affect other tasks or the
root value. The tag is effectively immutable for the server's lifetime.

**Flush-then-tag via a background drain loop.** Unlike the simulation loop,
which has a natural per-scenario boundary, the server has no such hook. The
implementation registers a FastAPI lifecycle that runs an asyncio task in the
background (`src/mlflow_adk/server.py:42-72`):

- **Startup**: `_drain_loop` is spawned. Every 5 seconds it calls
  `provider.force_flush()` to drain the `BatchSpanProcessor` queue, then drains
  the tag buffer and applies each `(request_id, key, value)` triple via
  `mlflow.set_trace_tag`.
- **Shutdown**: the drain task is cancelled and awaited, then one final drain
  runs to catch any tags buffered between the last poll and SIGTERM. Uvicorn
  invokes shutdown hooks before exit, so Ctrl-C and graceful termination both
  trigger this path.

The 5-second cadence is a trade-off — interactive traces appear in MLflow
immediately, then gain their tags within the next polling interval. The drain
itself is cheap when idle (`force_flush` on an empty `BatchSpanProcessor` queue
returns near-instantly, and the buffer check is one lock + empty-list test), so
there is no benefit to dynamic event-driven signalling.

---

## Filtering by tag

```python
import mlflow

# All interactive traces
mlflow.search_traces(filter_string="tags.source = 'interactive'")

# All simulation traces
mlflow.search_traces(filter_string="tags.source = 'simulation'")

# All traces from a given commit
mlflow.search_traces(filter_string="tags.git_commit = '<sha>'")

# A specific scenario
mlflow.search_traces(filter_string="tags.scenario = 'temperature_basic'")

# Scoped to an experiment
mlflow.search_traces(
    experiment_names=["adk-sim"],
    filter_string="tags.scenario = 'temperature_basic'",
)
```

### Setting multiple tags at once

`mlflow.set_trace_tag` (and `MlflowClient.set_trace_tag`) set one key at a
time; there is no `set_trace_tags` plural method in MLflow 3.x. `flush_and_apply_tags()`
loops over the dict for you; if you ever need to do it by hand the shape is:

```python
for request_id, tags in drain_tagged_trace_ids():
    for key, value in tags.items():
        mlflow.set_trace_tag(request_id, key, value)
```
