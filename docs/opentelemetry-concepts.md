# OpenTelemetry: core concepts

## The core model

OpenTelemetry is a standard for recording what a program is doing and shipping
that record somewhere else (a database, a dashboard, a trace backend like
MLflow or Jaeger).

The unit of work is a **span** — a named, timed slice of execution. A span has
a start time, an end time, a bag of key-value **attributes**, and optionally a
**parent span**. Spans that share a `trace_id` form a tree: the root span at
the top, child spans nested beneath it. That tree is one **trace** — a complete
record of one request or one user turn.

---

## The provider and the global registry

To create spans you need a **Tracer**. To get a Tracer you ask a
**TracerProvider**. The TracerProvider is the central object that owns all the
configuration for how spans are built and what happens to them.

OTel ships a *no-op* TracerProvider by default — it creates spans that do
nothing. When you want real behaviour you create an `SDKTracerProvider` (from
`opentelemetry.sdk.trace`) and register it globally:

```python
trace.set_tracer_provider(my_provider)
```

After that, anywhere in the process:

```python
trace.get_tracer_provider()  # returns my_provider
```

This is a **process-wide singleton**. It doesn't matter which module calls it
or when — they all see the same object. `set_tracer_provider` is also
*idempotent after the first call*: if you call it a second time, the call is
silently ignored and the first provider stays.

---

## What happens when a span ends: the processor chain

The TracerProvider holds a list of **SpanProcessors**. A SpanProcessor is
middleware: it has two hooks — `on_start` (called when a span begins) and
`on_end` (called when a span finishes). You can have as many processors as you
want; the provider holds them in a `SynchronousMultiSpanProcessor` that fans
every `span.end()` call out to each processor in order.

```
span.end()
  └── SynchronousMultiSpanProcessor.on_end(span)
        ├── ADK's BatchSpanProcessor.on_end(span)   → queues span for OTLP export
        ├── ADK's DebugProcessor.on_end(span)        → populates the web UI
        └── _SessionIdSpanProcessor.on_end(span)     → our code
```

`add_span_processor` appends to that list. Because the list is mutable, you
can add processors at any point after the provider exists — the new processor
becomes part of every subsequent `span.end()` call.

---

## The timing puzzle

There are two separate setup concerns, and each has a different timing
requirement relative to when the OTel provider is created:

**Before the provider** — OTLP export destination must be configured first.
ADK's `_setup_telemetry()` reads env vars like
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` when it builds the `BatchSpanProcessor`.
Those vars must already be set at that point.  `setup_otlp_export()` in
`tracing.py` handles this.

**After the provider** — the span processor can only be appended to a provider
that exists.  `configure_tracing()` in `tracing.py` handles this.

### Sequence when using `server.py` (the Python launcher)

```
1. setup_otlp_export()
      → sets OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
      → sets OTEL_EXPORTER_OTLP_HEADERS (with experiment ID)
      → no provider yet; processor not registered

2. get_fast_api_app()
      → ADK calls _setup_telemetry() synchronously inside
      → reads the env vars from step 1
      → creates SDKTracerProvider with BatchSpanProcessor (OTLP), DebugProcessor, etc.
      → calls trace.set_tracer_provider(provider)           ← provider is now global

3. configure_tracing()
      → trace.get_tracer_provider() returns the provider from step 2
      → provider.add_span_processor(_SessionIdSpanProcessor())

4. uvicorn.run(app)
      → server starts; agent module is imported lazily on first request
      → simple_agent/__init__.py calls configure_tracing()
      → idempotency guard fires; processor already registered; no-op

5. First chat message
      → ADK creates invoke_agent span, runs agent, ends all spans
      → on_end fires for every processor, including ours
```

### Sequence when using `adk web` CLI directly

When the CLI launches the server, `setup_otlp_export()` is never called from
Python — the env vars come from the Makefile instead.  The processor is
registered later, via `configure_tracing()` in `simple_agent/__init__.py`,
which ADK's lazy agent loading triggers on the first request (always after
`_setup_telemetry()` has run).

---

## Why span attributes are still mutable after `span.end()`

When `span.end()` is called, the span is "ended" in the sense that its clock is
stopped and it can't be started again. But the Python object isn't frozen. Its
`_attributes` is a `BoundedAttributes` dict — just a dict with a size cap —
and Python dicts are always mutable.

The `BatchSpanProcessor` doesn't copy the span's data immediately. It pushes a
**reference** to the span object onto an internal queue, and a background thread
picks it up later to serialise and send it. So there is a window — between
`span.end()` and the background flush — where you can still write to
`span._attributes` and the export thread will see the new values. Our processor
runs inside `on_end`, which is called synchronously during `span.end()`, so we
are always inside that window.

---

## Putting it together

The reason the solution works without a custom runner is that OTel's design is
deliberately layered:

- **Tracers** are thin; they don't know what happens to spans.
- **The provider** owns the processor chain; it's a global singleton.
- **Processors** are open middleware: you can append one at any time.

`setup_otlp_export()` configures where spans go before the provider exists.
`configure_tracing()` attaches the processor after the provider exists.
Everything in between — creating the provider, building the export pipeline —
is ADK's job and we don't touch it.
