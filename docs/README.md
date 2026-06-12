# Documentation

## Guide — using the project

- [simulated-conversations.md](guide/simulated-conversations.md) — write
  conversation files (scenario and static) and run `make simulate`.
- [mlflow-prompts.md](guide/mlflow-prompts.md) — the prompt workflow: edit,
  register, alias, freeze.
- [evaluation.md](guide/evaluation.md) — score traces with `make evaluate`; the
  turn-level and session-level scorers and judges.
- [registering-judges.md](guide/registering-judges.md) — what `make
  register_judges` does, why register a judge (versioning + monitoring), and why
  batch scoring doesn't need it.

## Internals — how it works under the hood

- [opentelemetry-concepts.md](internals/opentelemetry-concepts.md) — OTel
  primer: spans, providers, the processor chain.
- [mlflow-session-view-bridge.md](internals/mlflow-session-view-bridge.md) — the
  custom `SpanProcessor` that makes ADK spans render in MLflow's Chat Sessions
  view.
- [mlflow-trace-tags.md](internals/mlflow-trace-tags.md) — tags vs metadata vs
  span attributes; the `trace_tags` ContextVar mechanism; filtering traces.
- [mlflow-runs-and-traces.md](internals/mlflow-runs-and-traces.md) — how
  evaluation Runs link to the traces they score.
