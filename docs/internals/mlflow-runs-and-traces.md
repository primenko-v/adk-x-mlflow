# Runs, traces, and evaluation

This document explains how MLflow **Runs** relate to the **traces** this
project produces, and how `evaluate.py` wires the two together. The
evaluation pipeline is implemented — see `src/mlflow_adk/evaluate.py`. For a
task-oriented walkthrough of *running* an evaluation, see
[evaluation.md](../guide/evaluation.md); this doc is the under-the-hood account of the
Run ⇆ trace data model the pipeline relies on.

## The two entities, briefly

**Trace** — produced by every agent invocation, whether interactive
(`server.py`) or simulated (`simulate.py`). One per turn. Mutable tags,
immutable metadata, hierarchical spans, optional session grouping. This is
what the OTLP pipeline lands in MLflow.

**Run** — MLflow's classic batch-level entity. Mutable tags, immutable
params, time-series metrics, attached artifacts. `evaluate.py` opens exactly
one Run per CLI invocation (`mlflow.start_run`, `evaluate.py:322`) and lands
all scorer aggregates on it.

A Run is **not** a parent of any trace. The two live in independent
hierarchies and only ever connect via explicit references that one carries to
the other. The rest of this document is about those references.

---

## How `evaluate.py` uses a Run

`run_evaluation()` (`evaluate.py:247`) reads *pre-existing* traces — produced
by an earlier `make simulate` or interactive session — and scores them under
one Run:

1. Selects traces (explicit trace-id file, or a `search_traces` tag filter).
2. Opens a Run (`mlflow.start_run`).
3. Logs the selection + scorer set as Run **params** (`mlflow.log_params`,
   `evaluate.py:323`): `filter_string`, `trace_count`, `session_count`,
   `turn_scorers`, `session_scorers_mlflow`. The Run page alone is enough to
   reproduce or interpret the evaluation.
4. Runs two scoring layers (both via `mlflow.genai.evaluate`, both inside the
   same Run so their aggregates compare across Runs):
   - **Turn-level** — deterministic Python scorers over a DataFrame of every
     trace (`TURN_SCORERS`, `evaluate.py:341`). Feedback attaches to each real
     trace; aggregates (`mean`, `p90`, `max`) become Run metrics.
   - **Session-level** — MLflow native conversation judges + our custom judges,
     one `evaluate` call per session (`_run_mlflow_conversation_scorers`,
     `evaluate.py:199`). Feedback attaches to each session record.
5. Reverse-links every scored trace back to the Run
   (`_tag_traces_with_eval_run_id`, `evaluate.py:233`).
6. Forward-links the prompt version to the Run when unambiguous
   (`link_prompt_version_to_run`, `evaluate.py:362`).

> This is the **score-pre-existing-traces** shape: the eval Run does not
> produce new traces, it scores ones that already exist. (MLflow also supports
> an inline `predict_fn` flavor that produces traces *during* the eval; this
> project deliberately separates simulation from scoring, so that flavor is
> not used — see [What we don't do](#what-we-dont-do).)

---

## The trace ⇆ Run linkage

There are two directions, and this project wires both.

### Run → traces (which traces did this Run score?)

`mlflow.genai.evaluate` writes per-trace results as a Run **artifact** (the
results table the Run page renders). That table records the scored trace IDs,
so MLflow's own machinery covers this direction with no help from us.

### Traces → Run (which Run scored this trace?)

After scoring, `evaluate.py` tags each trace with `eval_run_id = <run_id>`
(`evaluate.py:242`). That makes the reverse query a one-liner from the Traces
view:

```python
mlflow.search_traces(filter_string="tags.eval_run_id = '<run_id>'")
```

It also powers **dedup**: filter-based selection adds `tags.eval_run_id IS
NULL` (`build_filter_string(..., exclude_already_evaluated=True)`,
`evaluate.py:86`) so re-running an eval over the same filter doesn't re-score
— and re-pay for — traces a prior Run already touched. `--rescore` overrides
this.

### Why a tag, and why `eval_run_id` rather than `mlflow.sourceRun`

MLflow has a dedicated trace-metadata key for exactly this — `mlflow.sourceRun`
(`TraceMetadataKey.SOURCE_RUN`). For in-process tracing (`@mlflow.trace`,
`mlflow.start_span`), `BaseMlflowSpanProcessor` reads the active Run and writes
that metadata at trace-creation time, so the linkage is automatic.

**That auto-population never fires here.** Our spans are created by ADK's OTel
SDK in the client process, then exported over HTTP to MLflow's `/v1/traces`
OTLP endpoint. The MLflow *server* (a separate process) creates the trace; it
has no access to the client's `mlflow.active_run()`. The in-process processor
that would set `mlflow.sourceRun` is never involved.

And we cannot retrofit it: trace **metadata is immutable** by design — writable
only at creation. There is no `set_trace_metadata` API, and there should not
be one.

So we use a **tag** (`mlflow.set_trace_tag` works after the fact — the same
mechanism `flush_and_apply_tags()` uses for provenance; see
[mlflow-trace-tags.md](mlflow-trace-tags.md)). We deliberately name it
`eval_run_id`, *not* `mlflow.sourceRun`: the MLflow UI hides `mlflow.*`-prefixed
tags from the trace-detail chip display (treats them as system tags), so a
`mlflow.sourceRun` tag would be invisible to anyone scanning a trace. This is
the same UI-visibility rationale that governs all our provenance tags
([mlflow-trace-tags.md](mlflow-trace-tags.md)).

---

## Prompt versions and Runs

The agent's instruction is registered in the MLflow Prompt Registry (see
[mlflow-prompts.md](../guide/mlflow-prompts.md)) and exposed as `agent.PROMPT_VERSION`.
There are three prompt↔entity links, wired at two different places:

| Question | Backing data | Wired in |
|---|---|---|
| Which prompt version produced this trace? | `prompt_version` trace tag + trace-side prompt link | `simulate.py` (tag at :203, `link_prompt_to_traces` at :262) |
| Which prompt version did this eval Run score? | `link_prompt_version_to_run` | `evaluate.py:362` |
| Which traces did this eval Run score? | Run artifact / `eval_run_id` tag | `evaluate.py:242` |

The Run-side prompt link is only made when the scored traces share **one**
prompt version (`_unique_prompt_version`, `evaluate.py:91`). Pinning a single
prompt to a Run that scored multiple versions would be misleading, so a mixed
selection logs a warning and skips the link.

> `link_prompt_version_to_run`'s docstring warns it is not thread-safe. The
> evaluation loop is single-threaded, so this is fine; if eval is ever
> parallelised, serialise the linker calls.

---

## Sessions and Runs

Sessions are a UI grouping by `session.id` (see
[mlflow-trace-tags.md](mlflow-trace-tags.md) and
[mlflow-session-view-bridge.md](mlflow-session-view-bridge.md)). There is no
`mlflow.sourceRun` for sessions and no API to attach Run linkage to a session
directly.

The transitive linkage works fine in practice: every trace in a session
scored by a given Run carries the same `eval_run_id` tag, so "all sessions
this Run scored" is "distinct `session.id` across traces with that tag" — one
search call.

Session-level scoring itself groups traces by MLflow's native
`mlflow.trace.session` metadata (`_group_traces_by_session`, `evaluate.py:134`),
which the OTLP ingest derives from the `session.id` span attribute that
`_SessionIdSpanProcessor` writes. Traces with no session metadata are skipped
for session-level scoring — they can't belong to a multi-turn session by
definition.

<a name="what-we-dont-do"></a>

## What we don't do

- **Don't** set `mlflow.sourceRun` as trace metadata after the fact. It is
  immutable; there is no API; the `eval_run_id` tag is the supported path.
- **Don't** wrap `simulate.py` in `mlflow.start_run()`. Simulation produces
  traces, not evaluation results — there is nothing to correlate a Run with
  until scoring happens. Keeping simulation and evaluation as separate steps
  is what lets the same traces be re-scored under different scorer sets later.
- **Don't** introduce Runs for the interactive server. A long-lived process
  has no natural batch boundary; one Run for the server's lifetime is not
  useful.
- **Don't** log per-trace metrics on the Run via `log_metric` inside the
  scenario loop. Run metrics are scalar + step axis; they cannot model "one
  number per trace." Per-trace numbers live on the trace as Feedback (that is
  what the turn-level scorers produce), queryable via `search_traces`.
