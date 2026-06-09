# Evaluating conversations

Evaluation **scores traces that already exist** in MLflow — the ones produced
by `make simulate` or by interactive sessions — and lands all the results on a
single MLflow **Run**. It never drives the agent itself; simulation and scoring
are separate steps, so the same traces can be re-scored under different scorer
sets later.

For the data-model details (how the Run links back to the traces, why an
`eval_run_id` tag instead of `mlflow.sourceRun`), see
[mlflow-runs-and-traces.md](../internals/mlflow-runs-and-traces.md).

## The simulate → evaluate handoff

The two steps chain through a file of trace IDs:

```bash
make mlflow      # terminal 1 — must be running
make simulate    # produces traces; writes their IDs to .last_trace_ids.txt
make evaluate    # scores exactly those traces
```

`make simulate` runs every conversation under `simulations/conversations/`
(see [simulated-conversations.md](simulated-conversations.md)) and writes the
resulting trace IDs to `.last_trace_ids.txt` (`--write-trace-ids`). `make
evaluate` reads that file and scores those exact traces — no tag-filter
guesswork. If the file is absent, it falls back to a tag filter
(`--source simulation`).

## The two scoring layers

Both layers run inside one Run (`evaluate.py`), so their aggregates compare
across Runs.

### Turn-level — deterministic, free, no LLM

Plain Python scorers over every trace individually
(`scorers/performance.py`). Feedback attaches to each trace and shows inline in
the **Traces** view; aggregates (`mean`, `p90`, `max`) become Run metrics.

| Scorer | What it measures |
|---|---|
| `turn_latency_ms` | Wall-clock duration of the turn, in ms. |
| `turn_tokens` | Input + output LLM tokens for the turn (from MLflow's `trace.info.token_usage`). |

These are cheap and have no API cost — they read fields already on the trace.

### Session-level — LLM judges, one call per session

Whole-conversation judges run once per session (traces grouped by
`session.id`). Feedback attaches to the session and renders inline in the
**Chat Sessions** view; aggregates flow to the Run. Three groups:

- **Built-in MLflow conversation judges** (`judges/mlflow_builtin.py`):
  `ConversationCompleteness`, `ConversationalSafety`,
  `ConversationalRoleAdherence`, `ConversationalToolCallEfficiency`. These emit
  categorical (yes/no) ratings.
- **Custom judge** (`judges/mlflow_custom.py`): `session_groundedness` — checks
  every factual claim the assistant makes is supported by an earlier tool call
  or by something the user said, allowing multi-turn caching (a value fetched
  in turn 1 may be re-cited in turn 3 without re-calling the tool).
- **Numeric mirrors** (`make_numeric_mirrors`): 1/0 copies of three of the
  built-in judges. The Quality tab of the GenAI Overview only draws a
  moving-average *line* for numeric assessments — categorical ratings get a bar
  chart only — so the mirrors give the same judgement a trend line.

### The judge model

LLM judges use `settings.judge_model` (env `JUDGE_MODEL`), in MLflow's
`<provider>:/<model>` form, e.g. `gemini/gemini-2.5-flash` or
`openai:/gpt-4.1-mini`. If unset, MLflow defaults to `openai:/gpt-4.1-mini`,
which needs `OPENAI_API_KEY`. Set `JUDGE_MODEL` in `.env` to route judges
through a model you have credentials for.

Per-scorer judge failures don't raise — MLflow records them as error Feedback.
`evaluate.py` reads them back and logs a warning so they aren't silent in the
terminal; the stack traces live in the trace's **Assessments** tab in the UI.

## Selecting which traces to score

`evaluate.py` has two selection modes:

- **Explicit** — `--trace-ids <file>` (one ID per line). Those exact traces are
  scored; filter flags and dedup are ignored. This is the `make evaluate` path
  via `.last_trace_ids.txt`.
- **Filter-based** — no `--trace-ids`. Compose a `search_traces` filter from
  `--source`, `--prompt-version`, `--git-commit`, `--scenario`. Unless
  `--rescore` is passed, the filter also excludes traces that already carry an
  `eval_run_id` tag, so re-runs don't re-score (and re-pay for) the same data.

```bash
# Score everything from the simulation source, skipping already-scored traces
uv run python -m mlflow_adk.evaluate --experiment adk-sim --source simulation

# Re-score one scenario even if it was scored before
uv run python -m mlflow_adk.evaluate --experiment adk-sim \
    --scenario temperature_basic --rescore
```

## Registering custom judges (optional)

The custom judges run from `evaluate.py` regardless. Registering them versions
their instructions and unlocks continuous monitoring of live traces:

```bash
make register_judges   # registers judges from judges/mlflow_custom.py into adk-sim
```

See [registering-judges.md](registering-judges.md) for why and when to register
(and why batch scoring doesn't need it).

## Where results show up

- **Run page** — params (`filter_string`, `trace_count`, `session_count`, the
  scorer names) and metric aggregates. Reproducing or interpreting an eval
  needs nothing beyond this page. The Run also links to the prompt version it
  scored (when the scored traces share one version) and to its scored-trace
  results table.
- **Traces view** — per-turn Feedback (latency, tokens) inline on each trace.
  Filter to one eval's traces with `tags.eval_run_id = '<run_id>'`.
- **Chat Sessions view** — per-session judge Feedback inline on each session.

See [mlflow-trace-tags.md](../internals/mlflow-trace-tags.md) for the full set
of filterable tags.
