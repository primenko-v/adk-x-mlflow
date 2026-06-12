# Registering judges

`make register_judges` publishes the custom judges in
`judges/mlflow_custom.py` to an experiment's scorer registry on the MLflow
server. This page explains why you'd do that — and why you mostly don't have
to.

## Registration is not required to run a judge

The custom judges already run during `make evaluate` **without** being
registered. `evaluate.py` instantiates them fresh (`make_custom_judges()`) and
hands them to `mlflow.genai.evaluate(...)`; the scorer's code runs in-process
against the batch of traces you selected. Nothing is read from the registry.

So registration is a *separate concern* from on-demand batch scoring. It buys
two things that batch scoring doesn't need.

## Why register, then

### 1. Versioning the judge's instructions

A judge is, in essence, a prompt — its `instructions` text
(`_GROUNDEDNESS_INSTRUCTIONS` in `mlflow_custom.py`). Registering the same
judge name with changed instructions creates a **new version**; MLflow keeps
the history and `list_scorers` returns the latest. This is the same idea as the
prompt registry ([mlflow-prompts.md](mlflow-prompts.md)), applied to judge
prompts: you get a tracked record of how a judge's definition evolved, so you
can tell which version of the rubric produced a given batch of assessments.

`register()` is idempotent on unchanged text and auto-bumps the version when
the instructions change — re-run `make register_judges` after editing a judge
and you get a new version with no extra ceremony.

### 2. Continuous monitoring of live traces

Batch scoring is pull-based: you point `evaluate.py` at traces that already
exist. A *registered* scorer can instead be **started** to score traces as they
arrive — the production-monitoring path:

```python
from mlflow.genai.scorers import ScorerSamplingConfig, get_scorer

judge = get_scorer(name="session_groundedness", experiment_id=exp_id)
judge.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))   # score 20% of new traces
# judge.update(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
# judge.stop()   # sets sample rate to 0; stays registered
```

`ScorerSamplingConfig` takes a `sample_rate` and an optional `filter_string`,
so you can score, say, 20% of interactive traces and leave simulation traces to
the batch path. Only a *registered* scorer can be started this way.

> This project doesn't wire up monitoring today — evaluation is the
> programmatic batch path in `evaluate.py`. Registration is what keeps that
> door open without a code change, and gives you judge versioning in the
> meantime.

## Batch scoring vs monitoring

| | Batch scoring | Monitoring |
|---|---|---|
| Triggered by | `make evaluate` (a manual run) | new traces arriving, sampled |
| Needs registration? | No | Yes |
| Judge source | fresh instances in `evaluate.py` | registry, via `start()` |
| Cost | one pass over the selected traces | scales with `sample_rate` × traffic |
| Used here | yes | not currently |

## How registration works

```bash
make register_judges   # registers judges from mlflow_custom.py into the adk-sim experiment
```

This calls `mlflow_custom.register("adk-sim")` (`mlflow_custom.py:163`), which
loops over `make_custom_judges()` and calls `judge.register(experiment_id=...)`
on each. Two things to note:

- **Only our custom judges are registered.** The built-in MLflow conversation
  judges (`ConversationCompleteness`, `ConversationalSafety`, …) ship with
  MLflow and need no publishing — `make_conversation_scorers()` instantiates
  them directly. Registration is only for judges we author.
- **The experiment is the namespace.** A registered scorer belongs to one
  experiment, so register into the same experiment whose traces you'll score or
  monitor (here, `adk-sim`).

## Managing registered judges

Registered judges appear in the experiment's **Judges / Scorers** view in the
MLflow UI, where aliases and activation are managed. Programmatically:

```python
from mlflow.genai.scorers import list_scorers, get_scorer, delete_scorer
```

`list_scorers(experiment_id=...)` returns the latest version of each registered
scorer; `get_scorer` fetches one by name; `delete_scorer` removes it.
