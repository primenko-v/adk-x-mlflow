# MLflow x Google ADK

Minimal project demonstrating MLflow tracing and observability for agents built with [Google ADK](https://google.github.io/adk-docs/). The agent runs on Gemini via Vertex AI. Traces are captured through ADK's native OpenTelemetry integration and exported to a local MLflow server, where they can be inspected and evaluated.

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Access to a Google Cloud project with Vertex AI enabled

### Install

```bash
git clone --recurse-submodules <repo>
cd adk-x-mlflow
uv sync --group dev
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` and set `GOOGLE_CLOUD_PROJECT`. Then authenticate with:

```bash
gcloud auth application-default login
```

## Running

Run the agent interactively via the ADK web UI (http://localhost:8000):

```bash
make agent
```

To also capture traces in MLflow, start the MLflow server first:

```bash
make mlflow        # terminal 1 — MLflow UI at http://localhost:5000
make agent_x_mlflow  # terminal 2 — agent + tracing
```

`make agent_x_mlflow` will refuse to start if the MLflow server is not reachable.
It runs `src/mlflow_adk/server.py` — a Python launcher that sets up the MLflow
experiment and OTLP export before starting the ADK web server, so tracing
configuration is explicit and reusable from other scripts (simulations, etc.).

### Simulation mode

Run conversations against the agent to produce traces:

```bash
make mlflow    # terminal 1 — must be running
make simulate  # terminal 2
```

Simulation traces land in a dedicated MLflow experiment (the `make simulate` target uses `adk-sim`; override with `--experiment NAME`) so they don't mix with interactive sessions.

Conversations are YAML files under `simulations/conversations/` (read recursively), in two kinds — chosen per file by its contents:

- **Scenario** (a `conversation_plan`) — an LLM plays the user and improvises each turn to follow the plan.
- **Static** (a `messages` list) — your exact messages are replayed verbatim, in order.

```yaml
# simulations/conversations/scenarios/curious_traveler.yaml
starting_prompt: "What's the weather like in London?"
conversation_plan: |
  - Ask about the temperature in London.
  - Ask which cities the assistant supports.
  - Stop once you have those answers.
```

See [docs/guide/simulated-conversations.md](docs/guide/simulated-conversations.md) for both formats. CLI flags:

```bash
uv run python -m mlflow_adk.simulate --agent my.agent.module --experiment my-exp

# Skip MLflow entirely and dump spans to a local JSONL file instead
# (omitting --experiment disables MLflow export):
uv run python -m mlflow_adk.simulate --output-traces traces.jsonl
```

`--experiment` and `--output-traces` are independent — you can run with both sinks active or just one. If neither is provided, the simulation refuses to start (no trace sink configured).

> **Important:** Unlike traditional MLflow logging, the ADK integration via OTel requires a running MLflow server with a SQL-based backend. File-based storage (`./mlruns`) does NOT support OpenTelemetry ingestion.

### Live API models

Two model configurations are supported, selected via `.env`:

**Standard (non-Live API)** — default in `.env.example`:
```
GOOGLE_CLOUD_LOCATION=global
AGENT_MODEL=gemini-3.1-flash-lite
```

**Live API** — required for audio/streaming models:
```
GOOGLE_CLOUD_LOCATION=europe-west1
AGENT_MODEL=gemini-live-2.5-flash-native-audio
```

The `global` endpoint does not support Live API models; use a regional endpoint (e.g. `europe-west1`) when running those.

`make simulate_live` applies the Live API location/model overrides for you and lands traces in a separate `adk-sim-live` experiment so they don't mix with standard runs:

```bash
make simulate_live   # europe-west1 + gemini-live-..., experiment adk-sim-live
make evaluate_live   # scores those live traces
```

### Evaluation

Score the traces a simulation produced (latency/token scorers per turn, LLM-judge scorers per conversation), landing the results on one MLflow Run:

```bash
make simulate   # writes the produced trace IDs to .last_trace_ids.txt
make evaluate   # scores exactly those traces
```

For Live API runs use `make simulate_live` / `make evaluate_live`, which target the `adk-sim-live` experiment instead.

See [docs/guide/evaluation.md](docs/guide/evaluation.md) for the scorer/judge catalog, trace selection, and judge-model configuration.

## google-adk: fork and editable install

This project uses a fork of `google-adk` checked in as a git submodule at `vendor/google-adk`. It is installed as an editable package, meaning Python imports the library directly from that directory — there is no copying or packaging step.

**Consequence:** any change you make inside `vendor/google-adk` (or pull from your fork) takes effect immediately. You do not need to reinstall anything.

The only time you need to re-run `uv sync` is when the fork's own dependencies change — i.e., its `pyproject.toml` adds or removes packages.

To pull the latest changes from the fork:

```bash
git -C vendor/google-adk pull
```

To pin to a specific commit:

```bash
git -C vendor/google-adk checkout <sha>
git add vendor/google-adk
git commit -m "bump google-adk fork to <sha>"
```

## Development

```bash
make format   # format with ruff
make lint     # lint with ruff

uv run pytest                 # all tests
uv run pytest -m unit         # unit tests only
uv run pytest -m integration  # requires GOOGLE_CLOUD_PROJECT + ADC
```
