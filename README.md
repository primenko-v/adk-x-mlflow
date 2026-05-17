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

Run a scripted multi-turn conversation against the agent using an LLM-backed user simulator:

```bash
make mlflow    # terminal 1 — must be running
make simulate  # terminal 2
```

Simulation traces land in a dedicated MLflow experiment (`adk-simulation` by default, overridable with `--experiment`) so they don't mix with interactive sessions.

Scenarios are YAML files in `simulations/scenarios/`. Each file becomes one eval case:

```yaml
starting_prompt: "What's the weather like in London?"
conversation_plan: |
  - Ask about the temperature in London.
  - Ask which cities the assistant supports.
  - Ask the temperature in a city the assistant doesn't support.
  - Stop once you have those three answers.
```

The user simulator is LLM-backed (`gemini-2.5-flash` via ADK defaults) and drives the conversation autonomously according to the plan. CLI flags:

```bash
uv run python -m mlflow_adk.simulate --agent my.agent.module --experiment my-exp

# Skip MLflow entirely and dump spans to a local JSONL file instead:
uv run python -m mlflow_adk.simulate --no-mlflow --output-traces traces.jsonl
```

`--mlflow` (default on) and `--output-traces` are independent — you can run with both sinks active, just one, or neither.

> **Important:** Unlike traditional MLflow logging, the ADK integration via OTel requires a running MLflow server with a SQL-based backend. File-based storage (`./mlruns`) does NOT support OpenTelemetry ingestion.

## Model configuration

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
