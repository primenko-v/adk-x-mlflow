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

Create a `.env` file in the project root:

```
GOOGLE_CLOUD_PROJECT=your-project
GOOGLE_CLOUD_LOCATION=europe-north1
GOOGLE_GENAI_USE_VERTEXAI=true
```

## Running

Start the MLflow server first (required — file-based storage does not support OpenTelemetry ingestion):

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then run the agent:

```bash
uv run python main.py
```

Open the MLflow UI at http://localhost:5000 to inspect traces.

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
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Run tests
uv run pytest
uv run pytest -m unit         # unit tests only
uv run pytest -m integration  # requires real credentials
```
