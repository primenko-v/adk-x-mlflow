# CLAUDE.md

## Project

Minimal demonstration of MLflow tracing and observability for Google ADK (Agent Development Kit) agents. The agent runs on Gemini via Vertex AI; traces are captured through ADK's native OpenTelemetry integration and exported to a local MLflow server. The project also explores MLflow-based evaluation of agent conversations (single-turn and multi-turn).

## Commands

```bash
# Clone repo (includes submodule)
git clone --recurse-submodules <repo>
# or if already cloned:
git submodule update --init

# Install dependencies
uv sync

# Copy and fill in .env
cp .env.example .env

# Start MLflow server (required before the agent; OTel traces land here)
make mlflow

# Run the agent interactively via ADK web UI (http://localhost:8000)
make agent

# Run the agent with MLflow tracing (checks MLflow is up first)
make agent_with_mlflow

# Lint / format
make lint
make format
```

`make agent_with_mlflow` checks that the MLflow server is reachable (`/health`), then sets
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` inline so `adk web` wires its OTel tracer to MLflow at startup.
The plain `make agent` target runs without any tracing.

## Rules

- Use `uv` for all dependency management and script execution — no bare `pip` or `python` calls.
- `google-adk` is installed from the fork at `vendor/google-adk` (editable). Do not add it from PyPI. To update the fork: `git -C vendor/google-adk pull`.
- Format and lint with `ruff`. Run `ruff format .` then `ruff check .` before committing.
- KISS: keep code minimal and direct. No abstractions until there is a clear reason for them.
- Python 3.13+. Use `src/` layout.
- **Configuration via Pydantic settings** — never read env vars with `os.environ.get`. All config lives in `src/mlflow_adk/settings.py` as a `pydantic_settings.BaseSettings` subclass. This gives free `.env` loading, type coercion, validation, and a serialisable object you can log or pass around (`settings.model_dump()` / `settings.model_dump_json()`).

## Testing

Follow TDD: write the test first, then write the code to make it pass. Tests are not an afterthought.

**Framework**: pytest

**What to test**: meaningful behaviour — a workflow completing correctly, a tool returning the right data, a trace being emitted. Do not write tests for trivial mechanics (object instantiation, attribute assignment, type checks). If a test does not catch a real bug, it should not exist.

**Test structure**:
- One test file per module: `module.py` → `test_module.py`
- Shared fixtures in `conftest.py`
- `@pytest.mark.unit` — no external dependencies
- `@pytest.mark.integration` — requires real API/credentials

**Style**:
- One thing per test, keep it short
- Name: `test_<what>_<condition>_<expected>` e.g. `test_temperature_tool_unknown_city_returns_none`
- Arrange-Act-Assert, each section a few lines at most
- Mock external dependencies (APIs, LLMs), never mock the unit under test
- Compare whole objects, not individual fields:
  ```python
  expected = Forecast(city="Berlin", temp_c=12)
  assert result == expected
  ```
