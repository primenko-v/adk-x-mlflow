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
make agent_x_mlflow

# Lint / format
make lint
make format
```

`make agent_x_mlflow` checks that the MLflow server is reachable (`/health`), then runs
`python -m mlflow_adk.server`, which configures the MLflow experiment and OTLP export
before starting the ADK web server. The plain `make agent` target runs vanilla `adk web`
without any tracing.

## Rules

- Use `uv` for all dependency management and script execution — no bare `pip` or `python` calls.
- `google-adk` is installed from the fork at `vendor/google-adk` (editable). Do not add it from PyPI. To update the fork: `git -C vendor/google-adk pull`.
- Format and lint with `ruff`. Run `ruff format .` then `ruff check .` before committing. Never run ruff on `vendor/` — it is third-party code.
- Python 3.13+. Use `src/` layout.
- **All imports at the top of the file.** No function-scoped imports, no `try/except ImportError` guards for "optional" dependencies. Every import the file uses goes in the module-level import block at the top, sorted by `ruff`. If you find yourself wanting a deferred import (to break a cycle, dodge a heavy dependency, or guard an optional one), restructure the modules instead.
- **Configuration via Pydantic settings** — never read env vars with `os.environ.get`. All config lives in `src/mlflow_adk/settings.py` as a `pydantic_settings.BaseSettings` subclass. This gives free `.env` loading, type coercion, validation, and a serialisable object you can log or pass around (`settings.model_dump()` / `settings.model_dump_json()`).
- **Docstrings and comments — overrides the global one-line rule.** Multi-line docstrings are fine when they document a non-obvious *why*: a subtle invariant, a counterintuitive behaviour, an explanation of a workaround. Roughly **under ~10 lines**. If you need more than that, the content is documentation, not a comment — put it in `docs/` and have the docstring link to it. Don't restate what the code obviously does (function name + signature + types already do that). Don't include usage examples that argparse `--help` or function signatures already convey.

## Development Methodology

**TDD (Test-Driven Development)**: Red-Green-Refactor cycle
1. Write a failing test first (red)
2. Write minimal code to pass the test (green)
3. Refactor while keeping tests passing

**KISS (Keep It Simple, Stupid)**: Prefer simple solutions. Don't add abstractions, operators, or features until actually needed.

**No re-exports**: Import from the actual module path, not from `__init__.py` re-exports. Keep `__init__.py` files minimal (docstring only). This makes imports explicit and traceable.

**Pydantic vs dataclass**: Use Pydantic `BaseModel` for types that cross a system boundary (e.g. API responses, YAML config, queue payloads, HTTP requests). Use `@dataclass` for internal data that never leaves the process.

## Testing

Tests are not an afterthought — write the test first (see TDD above).

**Framework**: pytest

**What to test**: meaningful behaviour — a workflow completing correctly, a tool returning the right data, a trace being emitted. Before writing a test, name a concrete bug it would catch that wouldn't surface on the first production run. If you can't, don't write it.

**What NOT to test** — recognise these anti-patterns and skip the test entirely:

- *Config pass-through to a library class.* `Foo(model=settings.x)` doesn't need a test asserting `result.model == settings.x` — that tests the library's pydantic field, not your code. Broken forwarding surfaces on the first real call.
- *Hardcoded-literal name assertions.* `assert scorer.name == "session_groundedness"` re-asserts a literal you wrote into the constructor one file over. Renames touch both files together; the test catches nothing.
- *"Returns fresh instances each call."* `assert a is not b` on a factory that returns a list literal tests Python, not your code.
- *Trivial pass-through wrappers.* For a function whose body is one `Constructor(**kw)` or one `library.do(...)`, the only meaningful test is a behavioural integration test (real or recorded LLM, real trace, etc.). Skip the unit test rather than fake one.

**Test structure**:
- One test file per module: `module.py` → `test_module.py`
- Shared fixtures in `conftest.py`
- `@pytest.mark.unit` — no external dependencies
- `@pytest.mark.integration` — requires real API/credentials

**Style**:
- One thing per test, keep it short
- Name: `test_<what>_<condition>_<expected>` e.g. `test_temperature_tool_unknown_city_returns_none`
- Arrange-Act-Assert, each section a few lines at most
- Mock external dependencies (APIs, LLMs), never mock the unit under test. Don't monkeypatch an internal collaborator to manufacture the assertion's expected value (e.g. patching a factory to return `["S1","S2","S3","S4"]` then asserting count `== 4`) — the mock dictated the outcome.
- Compare whole objects, not individual fields:
  ```python
  expected = Forecast(city="Berlin", temp_c=12)
  assert result == expected
  ```
