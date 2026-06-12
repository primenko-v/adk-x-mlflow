-include .env
export

MLFLOW_TRACKING_URI ?= http://localhost:5000
EXPERIMENT ?= DefaultSimulation
SIM_FILE_OUTPUT ?= traces

# Which simulation-list file picks the conversations to run:
# simulations/$(SIM_LIST_NAME).txt holds one glob pattern per line (matched against
# conversation filename stems). Default simulations/default.txt runs
# everything; `make simulate SIM_LIST_NAME=smoke` reads simulations/smoke.txt instead.
SIM_LIST_NAME ?= default

.PHONY: mlflow agent agent_x_mlflow simulate simulate_live simulate_to_file evaluate evaluate_live register_prompt register_judges freeze_prompt format lint test check-mlflow

mlflow:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

check-mlflow:
	@curl -sf $(MLFLOW_TRACKING_URI)/health > /dev/null 2>&1 || \
		{ echo "MLflow server is not running. Start it first with: make mlflow"; exit 1; }

agent:
	uv run adk web src/mlflow_adk/agents/

agent_x_mlflow: check-mlflow
	uv run python -m mlflow_adk.server --experiment adk-demo

simulate: check-mlflow
	uv run python -m mlflow_adk.simulate --experiment adk-sim --write-trace-ids .last_trace_ids.txt --simulation-list simulations/$(SIM_LIST_NAME).txt

simulate_live: GOOGLE_CLOUD_LOCATION = europe-west1
simulate_live: AGENT_MODEL = gemini-live-2.5-flash-native-audio
simulate_live: check-mlflow
	uv run python -m mlflow_adk.simulate --experiment adk-sim-live --write-trace-ids .last_trace_ids_live.txt --simulation-list simulations/$(SIM_LIST_NAME).txt

simulate_to_file:
	uv run python -m mlflow_adk.simulate --output-traces traces.jsonl

evaluate: check-mlflow
	@if [ -f .last_trace_ids.txt ]; then \
		echo "Scoring traces from .last_trace_ids.txt"; \
		uv run python -m mlflow_adk.evaluate --experiment adk-sim --trace-ids .last_trace_ids.txt; \
	else \
		echo "No .last_trace_ids.txt found — falling back to filter-based selection."; \
		uv run python -m mlflow_adk.evaluate --experiment adk-sim --source simulation; \
	fi

evaluate_live: check-mlflow
	@if [ -f .last_trace_ids_live.txt ]; then \
		echo "Scoring traces from .last_trace_ids_live.txt"; \
		uv run python -m mlflow_adk.evaluate --experiment adk-sim-live --trace-ids .last_trace_ids_live.txt; \
	else \
		echo "No .last_trace_ids_live.txt found — falling back to filter-based selection."; \
		uv run python -m mlflow_adk.evaluate --experiment adk-sim-live --source simulation; \
	fi

register_prompt: check-mlflow
	uv run python -c "from mlflow_adk.agents.simple_agent import prompts; prompts.register()"

register_judges: check-mlflow
	uv run python -c "from mlflow_adk.judges import mlflow_custom; mlflow_custom.register('adk-sim')"

freeze_prompt: check-mlflow
	uv run python -m mlflow_adk.freeze_prompt

test:
	uv run pytest tests/ -m unit -v

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix
