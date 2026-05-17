-include .env
export

MLFLOW_TRACKING_URI ?= http://localhost:5000
EXPERIMENT ?= DefaultSimulation
SIM_FILE_OUTPUT ?= traces

.PHONY: mlflow agent agent_x_mlflow simulate simulate_live simulate_to_file format lint test check-mlflow

mlflow:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

check-mlflow:
	@curl -sf $(MLFLOW_TRACKING_URI)/health > /dev/null 2>&1 || \
		{ echo "MLflow server is not running. Start it first with: make mlflow"; exit 1; }

agent:
	uv run adk web src/mlflow_adk/agents/

agent_x_mlflow: check-mlflow
	uv run python -m mlflow_adk.server

simulate: GOOGLE_CLOUD_LOCATION = global
simulate: AGENT_MODEL = gemini-3.1-flash-lite
simulate: check-mlflow
	uv run python -m mlflow_adk.simulate --experiment "$(EXPERIMENT)"

simulate_live: GOOGLE_CLOUD_LOCATION = europe-west1
simulate_live: AGENT_MODEL = gemini-live-2.5-flash-native-audio
simulate_live: check-mlflow
	uv run python -m mlflow_adk.simulate --experiment "$(EXPERIMENT)"

simulate_to_file:
	uv run python -m mlflow_adk.simulate --no-mlflow --output-traces "$(SIM_FILE_OUTPUT).jsonl"

test:
	uv run pytest tests/ -m unit -v

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix
