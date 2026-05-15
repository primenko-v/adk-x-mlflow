-include .env
export

MLFLOW_TRACKING_URI ?= http://localhost:5000
MLFLOW_EXPERIMENT ?= adk-demo

.PHONY: mlflow agent agent_x_mlflow format lint test

mlflow:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

agent:
	uv run adk web src/mlflow_adk/agents/

agent_x_mlflow:
	@curl -sf $(MLFLOW_TRACKING_URI)/health > /dev/null 2>&1 || \
		{ echo "MLflow server is not running. Start it first with: make mlflow"; exit 1; }
	uv run python -m mlflow_adk.server

test:
	uv run pytest tests/ -m unit -v

format:
	uv run ruff format .

lint:
	uv run ruff check .
