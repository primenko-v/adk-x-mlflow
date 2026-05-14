-include .env
export

MLFLOW_TRACKING_URI ?= http://localhost:5000
MLFLOW_EXPERIMENT ?= adk-demo

.PHONY: mlflow agent agent_x_mlflow format lint

mlflow:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

agent:
	uv run adk web src/mlflow_adk/agents/

agent_x_mlflow:
	@curl -sf $(MLFLOW_TRACKING_URI)/health > /dev/null 2>&1 || \
		{ echo "MLflow server is not running. Start it first with: make mlflow"; exit 1; }
	@EXPERIMENT_ID=$$(uv run python -c "import mlflow; mlflow.set_tracking_uri('$(MLFLOW_TRACKING_URI)'); print(mlflow.set_experiment('$(MLFLOW_EXPERIMENT)').experiment_id)") && \
	OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=$(MLFLOW_TRACKING_URI)/v1/traces \
	OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id=$$EXPERIMENT_ID \
	uv run adk web src/mlflow_adk/agents/

format:
	uv run ruff format .

lint:
	uv run ruff check .
