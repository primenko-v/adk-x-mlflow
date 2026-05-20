-include .env
export

MLFLOW_TRACKING_URI ?= http://localhost:5000

.PHONY: mlflow agent agent_x_mlflow simulate simulate_to_file register_prompt freeze_prompt format lint test

mlflow:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

agent:
	uv run adk web src/mlflow_adk/agents/

agent_x_mlflow:
	@curl -sf $(MLFLOW_TRACKING_URI)/health > /dev/null 2>&1 || \
		{ echo "MLflow server is not running. Start it first with: make mlflow"; exit 1; }
	uv run python -m mlflow_adk.server --experiment adk-demo

simulate:
	@curl -sf $(MLFLOW_TRACKING_URI)/health > /dev/null 2>&1 || \
		{ echo "MLflow server is not running. Start it first with: make mlflow"; exit 1; }
	uv run python -m mlflow_adk.simulate --experiment adk-sim

simulate_to_file:
	uv run python -m mlflow_adk.simulate --output-traces traces.jsonl

register_prompt:
	@curl -sf $(MLFLOW_TRACKING_URI)/health > /dev/null 2>&1 || \
		{ echo "MLflow server is not running. Start it first with: make mlflow"; exit 1; }
	uv run python -c "from mlflow_adk.agents.simple_agent import prompts; prompts.register()"

freeze_prompt:
	@curl -sf $(MLFLOW_TRACKING_URI)/health > /dev/null 2>&1 || \
		{ echo "MLflow server is not running. Start it first with: make mlflow"; exit 1; }
	uv run python -m mlflow_adk.freeze_prompt

test:
	uv run pytest tests/ -m unit -v

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix
