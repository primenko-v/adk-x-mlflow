"""Programmatic launcher for the ADK web server with MLflow tracing."""

from pathlib import Path

import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

from mlflow_adk.tracing import configure_tracing, setup_otlp_export

_AGENTS_DIR = str(Path(__file__).parent / "agents")


def run(port: int = 8000) -> None:
    setup_otlp_export()  # experiment + OTLP env vars
    app = get_fast_api_app(agents_dir=_AGENTS_DIR, web=True)  # creates provider
    configure_tracing()  # register span processor
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    run()
