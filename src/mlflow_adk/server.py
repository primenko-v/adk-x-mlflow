"""Programmatic launcher for the ADK web server with MLflow tracing."""

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

from mlflow_adk.tracing import (
    configure_tracing,
    flush_and_apply_tags,
    git_info,
    setup_otlp_export,
    trace_tags,
)

_AGENTS_DIR = str(Path(__file__).parent / "agents")
_DRAIN_INTERVAL_SECONDS = 5.0

logger = logging.getLogger(__name__)


async def _drain_loop() -> None:
    """Periodically flush spans and apply buffered tags.

    Runs forever until cancelled. Cancellation triggers a final drain in the
    shutdown hook (see ``_register_drain_lifecycle``).
    """
    while True:
        await asyncio.sleep(_DRAIN_INTERVAL_SECONDS)
        try:
            flush_and_apply_tags()
        except Exception:
            logger.exception("Drain loop iteration failed")


def _register_drain_lifecycle(app: FastAPI) -> None:
    """Wrap the app's lifespan to run the drain loop alongside it.

    ADK constructs the FastAPI app with a non-``None`` ``lifespan=`` argument
    (see ``adk_web_server.get_fast_api_app`` line ~956 — it wires its own
    ``internal_lifespan`` to manage the file-system observer and runner
    cleanup). Starlette silently ignores ``add_event_handler("startup", ...)``
    when a custom lifespan was passed at construction time, so we must wrap
    the existing lifespan instead.
    """
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def wrapped_lifespan(app_: FastAPI):
        drain_task = asyncio.create_task(_drain_loop())
        try:
            async with original_lifespan(app_) as state:
                yield state
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
            # One last drain to catch tags buffered between the last poll
            # and shutdown.
            flush_and_apply_tags()

    app.router.lifespan_context = wrapped_lifespan


def run(port: int = 8000, experiment: str | None = None) -> None:
    """Start the ADK web server.

    ``experiment`` is the MLflow experiment name. Default ``None`` disables
    MLflow capture entirely — the server runs as plain ADK with no tracing.
    Pass a name (``run(experiment="my-exp")``) to enable OTLP export of
    spans to MLflow, plus the per-trace tagging machinery.
    """
    if experiment is not None:
        setup_otlp_export(experiment)

    app = get_fast_api_app(agents_dir=_AGENTS_DIR, web=True)

    if experiment is not None:
        configure_tracing()
        # The server runs in one mode for its entire lifetime, so the tag is
        # set once. Each request task inherits this context value by
        # asyncio's copy-on-create rule; no per-request setup needed. No
        # agent_module here because ADK serves multiple agents from a single
        # web process; the agent in use varies per request.
        trace_tags.set({"source": "interactive", **git_info()})
        _register_drain_lifecycle(app)

    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment name. Omit to disable MLflow capture.",
    )
    args = parser.parse_args()
    run(port=args.port, experiment=args.experiment)
