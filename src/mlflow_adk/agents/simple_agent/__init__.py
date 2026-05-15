from mlflow_adk.tracing import configure_tracing

from .agent import root_agent

__all__ = ["root_agent"]

# Hook for `adk web`: the agent module is imported lazily after ADK has already
# set the global SDKTracerProvider, so configure_tracing() finds it here.
# When running without OTel export (make agent) this is a no-op.
configure_tracing()
