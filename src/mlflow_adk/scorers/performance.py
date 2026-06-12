"""Agent-agnostic performance scorers."""

from mlflow.entities import Feedback, Trace
from mlflow.genai.scorers import scorer


@scorer(aggregations=["mean", "p90", "max"])
def turn_latency_ms(trace: Trace) -> Feedback:
    """Per-turn wall-clock duration in milliseconds.

    Aggregations are explicit: mean for the typical-case number, p90 for
    the tail, max so a single slow turn is visible in the Run metric panel
    rather than buried in an average.
    """
    duration = trace.info.execution_duration
    if duration is None:
        return Feedback(
            name="turn_latency_ms",
            value=None,
            error="trace.info.execution_duration is None",
        )
    return Feedback(name="turn_latency_ms", value=float(duration))


@scorer(aggregations=["mean", "p90", "max"])
def turn_tokens(trace: Trace) -> Feedback:
    """Per-turn LLM token usage (input + output).

    Pulled from MLflow's built-in ``trace.info.token_usage`` aggregate —
    populated server-side from each ``call_llm`` span's ``mlflow.chat.tokenUsage``
    attribute. ``None`` (with an error message) when no LLM span on the
    trace reported usage — typical for tool-only or pure-routing turns.
    """
    usage = trace.info.token_usage
    if not usage or usage.get("total_tokens") is None:
        return Feedback(
            name="turn_tokens",
            value=None,
            error="trace has no aggregated token usage",
        )
    return Feedback(name="turn_tokens", value=float(usage["total_tokens"]))
