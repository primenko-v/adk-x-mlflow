"""Deterministic scorers — no LLM call, no network, free and instant.

- ``tool_calls``   — agent-specific rules (currently the simple_agent's
  temperature/cities tools): "right tool for the question", "no
  hallucinated number".
- ``performance``  — agent-agnostic per-turn signals (latency, token
  usage) pulled from the trace itself.

LLM-as-judge scorers live in ``mlflow_adk.judges``.
"""
