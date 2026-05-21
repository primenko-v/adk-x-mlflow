"""LLM-as-judge scorers for the eval pipeline.

These cost LLM tokens to run and assess a full session (multi-turn
conversation), unlike the deterministic per-turn scorers in
``mlflow_adk.scorers``.

- ``mlflow_builtin``  — MLflow's catalog of conversation scorers
  (``ConversationCompleteness``, ``ConversationalSafety``, …) wired to
  the judge model from settings.
- ``mlflow_custom``   — custom MLflow judges with project-owned prompts,
  registered against the active experiment.
"""
