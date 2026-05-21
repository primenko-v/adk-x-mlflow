"""Custom MLflow LLM judges — our own prompts, our own model.

Built directly on :pyclass:`mlflow.genai.judges.InstructionsJudge` so we
can opt into ``include_tool_calls_in_conversation`` — a flag the
top-level ``make_judge`` factory doesn't expose, but MLflow's own
built-in conversation scorers rely on. Each judge:

- Plugs into the standard scorer pipeline via ``mlflow.genai.evaluate``.
- Emits a Feedback per session, attached to the session record in the
  Chat Sessions UI.
- Uses the same ``settings.judge_model`` as the built-in judges, so all
  session-level scoring on a Run goes through the same model.

Registration to the MLflow Judges UI is a separate ``register()`` step
(see ``make register_judges``) — mirrors the prompt pattern: source is in
this file, registration publishes it and auto-bumps the version when
instructions change.

Tool spans are woven into the conversation the judge sees by MLflow's
auto-extractor — relies on ``tracing.py`` setting ``mlflow.spanType=TOOL``
on ADK ``execute_tool`` spans.
"""

from textwrap import dedent

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.judges import CategoricalRating
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.scorers import (
    ConversationalSafety,
    ConversationalToolCallEfficiency,
    ConversationCompleteness,
)

from mlflow_adk.settings import settings

_GROUNDEDNESS_INSTRUCTIONS = dedent(
    """\
    You are evaluating whether an AI assistant's responses are faithful to
    the evidence available in the conversation.

    Conversation: {{ conversation }}

    For every factual claim the assistant makes — numbers, supported
    items, capabilities, etc. — decide whether it is supported by EITHER:

    - A tool call earlier in this conversation, OR
    - Information the user provided.

    Multi-turn caching is acceptable: if a tool returned a value in an
    earlier turn, the assistant may reference that value in a later turn
    without re-calling the tool. Such recall counts as grounded.

    Output 1 if every factual claim by the assistant is grounded in the
    available evidence. Output 0 if the assistant invented, contradicted,
    or extrapolated beyond the evidence in any turn.
    """
)


def make_session_groundedness_judge():
    """Session-level groundedness check, one LLM call per session.

    Sees the whole conversation including tool I/O via ``{{ conversation }}``
    + ``include_tool_calls_in_conversation=True``. Catches multi-turn
    caching correctly: an answer cited from an earlier tool call still
    scores as grounded even when no fresh tool call happens this turn.
    """
    return InstructionsJudge(
        name="session_groundedness",
        instructions=_GROUNDEDNESS_INSTRUCTIONS,
        model=settings.judge_model,
        feedback_value_type=int,
        include_tool_calls_in_conversation=True,
    )


def make_custom_judges() -> list:
    """All custom session-level judges, fresh instances per call.

    Mirrors :pyfunc:`make_conversation_scorers` in
    :mod:`mlflow_adk.judges.mlflow_builtin` — the two factories are
    concatenated in ``evaluate.py`` to form the session-level scorer set.
    """
    return [make_session_groundedness_judge()]


def _categorical_to_int(fb: Feedback, name: str) -> Feedback:
    """Translate a yes/no ``Feedback`` to a 1/0 ``Feedback``.

    The Quality sub-tab of the MLflow GenAI Overview only renders the
    "Moving average over time" line for assessments whose values average
    to a number. Categorical ratings get a bar chart only. This shim
    re-emits a parallel numeric Feedback so the same judgement shows up
    as a trend line.

    ``UNKNOWN`` ratings and judge errors become ``value=None`` so the
    aggregator skips them (matches MLflow's own
    ``scorers/aggregation.py`` None-skip).
    """
    if fb.error is not None:
        return Feedback(
            name=name, error=fb.error, source=fb.source, rationale=fb.rationale
        )
    value_str = str(fb.value).lower() if fb.value is not None else ""
    if value_str == CategoricalRating.YES:
        numeric: int | None = 1
    elif value_str == CategoricalRating.NO:
        numeric = 0
    else:
        numeric = None
    return Feedback(
        name=name,
        value=numeric,
        source=fb.source,
        rationale=fb.rationale,
        metadata=fb.metadata,
    )


class _NumConversationCompleteness(ConversationCompleteness):
    name: str = "conversation_completeness_num"

    def __call__(self, *, session=None, expectations=None, **kwargs) -> Feedback:
        fb = super().__call__(session=session, expectations=expectations, **kwargs)
        return _categorical_to_int(fb, name=self.name)


class _NumConversationalSafety(ConversationalSafety):
    name: str = "conversational_safety_num"

    def __call__(self, *, session=None, expectations=None, **kwargs) -> Feedback:
        fb = super().__call__(session=session, expectations=expectations, **kwargs)
        return _categorical_to_int(fb, name=self.name)


class _NumConversationalToolCallEfficiency(ConversationalToolCallEfficiency):
    name: str = "conversational_tool_call_efficiency_num"

    def __call__(self, *, session=None, expectations=None, **kwargs) -> Feedback:
        fb = super().__call__(session=session, expectations=expectations, **kwargs)
        return _categorical_to_int(fb, name=self.name)


def make_numeric_mirrors() -> list:
    """Numeric (1/0) mirrors of the built-in yes/no conversation judges.

    Run alongside the originals — the same judgement is logged twice per
    session: once categorically (renders as a bar in the Quality tab) and
    once numerically (renders as a moving-average line). Useful to isolate
    whether the Quality tab's line-chart gate is value-type or
    feature-absent.
    """
    kw = {"model": settings.judge_model} if settings.judge_model else {}
    return [
        _NumConversationCompleteness(**kw),
        _NumConversationalSafety(**kw),
        _NumConversationalToolCallEfficiency(**kw),
    ]


def register(experiment: str) -> None:
    """Publish each custom judge to the MLflow Judges UI of ``experiment``.

    Same name re-registered with different instructions bumps the version
    automatically (MLflow scorer versioning) — that's how we track the
    judge's prompt over time. Aliases / activation for monitoring are
    managed in the UI.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    exp = mlflow.set_experiment(experiment)
    for judge in make_custom_judges():
        registered = judge.register(experiment_id=exp.experiment_id)
        print(f"Registered judge '{registered.name}' in experiment '{experiment}'.")
