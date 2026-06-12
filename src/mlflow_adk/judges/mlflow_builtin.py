"""Session-level scorers — MLflow native LLM-judge backend.

These are the default whole-conversation scorers — they ship with MLflow
in ``mlflow.genai.scorers`` and assess a full multi-turn dialogue (a list
of traces that share a ``session.id``) via an LLM judge. Feedback attaches
to the session record so the result renders inline in the Chat Sessions
UI; aggregates flow to the active MLflow Run.

The judge model is configurable via ``settings.judge_model`` (forwarded to
each scorer's ``model=`` arg). ``None`` defers to MLflow's own default —
see ``settings.py`` for the format and fallback rules.

Custom judges live in ``mlflow_custom.py`` and run side-by-side with
these under the same Run.
"""

from mlflow.genai.scorers import (
    ConversationalRoleAdherence,
    ConversationalSafety,
    ConversationalToolCallEfficiency,
    ConversationCompleteness,
)

from mlflow_adk.settings import settings


def make_conversation_scorers() -> list:
    """Instantiate the default MLflow session-level scorer set.

    Returns a fresh list per call — scorers carry no per-call state but
    handing back a shared module-level instance would couple the names and
    judge-model overrides across callers, which is more friction than the
    micro-optimisation is worth.
    """
    kw = {"model": settings.judge_model} if settings.judge_model else {}
    return [
        ConversationCompleteness(**kw),
        ConversationalSafety(**kw),
        ConversationalRoleAdherence(**kw),
        ConversationalToolCallEfficiency(**kw),
    ]
