"""End-to-end evaluation runner.

Reads existing MLflow traces (produced by ``simulate.py`` or ``server.py``),
scores them, and writes results onto one MLflow Run.

Two scoring layers, both landing on the same Run for cross-Run comparison:

- **Turn-level** — deterministic Python scorers (``scorers/tool_calls.py``,
  ``scorers/performance.py``). Feedback attaches to each real trace;
  visible inline in the Traces view; cheap (no LLM cost).
- **Session-level** — MLflow native LLM-judge scorers
  (``ConversationCompleteness``, ``ConversationalSafety``, etc.) plus
  any custom judges from ``judges/mlflow_custom.py``. Feedback attaches
  to each session; visible inline in the Chat Sessions view. Aggregates
  flow to the Run.

Each scorer is declared in its own module:

- ``scorers/performance.py`` — deterministic per-turn scorers (latency,
  tokens). Free, no LLM.
- ``judges/mlflow_builtin.py`` — MLflow's built-in conversation judges
  (``ConversationCompleteness`` etc.).
- ``judges/mlflow_custom.py`` — our own session-level judges, e.g. the
  multi-turn groundedness check.

The orchestrator references them via ``TURN_SCORERS`` (turn-level) and
the concatenated ``make_conversation_scorers() + make_custom_judges()``
(session-level). One Run per CLI invocation; Run params capture filter +
scorer set, so the Run page is enough to reproduce or interpret the
evaluation.
"""

import argparse
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Trace
from mlflow.tracing.constant import AssessmentMetadataKey, TraceMetadataKey

from mlflow_adk.agents.simple_agent.agent import PROMPT_NAME
from mlflow_adk.judges.mlflow_builtin import make_conversation_scorers
from mlflow_adk.judges.mlflow_custom import make_custom_judges, make_numeric_mirrors
from mlflow_adk.scorers.performance import turn_latency_ms, turn_tokens
from mlflow_adk.settings import settings

logger = logging.getLogger(__name__)


TURN_SCORERS = [
    turn_latency_ms,
    turn_tokens,
]


def build_filter_string(
    source: str | None = None,
    prompt_version: str | None = None,
    git_commit: str | None = None,
    scenario: str | None = None,
    exclude_already_evaluated: bool = False,
) -> str | None:
    """Compose an MLflow search_traces filter string from tag predicates.

    Empty filter returns ``None`` — caller passes that straight to
    ``search_traces`` which then returns everything in the experiment.
    Quoting follows MLflow's filter syntax: ``tags."key" = 'value'``.

    ``exclude_already_evaluated`` adds ``tags.eval_run_id IS NULL`` so
    re-runs over the same selection don't re-score (and re-pay for)
    traces already tagged by a prior eval Run.
    """
    clauses: list[str] = []
    if source:
        clauses.append(f"tags.source = '{source}'")
    if prompt_version:
        clauses.append(f"tags.prompt_version = '{prompt_version}'")
    if git_commit:
        clauses.append(f"tags.git_commit = '{git_commit}'")
    if scenario:
        clauses.append(f"tags.scenario = '{scenario}'")
    if exclude_already_evaluated:
        clauses.append("tags.eval_run_id IS NULL")
    return " AND ".join(clauses) if clauses else None


def _unique_prompt_version(traces: Iterable) -> str | None:
    """Return the single prompt version across all traces, or None if mixed.

    Mixed or absent prompt_version → return None and skip the Run-side
    prompt linker; pinning one prompt to a Run that scored multiple
    versions would be misleading.
    """
    versions = {
        (t.info.tags or {}).get("prompt_version")
        for t in traces
        if (t.info.tags or {}).get("prompt_version")
    }
    if len(versions) == 1:
        return next(iter(versions))
    if len(versions) > 1:
        logger.warning(
            "Traces span multiple prompt versions %s; skipping prompt-link.",
            sorted(versions),
        )
    return None


def _load_trace_ids_file(path: Path) -> list[str]:
    """Read trace IDs from ``path`` — one per line, blanks ignored."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _fetch_traces_by_id(trace_ids: list[str]) -> list[Trace]:
    """Fetch traces one-by-one via MlflowClient.

    Per-id try/except so one missing or expired trace doesn't sink the
    whole eval — log and continue.
    """
    client = MlflowClient()
    traces: list[Trace] = []
    for trace_id in trace_ids:
        try:
            traces.append(client.get_trace(trace_id))
        except Exception:
            logger.exception("Could not fetch trace %s; skipping.", trace_id)
    return traces


def _group_traces_by_session(traces: list[Trace]) -> dict[str, list[Trace]]:
    """Group traces by MLflow's native ``mlflow.trace.session`` metadata.

    The server-side OTLP ingest copies any span's ``session.id`` attribute
    onto the trace as this metadata key (see
    ``_SessionIdSpanProcessor`` in tracing.py, plus MLflow's
    ``sqlalchemy_store.log_spans``). Traces without it are skipped — they
    can't belong to a multi-turn session by definition. Within each
    group, order is preserved (the caller sorts upstream by timestamp).
    """
    by_session: dict[str, list[Trace]] = defaultdict(list)
    skipped = 0
    for trace in traces:
        sid = (trace.info.trace_metadata or {}).get(TraceMetadataKey.TRACE_SESSION)
        if not sid:
            skipped += 1
            continue
        by_session[sid].append(trace)
    if skipped:
        logger.debug("Skipped %d trace(s) with no session metadata", skipped)
    for group in by_session.values():
        group.sort(key=lambda t: t.info.request_time or 0)
    return dict(by_session)


def _warn_about_judge_errors(result, session_id: str) -> int:
    """Log per-scorer failures hiding in an ``EvaluationResult``.

    MLflow's session-eval harness catches per-scorer exceptions and writes
    them as ``Feedback`` records with an ``error`` field (see
    ``mlflow.genai.evaluation.session_utils.run_scorer``). The CLI sees no
    exception, so unless we read the result back out, judge failures are
    silent in the terminal and only visible in the UI.

    Only assessments whose source-run-id matches the current Run are
    counted — prior Runs leave their own (possibly failed) feedback on the
    same trace, and re-reporting those would be noise. Returns the count
    of failed (scorer, session) pairs from this Run.
    """
    df = getattr(result, "result_df", None)
    if df is None or df.empty or "assessments" not in df.columns:
        return 0
    run_id = getattr(result, "run_id", None)
    failures = 0
    for assessments in df["assessments"]:
        for a in assessments or []:
            source_run_id = (a.get("metadata") or {}).get(
                AssessmentMetadataKey.SOURCE_RUN_ID
            )
            if run_id and source_run_id and source_run_id != run_id:
                continue
            error = (a.get("feedback") or {}).get("error")
            if not error:
                continue
            failures += 1
            logger.warning(
                "Session %s: scorer '%s' errored — %s: %s",
                session_id,
                a.get("assessment_name", "?"),
                error.get("error_code", "ERROR"),
                error.get("error_message", "(no message)"),
            )
    return failures


def _run_mlflow_conversation_scorers(sessions: dict[str, list[Trace]]) -> int:
    """Score each session with the MLflow session-level scorers.

    Combines the built-in conversation set with our custom judges into a
    single ``evaluate`` call per session — Feedback attaches to the
    session record (rendered inline in the Chat Sessions UI), aggregates
    accumulate onto the active Run.

    Returns the number of sessions actually scored.
    """
    scorers = (
        make_conversation_scorers() + make_custom_judges() + make_numeric_mirrors()
    )
    scored = 0
    total_judge_errors = 0
    for session_id, session_traces in sessions.items():
        try:
            result = mlflow.genai.evaluate(data=session_traces, scorers=scorers)
            scored += 1
            total_judge_errors += _warn_about_judge_errors(result, session_id)
        except Exception:
            logger.exception(
                "MLflow conversation scorers failed on session %s; skipping.",
                session_id,
            )
    if total_judge_errors:
        logger.warning(
            "Judge errors detected on %d scorer invocation(s); see the trace "
            "Assessments tab in the MLflow UI for stack traces.",
            total_judge_errors,
        )
    return scored


def _tag_traces_with_eval_run_id(trace_ids: list[str], run_id: str) -> None:
    """Reverse-link each scored trace back to the Run that scored it.

    Makes ``tags.eval_run_id = '<run_id>'`` queryable from the Traces
    view — complements the Run's per-trace artifact table (which goes
    the other direction).
    """
    for trace_id in trace_ids:
        try:
            mlflow.set_trace_tag(trace_id=trace_id, key="eval_run_id", value=run_id)
        except Exception:
            logger.exception("Failed to tag trace %s with eval_run_id", trace_id)


def run_evaluation(
    experiment: str,
    source: str | None = None,
    prompt_version: str | None = None,
    git_commit: str | None = None,
    scenario: str | None = None,
    max_results: int = 200,
    trace_ids_file: Path | None = None,
    rescore: bool = False,
) -> str:
    """Run the full evaluation pipeline and return the Run ID.

    Trace selection has two modes:

    - **Explicit** — ``trace_ids_file`` is a path with one trace ID per
      line. Those traces are fetched and scored verbatim; filter flags
      and dedup are ignored ("you named them, I trust you").
    - **Filter-based** — no ``trace_ids_file``. ``search_traces`` is
      composed from ``source`` / ``prompt_version`` / ``git_commit`` /
      ``scenario``. Unless ``rescore=True``, the filter also excludes
      traces already carrying an ``eval_run_id`` tag (server-side via
      ``IS NULL``) so re-runs don't re-score the same data.

    ``max_results`` is a guardrail against an accidentally-broad filter.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    exp = mlflow.set_experiment(experiment)

    if trace_ids_file is not None:
        ids = _load_trace_ids_file(trace_ids_file)
        if not ids:
            logger.warning(
                "Trace IDs file %s is empty; nothing to score.", trace_ids_file
            )
            return ""
        logger.info(
            "Eval: explicit trace-id input — %d id(s) from %s", len(ids), trace_ids_file
        )
        traces = _fetch_traces_by_id(ids)
        filter_string = f"(explicit trace-ids from {trace_ids_file})"
    else:
        filter_string = build_filter_string(
            source=source,
            prompt_version=prompt_version,
            git_commit=git_commit,
            scenario=scenario,
            exclude_already_evaluated=not rescore,
        )
        trace_df = mlflow.search_traces(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            return_type="pandas",
        )
        if trace_df.empty:
            logger.warning("No traces matched filter: %s", filter_string or "(none)")
            return ""
        traces = trace_df["trace"].tolist()

    if not traces:
        logger.warning("Eval: no traces left after selection; not opening a Run.")
        return ""

    sessions = _group_traces_by_session(traces)
    logger.info(
        "Eval: %d trace(s) → %d session(s); selection=%s",
        len(traces),
        len(sessions),
        filter_string or "(none)",
    )

    run_name = f"eval-{source or 'all'}"
    if prompt_version:
        run_name += f"-prompt-v{prompt_version}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "filter_string": filter_string or "(none)",
                "trace_count": len(traces),
                "session_count": len(sessions),
                "turn_scorers": ",".join(s.name for s in TURN_SCORERS),
                "session_scorers_mlflow": ",".join(
                    s.name
                    for s in (
                        make_conversation_scorers()
                        + make_custom_judges()
                        + make_numeric_mirrors()
                    )
                ),
            }
        )

        # 1. Turn-level: Feedback attaches to each real trace; aggregates → Run metrics.
        mlflow.genai.evaluate(
            data=pd.DataFrame({"trace": traces}),
            scorers=TURN_SCORERS,
        )

        # 2. Session-level — MLflow native conversation scorers.
        #    Loop per-session because the session-level scorer API takes one
        #    session's traces per ``evaluate`` call.
        if sessions:
            scored = _run_mlflow_conversation_scorers(sessions)
            logger.info("MLflow conversation scorers ran on %d session(s).", scored)

        # 3. Reverse-link traces → eval Run.
        _tag_traces_with_eval_run_id(
            trace_ids=[t.info.trace_id for t in traces], run_id=run.info.run_id
        )

        # 4. Run-side prompt link (when unambiguous).
        unique_version = prompt_version or _unique_prompt_version(traces)
        if unique_version is not None:
            try:
                MlflowClient().link_prompt_version_to_run(
                    run_id=run.info.run_id,
                    prompt=f"prompts:/{PROMPT_NAME}/{unique_version}",
                )
            except Exception:
                logger.exception(
                    "Failed to link prompt %s v%s to eval Run %s",
                    PROMPT_NAME,
                    unique_version,
                    run.info.run_id,
                )

        return run.info.run_id


def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiment",
        required=True,
        help="MLflow experiment name to read traces from.",
    )
    parser.add_argument(
        "--source",
        default=None,
        choices=("simulation", "interactive", None),
        help="Filter traces by their `source` tag.",
    )
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument(
        "--trace-ids",
        type=Path,
        default=None,
        help=(
            "Path to a file with one trace ID per line. When set, those "
            "exact traces are scored; filter flags and dedup are ignored."
        ),
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help=(
            "Include traces that already have an eval_run_id tag. "
            "By default these are skipped to avoid duplicate scoring."
        ),
    )
    args = parser.parse_args()

    run_id = run_evaluation(
        experiment=args.experiment,
        source=args.source,
        prompt_version=args.prompt_version,
        git_commit=args.git_commit,
        scenario=args.scenario,
        max_results=args.max_results,
        trace_ids_file=args.trace_ids,
        rescore=args.rescore,
    )
    if run_id:
        print(f"Eval Run: {run_id}")


if __name__ == "__main__":
    _cli()
