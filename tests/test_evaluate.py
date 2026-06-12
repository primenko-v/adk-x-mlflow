"""Tests for evaluate.py — filter composition and orchestration glue."""

from types import SimpleNamespace

import pytest

import mlflow_adk.evaluate as e


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, None),
        (
            {"source": "simulation", "prompt_version": "3", "git_commit": "abc"},
            "tags.source = 'simulation' AND "
            "tags.prompt_version = '3' AND "
            "tags.git_commit = 'abc'",
        ),
        ({"prompt_version": "2"}, "tags.prompt_version = '2'"),
        (
            {"source": "simulation", "exclude_already_evaluated": True},
            "tags.source = 'simulation' AND tags.eval_run_id IS NULL",
        ),
        ({"exclude_already_evaluated": True}, "tags.eval_run_id IS NULL"),
    ],
)
def test_build_filter_string(kwargs, expected):
    assert e.build_filter_string(**kwargs) == expected


def _trace(prompt_version: str | None) -> SimpleNamespace:
    tags = {}
    if prompt_version is not None:
        tags["prompt_version"] = prompt_version
    return SimpleNamespace(info=SimpleNamespace(tags=tags))


@pytest.mark.unit
@pytest.mark.parametrize(
    "versions, expected",
    [
        (["3", "3", "3"], "3"),
        (["2", "3"], None),
        ([None, None], None),
        (["3", None, "3"], "3"),
    ],
)
def test_unique_prompt_version(versions, expected):
    traces = [_trace(v) for v in versions]
    assert e._unique_prompt_version(traces) == expected


@pytest.mark.unit
def test_load_trace_ids_file_strips_blank_lines(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("tr-abc\n\ntr-def\n   \ntr-ghi\n")
    assert e._load_trace_ids_file(path) == ["tr-abc", "tr-def", "tr-ghi"]


@pytest.mark.unit
def test_fetch_traces_by_id_skips_failures(monkeypatch):
    class _FakeClient:
        def get_trace(self, trace_id):
            if trace_id == "tr-bad":
                raise RuntimeError("not found")
            return SimpleNamespace(info=SimpleNamespace(trace_id=trace_id))

    monkeypatch.setattr(e, "MlflowClient", lambda: _FakeClient())
    out = e._fetch_traces_by_id(["tr-a", "tr-bad", "tr-c"])
    assert [t.info.trace_id for t in out] == ["tr-a", "tr-c"]


def _trace_in_session(trace_id: str, session_id: str, request_time: int = 0):
    return SimpleNamespace(
        info=SimpleNamespace(
            trace_id=trace_id,
            trace_metadata={"mlflow.trace.session": session_id},
            request_time=request_time,
            tags={},
        )
    )


@pytest.mark.unit
def test_group_traces_by_session_groups_and_sorts_by_time():
    traces = [
        _trace_in_session("tr-2", "s1", request_time=200),
        _trace_in_session("tr-1", "s1", request_time=100),
        _trace_in_session("tr-3", "s2", request_time=50),
    ]
    groups = e._group_traces_by_session(traces)

    assert set(groups) == {"s1", "s2"}
    assert [t.info.trace_id for t in groups["s1"]] == ["tr-1", "tr-2"]
    assert [t.info.trace_id for t in groups["s2"]] == ["tr-3"]


@pytest.mark.unit
def test_group_traces_by_session_skips_traces_without_session_metadata():
    no_session = SimpleNamespace(
        info=SimpleNamespace(
            trace_id="tr-orphan", trace_metadata={}, request_time=0, tags={}
        )
    )
    traces = [no_session, _trace_in_session("tr-1", "s1")]
    groups = e._group_traces_by_session(traces)
    assert set(groups) == {"s1"}


@pytest.mark.unit
def test_run_mlflow_conversation_scorers_loops_per_session(monkeypatch):
    """One ``mlflow.genai.evaluate`` call per session, with that session's traces."""
    sessions = {
        "s1": [_trace_in_session("tr-1", "s1"), _trace_in_session("tr-2", "s1")],
        "s2": [_trace_in_session("tr-3", "s2")],
    }

    batch_sizes = []
    monkeypatch.setattr(
        e.mlflow.genai,
        "evaluate",
        lambda data, scorers: batch_sizes.append(len(data)),
    )

    scored = e._run_mlflow_conversation_scorers(sessions)

    assert scored == 2
    assert batch_sizes == [2, 1]


@pytest.mark.unit
def test_run_mlflow_conversation_scorers_continues_when_one_session_fails(monkeypatch):
    """A judge-call failure on one session must not abort the rest of the batch."""
    sessions = {sid: [_trace_in_session(f"tr-{sid}", sid)] for sid in ("a", "b", "c")}

    def fake_evaluate(data, scorers):
        if data[0].info.trace_id == "tr-b":
            raise RuntimeError("judge transient failure")

    monkeypatch.setattr(e.mlflow.genai, "evaluate", fake_evaluate)

    scored = e._run_mlflow_conversation_scorers(sessions)
    assert scored == 2  # a + c succeeded, b failed


@pytest.mark.unit
def test_tag_traces_with_eval_run_id_continues_on_failure(monkeypatch):
    calls = []

    def fake_set(trace_id, key, value):
        calls.append((trace_id, key, value))
        if trace_id == "tr-2":
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(e.mlflow, "set_trace_tag", fake_set)

    e._tag_traces_with_eval_run_id(["tr-1", "tr-2", "tr-3"], "run-xyz")

    assert [c[0] for c in calls] == ["tr-1", "tr-2", "tr-3"]
    assert all(c[1] == "eval_run_id" and c[2] == "run-xyz" for c in calls)
