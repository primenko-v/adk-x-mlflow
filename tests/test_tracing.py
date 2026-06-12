"""Tests for the trace_tags ContextVar + buffer + drain plumbing."""

from types import SimpleNamespace

import git
import pytest

from mlflow_adk.tracing import (
    _SessionIdSpanProcessor,
    drain_tagged_trace_ids,
    flush_and_apply_tags,
    git_info,
    trace_tags,
)


def _fake_span(*, trace_id: int, parent_is_none: bool) -> SimpleNamespace:
    """Construct a minimal stand-in for ReadableSpan.

    Only fields the processor reads/writes: attributes (dict-like), _attributes
    (mutable backing), context.trace_id, parent.
    """
    return SimpleNamespace(
        attributes={},
        _attributes={},
        context=SimpleNamespace(trace_id=trace_id),
        parent=None if parent_is_none else object(),
    )


@pytest.fixture(autouse=True)
def _clear_buffer():
    drain_tagged_trace_ids()
    yield
    drain_tagged_trace_ids()


@pytest.mark.unit
def test_root_span_with_tags_buffers_request_id_and_tags():
    processor = _SessionIdSpanProcessor()
    trace_id = 0x0123456789ABCDEF0123456789ABCDEF
    span = _fake_span(trace_id=trace_id, parent_is_none=True)

    token = trace_tags.set({"scenario": "weather", "source": "simulation"})
    try:
        processor.on_end(span)
    finally:
        trace_tags.reset(token)

    drained = drain_tagged_trace_ids()
    assert drained == [
        (
            "tr-0123456789abcdef0123456789abcdef",
            {"scenario": "weather", "source": "simulation"},
        )
    ]


@pytest.mark.unit
def test_non_root_span_does_not_buffer_even_with_tags_set():
    processor = _SessionIdSpanProcessor()
    span = _fake_span(trace_id=42, parent_is_none=False)

    token = trace_tags.set({"source": "simulation"})
    try:
        processor.on_end(span)
    finally:
        trace_tags.reset(token)

    assert drain_tagged_trace_ids() == []


@pytest.mark.unit
def test_root_span_without_tags_set_does_not_buffer():
    processor = _SessionIdSpanProcessor()
    span = _fake_span(trace_id=42, parent_is_none=True)

    processor.on_end(span)

    assert drain_tagged_trace_ids() == []


@pytest.mark.unit
def test_root_span_with_empty_tags_does_not_buffer():
    processor = _SessionIdSpanProcessor()
    span = _fake_span(trace_id=42, parent_is_none=True)

    token = trace_tags.set({})
    try:
        processor.on_end(span)
    finally:
        trace_tags.reset(token)

    assert drain_tagged_trace_ids() == []


@pytest.mark.unit
def test_flush_and_apply_tags_calls_force_flush_and_sets_each_tag(monkeypatch):
    processor = _SessionIdSpanProcessor()
    token = trace_tags.set({"scenario": "weather", "source": "simulation"})
    try:
        processor.on_end(_fake_span(trace_id=1, parent_is_none=True))
        processor.on_end(_fake_span(trace_id=2, parent_is_none=True))
    finally:
        trace_tags.reset(token)

    flush_calls = []
    monkeypatch.setattr(
        "mlflow_adk.tracing.otel_trace.get_tracer_provider",
        lambda: SimpleNamespace(force_flush=lambda: flush_calls.append(True)),
    )
    tag_calls = []
    monkeypatch.setattr(
        "mlflow_adk.tracing.mlflow.set_trace_tag",
        lambda trace_id, key, value: tag_calls.append((trace_id, key, value)),
    )

    flush_and_apply_tags()

    assert flush_calls == [True]
    expected_rid_1 = "tr-" + "0" * 31 + "1"
    expected_rid_2 = "tr-" + "0" * 31 + "2"
    assert sorted(tag_calls) == sorted(
        [
            (expected_rid_1, "scenario", "weather"),
            (expected_rid_1, "source", "simulation"),
            (expected_rid_2, "scenario", "weather"),
            (expected_rid_2, "source", "simulation"),
        ]
    )
    # Buffer must be empty after a successful apply.
    assert drain_tagged_trace_ids() == []


@pytest.mark.unit
def test_flush_and_apply_tags_continues_on_set_trace_tag_failure(monkeypatch):
    processor = _SessionIdSpanProcessor()
    token = trace_tags.set({"a": "1", "b": "2"})
    try:
        processor.on_end(_fake_span(trace_id=7, parent_is_none=True))
    finally:
        trace_tags.reset(token)

    monkeypatch.setattr(
        "mlflow_adk.tracing.otel_trace.get_tracer_provider",
        lambda: SimpleNamespace(force_flush=lambda: None),
    )

    calls = []

    def fake_set(trace_id, key, value):
        calls.append((trace_id, key, value))
        if key == "a":
            raise RuntimeError("boom")

    monkeypatch.setattr("mlflow_adk.tracing.mlflow.set_trace_tag", fake_set)

    flush_and_apply_tags()  # must not raise

    keys_attempted = [c[1] for c in calls]
    assert "a" in keys_attempted and "b" in keys_attempted


@pytest.mark.unit
def test_flush_and_apply_tags_skips_flush_when_provider_lacks_method(monkeypatch):
    # NoOpTracerProvider (used when OTel isn't configured) has no force_flush.
    monkeypatch.setattr(
        "mlflow_adk.tracing.otel_trace.get_tracer_provider",
        lambda: object(),
    )
    monkeypatch.setattr(
        "mlflow_adk.tracing.mlflow.set_trace_tag",
        lambda **_: None,
    )

    # Should not raise even with no spans buffered and no force_flush available.
    flush_and_apply_tags()


@pytest.mark.unit
def test_git_info_returns_expected_keys_inside_repo():
    # The test process runs inside this project's git checkout, so commit and
    # dirty are always present. Branch is omitted when HEAD is detached
    # (which it usually isn't here, but CI may run detached).
    info = git_info()
    assert "git_commit" in info
    assert "git_dirty" in info
    assert len(info["git_commit"]) == 40
    assert info["git_dirty"] in {"true", "false"}
    if "git_branch" in info:
        assert isinstance(info["git_branch"], str)
        assert info["git_branch"]


@pytest.mark.unit
def test_git_info_omits_branch_on_detached_head(monkeypatch):
    """Detached HEAD is common in CI checkouts — branch is optional, not empty."""

    class _DetachedHead:
        is_detached = True

        class commit:
            hexsha = "a" * 40

    class _FakeRepo:
        head = _DetachedHead()

        def is_dirty(self, untracked_files: bool = True) -> bool:
            return False

    monkeypatch.setattr(git, "Repo", lambda *a, **kw: _FakeRepo())
    info = git_info()
    assert "git_branch" not in info
    assert info["git_commit"] == "a" * 40
    assert info["git_dirty"] == "false"


@pytest.mark.unit
def test_git_info_returns_empty_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert git_info() == {}


@pytest.mark.unit
def test_tags_snapshot_is_independent_of_later_mutation():
    """The buffered tag dict must not change if the caller's dict is mutated."""
    processor = _SessionIdSpanProcessor()
    span = _fake_span(trace_id=1, parent_is_none=True)
    tags = {"scenario": "a"}

    token = trace_tags.set(tags)
    try:
        processor.on_end(span)
    finally:
        trace_tags.reset(token)

    tags["scenario"] = "mutated"

    drained = drain_tagged_trace_ids()
    assert drained == [("tr-" + "0" * 31 + "1", {"scenario": "a"})]
