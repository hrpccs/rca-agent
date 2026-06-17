"""Tests for :class:`rca_agent.store.trace_store.InMemoryTraceStore`.

Pure, no DB. Covers the lifecycle (start/finish), step append+replay ordering,
``list_runs`` newest-first + case filter + ``step_count``, ``get_run`` summary
and the unknown-run ``None`` path. Also asserts the SQL store structurally
satisfies the :class:`TraceStore` Protocol (duck typing without inheritance).
"""
from __future__ import annotations

import time

from rca_agent.contracts import RcaStep, StepKind
from rca_agent.store.mysql_store import MysqlStore
from rca_agent.store.trace_store import InMemoryTraceStore, TraceStore


def _step(step_id: str, case_id: str, kind: StepKind = StepKind.REASONING) -> RcaStep:
    return RcaStep(step_id=step_id, case_id=case_id, step_kind=kind, thought=step_id)


def test_start_run_returns_run_id_and_records_model():
    ts = InMemoryTraceStore()
    run_id = ts.start_run("c1", "deepseek-reasoner")
    assert isinstance(run_id, str) and run_id
    summary = ts.get_run(run_id)
    assert summary is not None
    assert summary["case_id"] == "c1"
    assert summary["model"] == "deepseek-reasoner"
    assert summary["status"] == "running"
    assert summary["finished_at"] is None
    assert summary["step_count"] == 0


def test_append_then_list_steps_returns_them_in_seq_order():
    ts = InMemoryTraceStore()
    run_id = ts.start_run("c1", "m")
    # Append out of seq order to prove list_steps sorts by seq.
    ts.append_step(run_id, "c1", 2, _step("s2", "c1"))
    ts.append_step(run_id, "c1", 1, _step("s1", "c1"))
    ts.append_step(run_id, "c1", 3, _step("s3", "c1"))
    out = ts.list_steps(run_id)
    assert [s.step_id for s in out] == ["s1", "s2", "s3"]


def test_list_steps_respects_limit():
    ts = InMemoryTraceStore()
    run_id = ts.start_run("c1", "m")
    for i in range(5):
        ts.append_step(run_id, "c1", i, _step(f"s{i}", "c1"))
    assert len(ts.list_steps(run_id, limit=3)) == 3


def test_list_steps_unknown_run_returns_empty():
    ts = InMemoryTraceStore()
    assert ts.list_steps("nope") == []


def test_finish_run_sets_status_and_token_usage():
    ts = InMemoryTraceStore()
    run_id = ts.start_run("c1", "m")
    ts.finish_run(run_id, "completed", token_usage={"total": 42})
    summary = ts.get_run(run_id)
    assert summary["status"] == "completed"
    assert summary["finished_at"] is not None
    assert summary["token_usage"] == {"total": 42}


def test_list_runs_filters_by_case_and_is_newest_first():
    ts = InMemoryTraceStore()
    r1 = ts.start_run("cA", "m")
    # Force r1's started_at strictly earlier than r2's so newest-first is
    # observable regardless of clock resolution.
    time.sleep(0.001)
    r2 = ts.start_run("cA", "m")
    time.sleep(0.001)
    r3 = ts.start_run("cB", "m")

    ts.append_step(r1, "cA", 0, _step("a0", "cA"))
    ts.append_step(r1, "cA", 1, _step("a1", "cA"))
    ts.append_step(r2, "cA", 0, _step("b0", "cA"))

    # No filter: newest first → r3, r2, r1.
    all_runs = ts.list_runs()
    assert [r["run_id"] for r in all_runs] == [r3, r2, r1]
    # step_count reflects appended steps.
    counts = {r["run_id"]: r["step_count"] for r in all_runs}
    assert counts[r1] == 2
    assert counts[r2] == 1
    assert counts[r3] == 0

    # Filter by case cA excludes r3.
    ca_runs = ts.list_runs(case_id="cA")
    assert [r["run_id"] for r in ca_runs] == [r2, r1]


def test_get_run_returns_summary_with_step_count():
    ts = InMemoryTraceStore()
    run_id = ts.start_run("c1", "m")
    ts.append_step(run_id, "c1", 0, _step("s0", "c1"))
    summary = ts.get_run(run_id)
    assert summary is not None
    assert summary["run_id"] == run_id
    assert summary["step_count"] == 1
    # Keys the frontend/UI depends on are all present.
    for key in (
        "run_id",
        "case_id",
        "status",
        "model",
        "started_at",
        "finished_at",
        "token_usage",
        "step_count",
    ):
        assert key in summary


def test_get_run_unknown_returns_none():
    ts = InMemoryTraceStore()
    assert ts.get_run("does-not-exist") is None


def test_mysql_store_structurally_satisfies_trace_store_protocol():
    """``MysqlStore`` duck-types TraceStore without subclassing it.

    Guards the Protocol/signature contract so a future rename or removal of an
    MysqlStore method is caught here rather than only at server wiring time.
    """
    # Build via from_engine with a sentinel engine so no DB connection is made.
    class _SentinelEngine:  # minimal stand-in; MysqlStore only stores it.
        pass

    store = MysqlStore.from_engine(_SentinelEngine())  # type: ignore[arg-type]
    assert isinstance(store, TraceStore)
