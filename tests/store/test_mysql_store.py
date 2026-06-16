"""Tests for :class:`rca_agent.store.mysql_store.MysqlStore`.

These exercise a live MySQL instance (per the task's running infra). If the DB
is unreachable, the suite is skipped rather than failed so CI without MySQL
does not turn red. The store is expected to round-trip :class:`RcaReport`
documents losslessly and to upsert cases / config.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from rca_agent.contracts import RcaReport, RcaStep, RootCause, StepKind
from rca_agent.store.mysql_store import MysqlStore, StoreError


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def store() -> MysqlStore:
    s = MysqlStore()
    try:
        s.ensure_schema()
    except StoreError as exc:  # MySQL not running in this environment
        pytest.skip(f"MySQL unavailable: {exc}")
    # Clean any leftover rows from prior runs for deterministic counts.
    with s._engine.begin() as conn:  # noqa: SLF001
        for t in ("rca_reports", "rca_runs", "cases", "config"):
            conn.execute(text(f"DELETE FROM `{t}`"))
    return s


def _sample_report(case_id: str = "t001") -> RcaReport:
    return RcaReport(
        case_id=case_id,
        task_id=case_id,
        alert_title="checkout 错误次数告警",
        root_cause=RootCause(
            summary="checkout pod crashloop",
            confidence=0.82,
            fault_type="k8s.pod_crashloop",
            evidence=["log line: OOMKilled"],
        ),
        steps=[
            RcaStep(
                step_id="s1",
                case_id=case_id,
                step_kind=StepKind.TOOL_CALL,
                tool_name="query_logs",
                tool_args={"pod": "checkout-0"},
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def test_save_and_get_report_roundtrip(store: MysqlStore):
    r = _sample_report()
    rid = store.save_report(r)
    assert isinstance(rid, str) and rid

    got = store.get_report(rid)
    assert got is not None
    assert got.case_id == "t001"
    assert got.alert_title == "checkout 错误次数告警"
    assert got.root_cause.summary == "checkout pod crashloop"
    assert got.root_cause.confidence == pytest.approx(0.82)
    assert got.root_cause.fault_type == "k8s.pod_crashloop"
    assert len(got.steps) == 1
    assert got.steps[0].tool_name == "query_logs"
    assert got.steps[0].step_kind == StepKind.TOOL_CALL


def test_get_report_missing_returns_none(store: MysqlStore):
    assert store.get_report("does-not-exist-" + "0" * 16) is None


def test_list_reports_filters_by_case(store: MysqlStore):
    store.save_report(_sample_report("t001"))
    store.save_report(_sample_report("t002"))
    assert len(store.list_reports("t001")) >= 1
    assert len(store.list_reports("t002")) >= 1
    # Cross-contamination guard: t001 list must not include t002 rows.
    for rep in store.list_reports("t001"):
        assert rep.case_id == "t001"


def test_list_reports_respects_limit(store: MysqlStore):
    for i in range(5):
        store.save_report(_sample_report(f"limit-{i}"))
    out = store.list_reports(limit=3)
    assert len(out) == 3


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def test_start_and_finish_run(store: MysqlStore):
    run_id = store.start_run("t001", "deepseek-reasoner")
    assert run_id
    store.finish_run(run_id, "completed", token_usage={"total": 1234})

    with store._engine.connect() as conn:  # noqa: SLF001
        row = conn.execute(
            text("SELECT status, finished_at, token_usage FROM rca_runs WHERE run_id = :r"),
            {"r": run_id},
        ).mappings().first()
    assert row is not None
    assert row["status"] == "completed"
    assert row["finished_at"] is not None
    assert json.loads(row["token_usage"])["total"] == 1234


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def test_upsert_and_get_case(store: MysqlStore):
    store.upsert_case(
        "case-xyz",
        task_json=json.dumps({"alert": "x"}),
        topology_summary="svc-a -> svc-b",
    )
    got = store.get_case("case-xyz")
    assert got is not None
    assert got["case_id"] == "case-xyz"
    assert got["topology_summary"] == "svc-a -> svc-b"
    assert json.loads(got["task_json"])["alert"] == "x"

    # Upsert (update path).
    store.upsert_case("case-xyz", task_json=json.dumps({"alert": "y"}))
    got2 = store.get_case("case-xyz")
    assert json.loads(got2["task_json"])["alert"] == "y"


def test_get_case_missing_returns_none(store: MysqlStore):
    assert store.get_case("nope") is None


# --------------------------------------------------------------------------- #
# Config KV
# --------------------------------------------------------------------------- #
def test_config_roundtrip_json(store: MysqlStore):
    store.set_config("threshold", {"error_rate": 0.1})
    assert store.get_config("threshold") == {"error_rate": 0.1}


def test_config_roundtrip_str(store: MysqlStore):
    store.set_config("note", "hello world")
    assert store.get_config("note") == "hello world"


def test_config_default(store: MysqlStore):
    assert store.get_config("absent-key", default="fallback") == "fallback"


# --------------------------------------------------------------------------- #
# Schema bootstrap
# --------------------------------------------------------------------------- #
def test_split_statements_ignores_comments_with_semicolons():
    # A ';' inside a `--` comment must NOT split the statement.
    sql = (
        "-- this comment has a ; semicolon\n"
        "CREATE TABLE IF NOT EXISTS foo (x INT);\n"
    )
    parts = [p.strip() for p in MysqlStore._split_statements(sql) if p.strip()]
    # The CREATE TABLE must survive intact as a single statement.
    assert any(p.startswith("CREATE TABLE") and p.endswith(";") for p in parts)
    # No fragment should be a bare comment remnant.
    assert not any("comment has a" in p for p in parts)


def test_ensure_schema_is_idempotent(store: MysqlStore):
    # Running twice (once in the fixture, once here) must not raise.
    store.ensure_schema()
