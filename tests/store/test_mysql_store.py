"""Tests for :class:`rca_agent.store.mysql_store.MysqlStore`.

The suite has two layers:

* **Live-gated** tests (``store`` fixture) exercise a real MySQL instance. If
  the DB is unreachable they are skipped so CI without MySQL does not turn red.
  The store is expected to round-trip :class:`RcaReport` documents losslessly
  and to upsert cases / config.
* **Offline** tests (no ``store`` fixture, no DB) cover the
  report↔row serialization round-trip, DB-error handling via an injected fake
  engine, and ``schema.sql`` DDL sanity. These always run.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from rca_agent.contracts import RcaReport, RcaStep, RootCause, StepKind
from rca_agent.store import mysql_store as _ms
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


# =========================================================================== #
# OFFLINE TESTS (no MySQL required — always run)
#
# These exercise serialization/error-handling/DDL without any DB. They do NOT
# use the ``store`` fixture (which requires live MySQL).
# =========================================================================== #


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeEngine(Engine):  # type: ignore[misc]
    """Minimal Engine stand-in whose ``begin``/``connect`` always raise.

    Subclassing :class:`sqlalchemy.engine.Engine` lets us pass isinstance /
    typing checks without instantiating the (heavy, real) base class. The
    abstract methods are never called because ``begin``/``connect`` raise
    before any real work happens. ``url`` is set so ``Engine.__repr__`` (which
    reads ``self.url``) does not crash on traceback/logging formatting.
    """

    def __init__(self, exc: BaseException) -> None:  # noqa: D401
        # Deliberately do NOT call Engine.__init__ (it needs a dialect/factory).
        # Engine.__slots__ is empty so the instance has a __dict__ — plain
        # attribute assignment works and keeps repr()/tracebacks happy.
        self.url = ""
        self._test_exc = exc

    def begin(self, *args: Any, **kwargs: Any):  # noqa: D401, ARG002
        raise self._test_exc

    def connect(self, *args: Any, **kwargs: Any):  # noqa: D401, ARG002
        raise self._test_exc


def _op_error(msg: str = "connection refused") -> OperationalError:
    """Build an :class:`OperationalError` like a real driver would on connect."""
    return OperationalError("statement", {}, Exception(msg))


def _single_db_error_record(caplog, op: str):
    """Return the unique WARNING log record whose message mentions ``op``.

    The store logs DB errors at WARNING (the caller retains the authoritative
    ERROR). Asserts exactly one matching record exists, so a regression that
    double-logs the same op (e.g. store + caller both at the same level) is
    caught rather than silently masked by ``next()`` returning the first match.
    """
    matches = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and op in r.getMessage()
    ]
    assert len(matches) == 1, (
        f"expected exactly 1 WARNING log for {op!r}, found {len(matches)}: "
        f"{[r.getMessage() for r in matches]}"
    )
    return matches[0]


def _rich_report(case_id: str = "t-rich") -> RcaReport:
    """A report exercising every field the row serializer must preserve.

    Covers root-cause summary/fault_type/confidence, ``entity_refs`` dicts,
    ``evidence`` pointers, ``contributing_factors``/``recommended_actions``,
    multiple steps with tool payloads, and CJK alert titles.
    """
    return RcaReport(
        case_id=case_id,
        task_id=case_id,
        alert_title="checkout 错误次数告警 🚨",
        root_cause=RootCause(
            summary="checkout pod OOMKilled due to memory leak in v2.3.1",
            confidence=0.87,
            fault_type="k8s.pod_crashloop",
            entity_refs=[
                {"entity_id": "checkout-0", "entity_type": "k8s.pod"},
                {"entity_id": "checkout-deploy", "entity_type": "k8s.deployment"},
            ],
            evidence=["log: OOMKilled", "metric: memory_rss > limit"],
            contributing_factors=["no memory limit set", "leak introduced in v2.3.1"],
            recommended_actions=["rollback to v2.3.0", "set memory limit"],
        ),
        steps=[
            RcaStep(
                step_id="s1",
                case_id=case_id,
                step_kind=StepKind.TOOL_CALL,
                tool_name="query_logs",
                tool_args={"pod": "checkout-0", "tail": 100},
                tool_result_text="... OOMKilled ...",
            ),
            RcaStep(
                step_id="s2",
                case_id=case_id,
                step_kind=StepKind.CONCLUDE,
                thought="memory leak in v2.3.1",
                confidence=0.87,
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# Report ↔ row serialization round-trip (no DB)
# --------------------------------------------------------------------------- #
def test_report_to_row_to_report_roundtrip_preserves_all_fields():
    """``_report_to_row`` then ``_row_to_report`` must be lossless.

    Calls the static helpers directly — no engine, no DB. Asserts root-cause
    fields, entity_refs, evidence, confidence, alert title, and steps survive.
    """
    original = _rich_report()

    # Serialize: report -> row dict (as save_report would insert it).
    row = MysqlStore._report_to_row(original, run_id="run-abc")
    assert row["run_id"] == "run-abc"
    assert row["case_id"] == original.case_id
    assert row["alert_title"] == original.alert_title
    assert row["confidence"] == pytest.approx(0.87)
    # JSON columns are strings (they get stored as LONGTEXT).
    assert isinstance(row["root_cause_json"], str)
    assert isinstance(row["steps_json"], str)

    # Deserialize: row dict -> report (as get_report would rehydrate it).
    # _row_to_report reads a mapping; emulate a DB row by adding report_id and
    # created_at (created_at is not consumed by _row_to_report).
    db_row: dict[str, Any] = {
        "report_id": "rid-1",
        "created_at": None,
        **row,
    }
    got = MysqlStore._row_to_report(db_row)

    # Root cause fields.
    assert got.root_cause.summary == original.root_cause.summary
    assert got.root_cause.confidence == pytest.approx(original.root_cause.confidence)
    assert got.root_cause.fault_type == original.root_cause.fault_type
    assert got.root_cause.entity_refs == original.root_cause.entity_refs
    assert got.root_cause.evidence == original.root_cause.evidence
    assert got.root_cause.contributing_factors == original.root_cause.contributing_factors
    assert got.root_cause.recommended_actions == original.root_cause.recommended_actions

    # Report-level.
    assert got.case_id == original.case_id
    assert got.alert_title == original.alert_title

    # Steps (order + payloads preserved).
    assert [s.step_id for s in got.steps] == [s.step_id for s in original.steps]
    assert got.steps[0].tool_name == original.steps[0].tool_name
    assert got.steps[0].tool_args == original.steps[0].tool_args
    assert got.steps[0].tool_result_text == original.steps[0].tool_result_text
    assert got.steps[0].step_kind == original.steps[0].step_kind
    assert got.steps[1].step_kind == StepKind.CONCLUDE
    assert got.steps[1].confidence == pytest.approx(0.87)


def test_report_to_row_confidence_zero_not_coerced_to_none():
    """A ``confidence`` of ``0.0`` must round-trip as ``0.0``, not become ``None``.

    Guards the ``float(report.root_cause.confidence) if ... is not None else
    None`` branch against a truthy-but-not-None value: ``0.0`` is falsy, so a
    naive ``if confidence:`` check would wrongly store NULL. (The ``else None``
    arm is itself unreachable today because ``RootCause.confidence`` is typed
    non-Optional ``float``, but the ``is not None`` guard still protects the
    falsy ``0.0`` case.)
    """
    r = RcaReport(
        case_id="t-zero",
        task_id="t-zero",
        alert_title="a",
        root_cause=RootCause(summary="x", confidence=0.0),
    )
    row = MysqlStore._report_to_row(r)
    assert row["confidence"] == 0.0


def test_report_to_row_empty_steps_serializes_to_empty_list():
    """A report with no steps must serialize steps_json to ``[]``."""
    r = RcaReport(
        case_id="t-empty",
        task_id="t-empty",
        alert_title="a",
        root_cause=RootCause(summary="x", confidence=0.5),
        steps=[],
    )
    row = MysqlStore._report_to_row(r)
    assert json.loads(row["steps_json"]) == []


# --------------------------------------------------------------------------- #
# DB-error handling via injected fake engine (no MySQL)
# --------------------------------------------------------------------------- #
def test_save_report_db_error_raises_store_error_and_logs(caplog):
    """On OperationalError save_report raises StoreError AND emits a structured log.

    The store logs at WARNING (the caller keeps the authoritative ERROR to
    avoid double-ERROR on every DB failure). The log carries ``op=save_report``
    and the case_id in ``extra`` so the error is attributable in production
    rather than silently swallowed.
    """
    store = MysqlStore.from_engine(_FakeEngine(_op_error("save refused")))
    report = _rich_report("t-save-err")

    with (
        caplog.at_level(logging.WARNING, logger="rca_agent.store.mysql_store"),
        pytest.raises(StoreError, match="save_report failed"),
    ):
        store.save_report(report, run_id="run-x")

    rec = _single_db_error_record(caplog, "save_report")
    assert rec.op == "save_report"
    assert rec.case_id == "t-save-err"
    assert rec.run_id == "run-x"
    assert "refused" in rec.error


def test_get_report_db_error_raises_store_error_and_logs(caplog):
    """On OperationalError get_report raises StoreError AND emits a structured log.

    Current contract: DB errors raise (None is returned only for a missing
    row, not for a connection failure). Store logs at WARNING.
    """
    store = MysqlStore.from_engine(_FakeEngine(_op_error("get refused")))

    with (
        caplog.at_level(logging.WARNING, logger="rca_agent.store.mysql_store"),
        pytest.raises(StoreError, match="get_report failed"),
    ):
        store.get_report("rid-9")

    rec = _single_db_error_record(caplog, "get_report")
    assert rec.op == "get_report"
    assert rec.report_id == "rid-9"
    assert "refused" in rec.error


def test_list_reports_db_error_raises_store_error_and_logs(caplog):
    """On OperationalError list_reports raises StoreError AND logs at WARNING."""
    store = MysqlStore.from_engine(_FakeEngine(_op_error("list refused")))

    with (
        caplog.at_level(logging.WARNING, logger="rca_agent.store.mysql_store"),
        pytest.raises(StoreError, match="list_reports failed"),
    ):
        store.list_reports(case_id="t-list", limit=10)

    rec = _single_db_error_record(caplog, "list_reports")
    assert rec.op == "list_reports"
    assert rec.case_id == "t-list"
    assert rec.limit == 10
    assert "refused" in rec.error


def test_from_engine_binds_engine_without_opening_connection():
    """``from_engine`` must construct a store bound to the given engine only.

    It must NOT create a real SQLAlchemy engine (which would require a valid
    DSN) and must NOT touch the environment.
    """
    fake = _FakeEngine(_op_error())
    store = MysqlStore.from_engine(fake)
    assert store._engine is fake  # noqa: SLF001
    # No url parsed (the fake engine has no DSN).
    assert store.url == ""
    # Tables are still defined so INSERT/SELECT objects can be built offline.
    assert "rca_reports" in store.metadata.tables


# --------------------------------------------------------------------------- #
# schema.sql DDL sanity (no DB)
#
# Assertions are scoped per-table / per-column (via regex) rather than global
# substring matches, so that a type change on ONE column does not pass by
# matching an unrelated column elsewhere in the file.
# --------------------------------------------------------------------------- #


def _schema_sql() -> str:
    return _ms._SCHEMA_FILE.read_text(encoding="utf-8")


def _table_body(ddl: str, table: str) -> str:
    """Return the body of ``CREATE TABLE ... `<table>` ( ... );`` (greedy)."""
    m = re.search(
        rf"CREATE TABLE\s+IF NOT EXISTS\s+`{table}`\s*\((.*?)\)\s*ENGINE",
        ddl,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert m, f"schema.sql missing CREATE TABLE for `{table}`"
    return m.group(1)


def _column_type(ddl_body: str, col: str) -> str | None:
    """Return the declared type of ``col`` within a table body, or None."""
    m = re.search(rf"`{col}`\s+([A-Z]+(?:\(\d+\))?)", ddl_body, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def test_schema_sql_creates_all_expected_tables():
    """``schema.sql`` must declare the four tables the store programs against."""
    ddl = _schema_sql()
    for table in ("cases", "rca_runs", "rca_reports", "config"):
        assert f"CREATE TABLE IF NOT EXISTS `{table}`" in ddl, (
            f"schema.sql missing CREATE TABLE for `{table}`"
        )


def test_schema_sql_rca_reports_has_key_columns():
    """``rca_reports`` must carry the columns+types the serializer writes/reads."""
    body = _table_body(_schema_sql(), "rca_reports")
    # The serializer (_report_to_row / _row_to_report) reads/writes these.
    for col, expected_type in [
        ("report_id", "VARCHAR(64)"),
        ("case_id", "VARCHAR(64)"),
        ("run_id", "VARCHAR(64)"),
        ("alert_title", "VARCHAR(255)"),
        ("root_cause_json", "LONGTEXT"),
        ("steps_json", "LONGTEXT"),
        ("confidence", "DOUBLE"),
        ("created_at", "DATETIME"),
    ]:
        got = _column_type(body, col)
        assert got == expected_type.upper(), (
            f"schema.sql rca_reports.`{col}` expected {expected_type}, got {got}"
        )


def test_schema_sql_cases_has_key_columns():
    """``cases`` must carry case_id (PK), task_json, topology_summary, timestamps."""
    body = _table_body(_schema_sql(), "cases")
    for col, expected_type in [
        ("case_id", "VARCHAR(64)"),
        ("task_json", "LONGTEXT"),
        ("topology_summary", "TEXT"),
        ("created_at", "DATETIME"),
    ]:
        got = _column_type(body, col)
        assert got == expected_type.upper(), (
            f"schema.sql cases.`{col}` expected {expected_type}, got {got}"
        )
    assert "PRIMARY KEY (`case_id`)" in body


def test_schema_sql_rca_runs_has_status_and_indexes():
    """``rca_runs`` must carry status + case_id index (start/finish lifecycle)."""
    body = _table_body(_schema_sql(), "rca_runs")
    assert _column_type(body, "status") == "VARCHAR(32)"
    assert "INDEX `ix_rca_runs_case_id` (`case_id`)" in body


def test_schema_sql_every_create_table_is_if_not_exists():
    """EVERY CREATE TABLE must be ``IF NOT EXISTS`` (ensure_schema is idempotent).

    Scopes the check per-statement rather than a single global substring, so a
    schema where only one of four tables is idempotent would fail.
    """
    ddl = _schema_sql()
    # Match each CREATE TABLE and the token immediately following the name;
    # require ``IF NOT EXISTS`` between CREATE TABLE and the name.
    statements = re.findall(
        r"CREATE TABLE\s+(IF NOT EXISTS\s+)?`?\w+`?", ddl, flags=re.IGNORECASE
    )
    assert len(statements) >= 4, f"expected >=4 CREATE TABLE, found {len(statements)}"
    missing = [i for i, ife in enumerate(statements) if not ife]
    assert not missing, (
        f"{len(missing)} CREATE TABLE statement(s) lack IF NOT EXISTS (ensure_schema "
        "would crash on re-run)"
    )
