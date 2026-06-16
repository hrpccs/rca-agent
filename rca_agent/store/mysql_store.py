"""MySQL persistence for the RCA server.

Owns four tables (see :mod:`rca_agent.store.schema`):

* ``cases``      — task + topology metadata per case
* ``rca_runs``   — one row per agent invocation (start/finish lifecycle)
* ``rca_reports``— persisted :class:`~rca_agent.contracts.RcaReport` documents
* ``config``     — simple key/value application config

The store programs only against the frozen contracts (``RcaReport`` /
``RootCause`` / ``RcaStep``) and ``rca_agent.config``. It never edits them.

Importing this module never touches MySQL — the engine is created lazily and
methods surface a clear :class:`StoreError` when the database is unreachable so
the server layer can degrade gracefully.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Double,
    Index,
    MetaData,
    String,
    Table,
    Text,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from rca_agent.config import get_settings
from rca_agent.contracts import RootCause, RcaReport, RcaStep

__all__ = ["MysqlStore", "StoreError"]


class StoreError(RuntimeError):
    """Raised when MySQL is unreachable or a persistence operation fails.

    The server layer is expected to catch this and degrade (e.g. keep serving
    from memory / return a 5xx) rather than crash the process.
    """


_SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


class MysqlStore:
    """SQLAlchemy 2.0 Core-backed persistence for RCA reports/runs/cases/config.

    The engine is constructed from ``settings.mysql_url`` (overridable via the
    ``url`` ctor argument). Tables are declared with Core :class:`MetaData` so
    they map 1:1 to ``schema.sql``; ``ensure_schema()`` executes that file
    idempotently.
    """

    def __init__(self, url: str | None = None) -> None:
        self.url: str = url or get_settings().mysql_url
        # Engine creation does NOT connect; imports stay side-effect free.
        from sqlalchemy import create_engine

        self._engine: Engine = create_engine(
            self.url,
            pool_pre_ping=True,
            pool_recycle=3600,
            future=True,
        )
        self.metadata = MetaData()
        self._define_tables()

    # ------------------------------------------------------------------ #
    # Table definitions (mirror schema.sql exactly)
    # ------------------------------------------------------------------ #
    def _define_tables(self) -> None:
        self.cases = Table(
            "cases",
            self.metadata,
            Column("case_id", String(64), primary_key=True),
            Column("task_json", Text(length=2**32 - 1)),
            Column("topology_summary", Text),
            Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
        )
        self.rca_runs = Table(
            "rca_runs",
            self.metadata,
            Column("run_id", String(64), primary_key=True),
            Column("case_id", String(64), nullable=True),
            Column("status", String(32), nullable=True),
            Column("model", String(64), nullable=True),
            Column("started_at", DateTime, nullable=True),
            Column("finished_at", DateTime, nullable=True),
            Column("token_usage", Text(length=2**32 - 1), nullable=True),
            Index("ix_rca_runs_case_id", "case_id"),
            mysql_engine="InnoDB",
        )
        self.rca_reports = Table(
            "rca_reports",
            self.metadata,
            Column("report_id", String(64), primary_key=True),
            Column("run_id", String(64), nullable=True),
            Column("case_id", String(64), nullable=True),
            Column("alert_title", String(255), nullable=True),
            Column("root_cause_json", Text(length=2**32 - 1)),
            Column("steps_json", Text(length=2**32 - 1)),
            Column("confidence", Double, nullable=True),
            Column("created_at", DateTime, server_default=text("CURRENT_TIMESTAMP")),
            Index("ix_rca_reports_case_id", "case_id"),
            Index("ix_rca_reports_run_id", "run_id"),
            mysql_engine="InnoDB",
        )
        self.config = Table(
            "config",
            self.metadata,
            Column("kv_key", String(128), primary_key=True),
            Column("kv_value", Text(length=2**32 - 1)),
            Column(
                "updated_at",
                DateTime,
                server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
            ),
            mysql_engine="InnoDB",
        )

    # ------------------------------------------------------------------ #
    # Schema bootstrap
    # ------------------------------------------------------------------ #
    def ensure_schema(self) -> None:
        """Execute ``schema.sql`` (idempotent) against the configured DB."""
        sql = _SCHEMA_FILE.read_text(encoding="utf-8")
        try:
            with self._engine.begin() as conn:
                # schema.sql contains multiple statements (CREATE DATABASE,
                # USE, CREATE TABLE ...). Execute them individually because
                # SQLAlchemy's default driver (pymysql) does not accept
                # multi-statement strings via text().
                for stmt in self._split_statements(sql):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))
        except SQLAlchemyError as exc:  # pragma: no cover - infra dependent
            raise StoreError(f"ensure_schema failed: {exc}") from exc

    @staticmethod
    def _strip_line_comments(sql: str) -> str:
        """Remove ``-- ...`` line comments (to end of line).

        Done before splitting so a stray ``;`` inside a comment cannot break
        statement boundaries. Only ``--`` comments are stripped; ``#`` and
        ``/* */`` are not used in schema.sql.
        """
        out: list[str] = []
        in_single = False
        in_double = False
        for line in sql.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("--"):
                continue  # whole line is a comment
            # Remove an inline trailing ``-- comment`` outside string literals.
            result: list[str] = []
            i = 0
            while i < len(line):
                ch = line[i]
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif (
                    ch == "-"
                    and i + 1 < len(line)
                    and line[i + 1] == "-"
                    and not in_single
                    and not in_double
                ):
                    break  # rest of line is a comment
                result.append(ch)
                i += 1
            out.append("".join(result))
        return "".join(out)

    @staticmethod
    def _split_statements(sql: str) -> list[str]:
        """Split on ';' outside string literals, after stripping ``--`` comments.

        A semicolon inside a string literal or a comment would otherwise corrupt
        the split. Comments are removed first (see :meth:`_strip_line_comments`).
        """
        sql = MysqlStore._strip_line_comments(sql)
        stmts: list[str] = []
        buf: list[str] = []
        in_single = False
        in_double = False
        for ch in sql:
            buf.append(ch)
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == ";" and not in_single and not in_double:
                stmts.append("".join(buf))
                buf = []
        tail = "".join(buf).strip()
        if tail:
            stmts.append(tail)
        return stmts

    # ------------------------------------------------------------------ #
    # Reports
    # ------------------------------------------------------------------ #
    def save_report(self, report: RcaReport, run_id: str | None = None) -> str:
        """Insert a new :class:`RcaReport` row and return its ``report_id``.

        ``RcaReport`` carries no ``report_id`` field, so a fresh UUID is minted
        for each call (one row per invocation). The caller should retain the
        returned id to re-fetch the document.

        Returns the ``report_id`` of the stored row.
        """
        report_id = uuid.uuid4().hex
        root_cause_json = report.root_cause.model_dump_json()
        steps_json = json.dumps(
            [s.model_dump(mode="json") for s in report.steps], default=_json_default
        )
        confidence = (
            float(report.root_cause.confidence)
            if report.root_cause.confidence is not None
            else None
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    self.rca_reports.insert(),
                    {
                        "report_id": report_id,
                        "run_id": run_id,
                        "case_id": report.case_id,
                        "alert_title": report.alert_title,
                        "root_cause_json": root_cause_json,
                        "steps_json": steps_json,
                        "confidence": confidence,
                    },
                )
        except SQLAlchemyError as exc:
            raise StoreError(f"save_report failed: {exc}") from exc
        return report_id

    def get_report(self, report_id: str) -> RcaReport | None:
        """Rehydrate an :class:`RcaReport` from its stored JSON."""
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    self.rca_reports.select().where(
                        self.rca_reports.c.report_id == report_id
                    )
                ).mappings().first()
        except SQLAlchemyError as exc:
            raise StoreError(f"get_report failed: {exc}") from exc
        if row is None:
            return None
        return self._row_to_report(row)

    def list_reports(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[RcaReport]:
        """List reports, newest-first, optionally filtered by ``case_id``."""
        sel = self.rca_reports.select()
        if case_id is not None:
            sel = sel.where(self.rca_reports.c.case_id == case_id)
        sel = sel.order_by(self.rca_reports.c.created_at.desc()).limit(limit)
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(sel).mappings().all()
        except SQLAlchemyError as exc:
            raise StoreError(f"list_reports failed: {exc}") from exc
        return [self._row_to_report(r) for r in rows]

    @staticmethod
    def _row_to_report(row: Any) -> RcaReport:
        rc = RootCause.model_validate_json(row["root_cause_json"])
        steps = [RcaStep.model_validate(s) for s in json.loads(row["steps_json"] or "[]")]
        return RcaReport(
            case_id=row["case_id"],
            task_id=row["case_id"],  # RcaReport requires task_id; case_id is the stable key
            alert_title=row["alert_title"] or "",
            root_cause=rc,
            steps=steps,
        )

    # ------------------------------------------------------------------ #
    # Runs
    # ------------------------------------------------------------------ #
    def start_run(self, case_id: str, model: str) -> str:
        """Create a run row in ``running`` state and return its ``run_id``."""
        run_id = uuid.uuid4().hex
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    self.rca_runs.insert(),
                    {
                        "run_id": run_id,
                        "case_id": case_id,
                        "status": "running",
                        "model": model,
                        "started_at": _now(),
                    },
                )
        except SQLAlchemyError as exc:
            raise StoreError(f"start_run failed: {exc}") from exc
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        """Mark a run finished with a terminal ``status`` and optional usage."""
        values: dict[str, Any] = {
            "status": status,
            "finished_at": _now(),
        }
        if token_usage is not None:
            values["token_usage"] = json.dumps(token_usage, default=_json_default)
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    self.rca_runs.update().where(
                        self.rca_runs.c.run_id == run_id
                    ),
                    values,
                )
        except SQLAlchemyError as exc:
            raise StoreError(f"finish_run failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Cases
    # ------------------------------------------------------------------ #
    def upsert_case(
        self,
        case_id: str,
        task_json: str,
        topology_summary: str | None = None,
    ) -> None:
        """Insert or update a case row."""
        try:
            with self._engine.begin() as conn:
                # MySQL-native INSERT ... ON DUPLICATE KEY UPDATE for true upsert.
                conn.execute(
                    text(
                        "INSERT INTO cases (case_id, task_json, topology_summary) "
                        "VALUES (:case_id, :task_json, :topology_summary) "
                        "ON DUPLICATE KEY UPDATE "
                        "task_json = VALUES(task_json), "
                        "topology_summary = VALUES(topology_summary)"
                    ),
                    {
                        "case_id": case_id,
                        "task_json": task_json,
                        "topology_summary": topology_summary,
                    },
                )
        except SQLAlchemyError as exc:
            raise StoreError(f"upsert_case failed: {exc}") from exc

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """Return the raw case row (column name -> value) or ``None``."""
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    self.cases.select().where(self.cases.c.case_id == case_id)
                ).mappings().first()
        except SQLAlchemyError as exc:
            raise StoreError(f"get_case failed: {exc}") from exc
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------ #
    # Config KV
    # ------------------------------------------------------------------ #
    def get_config(self, key: str, default: Any = None) -> Any:
        """Return the stored value for ``key`` (parsed as JSON if possible)."""
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    self.config.select().where(self.config.c.kv_key == key)
                ).mappings().first()
        except SQLAlchemyError as exc:
            raise StoreError(f"get_config failed: {exc}") from exc
        if row is None:
            return default
        raw = row["kv_value"]
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set_config(self, key: str, value: Any) -> None:
        """Insert or update a config value. Stored as JSON when non-str."""
        if isinstance(value, str):
            stored: str = value
        else:
            stored = json.dumps(value, default=_json_default)
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO config (kv_key, kv_value) VALUES (:k, :v) "
                        "ON DUPLICATE KEY UPDATE kv_value = VALUES(kv_value)"
                    ),
                    {"k": key, "v": stored},
                )
        except SQLAlchemyError as exc:
            raise StoreError(f"set_config failed: {exc}") from exc
