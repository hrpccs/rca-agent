"""Trace persistence Protocol + a dependency-free in-memory implementation.

The server streams :class:`~rca_agent.contracts.RcaStep` events over SSE and
*also* persists them incrementally so a dropped stream still leaves a durable
(partial) trace and the frontend can replay past runs. This module defines the
shape that persistence layer must satisfy (:class:`TraceStore`) and provides an
obviously-correct, DB-free implementation (:class:`InMemoryTraceStore`) for
tests and local development without MySQL.

:class:`rca_agent.store.mysql_store.MysqlStore` already implements
``start_run``/``finish_run`` and (via Unit T1) ``append_step``/``list_steps``/
``list_runs``/``get_run`` with the same signatures, so it *structurally*
satisfies :class:`TraceStore` without inheriting from it.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from rca_agent.contracts import RcaStep

__all__ = ["TraceStore", "InMemoryTraceStore"]


def _now() -> datetime:
    return datetime.now(UTC)


@runtime_checkable
class TraceStore(Protocol):
    """Structural contract for per-step trace persistence.

    ``start_run`` / ``finish_run`` bracket a run's lifecycle; ``append_step``
    durably records each emitted step; ``list_steps`` replays a run in order;
    ``list_runs`` / ``get_run`` return run summaries (with persisted step
    counts). Implementations need NOT subclass this Protocol — duck typing is
    sufficient (:class:`~rca_agent.store.mysql_store.MysqlStore` satisfies it).
    """

    def start_run(self, case_id: str, model: str) -> str:
        """Begin a run; return its ``run_id``."""
        ...

    def finish_run(
        self,
        run_id: str,
        status: str,
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        """Mark a run finished with terminal ``status`` and optional usage."""
        ...

    def append_step(self, run_id: str, case_id: str, seq: int, step: RcaStep) -> None:
        """Persist one ``step`` (``seq`` gives per-run emit order)."""
        ...

    def list_steps(self, run_id: str, limit: int = 20000) -> list[RcaStep]:
        """Return a run's steps in emit order (bounded by ``limit``)."""
        ...

    def list_runs(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Return run summaries newest-first, optionally filtered by case."""
        ...

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run summary (or ``None`` if absent)."""
        ...


class InMemoryTraceStore:
    """Plain-dict :class:`TraceStore` implementation for tests / dev.

    No database, no SQLAlchemy: runs live in ``self._runs`` and steps live in
    ``self._steps`` keyed by ``run_id``. ``step_count`` is just
    ``len(self._steps[run_id])``. Steps are stored as the exact objects passed
    in (we trust the caller does not mutate them post-append); ``list_steps``
    sorts by the caller-supplied ``seq`` so re-emission order is preserved even
    if appends arrived out of order.
    """

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._steps: dict[str, list[RcaStep]] = {}
        # Track the caller-supplied seq per (run_id, step_id) so list_steps can
        # sort by seq without mutating the stored RcaStep objects.
        self._seq: dict[str, list[int]] = {}

    def start_run(self, case_id: str, model: str) -> str:
        run_id = uuid.uuid4().hex
        self._runs[run_id] = {
            "run_id": run_id,
            "case_id": case_id,
            "status": "running",
            "model": model,
            "started_at": _now(),
            "finished_at": None,
            "token_usage": None,
        }
        self._steps[run_id] = []
        self._seq[run_id] = []
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        run = self._runs[run_id]
        run["status"] = status
        run["finished_at"] = _now()
        if token_usage is not None:
            run["token_usage"] = token_usage

    def append_step(self, run_id: str, case_id: str, seq: int, step: RcaStep) -> None:
        # case_id is accepted for Protocol parity with the SQL store (which
        # denormalizes it for case-scoped listings). The in-memory store keys
        # steps only by run_id; case_id is already on the RcaStep / run.
        self._steps[run_id].append(step)
        self._seq[run_id].append(seq)

    def list_steps(self, run_id: str, limit: int = 20000) -> list[RcaStep]:
        steps = self._steps.get(run_id, [])
        seqs = self._seq.get(run_id, [])
        # Pair each step with its seq, sort, then apply the limit — preserves
        # emit order even if appends were not monotonic in wall-clock.
        ordered = [s for _, s in sorted(zip(seqs, steps, strict=True), key=lambda p: p[0])]
        return ordered[:limit]

    def list_runs(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        runs = list(self._runs.values())
        if case_id is not None:
            runs = [r for r in runs if r["case_id"] == case_id]
        runs.sort(key=lambda r: r["started_at"], reverse=True)
        return [self._summary(r) for r in runs[:limit]]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self._runs.get(run_id)
        if run is None:
            return None
        return self._summary(run)

    def _summary(self, run: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": run["run_id"],
            "case_id": run["case_id"],
            "status": run["status"],
            "model": run["model"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "token_usage": run["token_usage"],
            "step_count": len(self._steps.get(run["run_id"], [])),
        }
