"""Benchmark / evaluation runner.

Runs the RCA agent over a set of cases and records structural + qualitative
metrics. The rca100 dataset is blind (no published ground truth yet), so we
score what can be measured objectively: whether the agent converged, confidence,
entity/evidence richness, tool usage, token cost, latency. When the benchmark's
prediction_schema / taxonomy is published, :mod:`scoring` can plug in exact
match / taxonomy scoring without changing the runner.

The scoring/structural-metric math (entity-set P/R/F1, fault_type match,
richness counts) lives in :mod:`rca_agent.eval.scoring`; this module only wires
those pure helpers into a per-case metrics dict + the aggregate summary, and
writes ``runs/eval_summary.{json,csv}`` plus per-case ``<cid>.report.json``.

``run_eval`` accepts an optional ``agent_factory`` so tests can inject a fake
agent (no LLM/network) instead of the real :func:`build_agent_for_case`.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from ..agent import build_agent_for_case
from ..cases import list_cases
from ..contracts import Case, RcaReport
from . import scoring

logger = logging.getLogger(__name__)


class AgentFactory(Protocol):
    """Builds a ``(case, agent)`` pair for one case id.

    Matches :func:`build_agent_for_case`'s signature so the default and injected
    paths share one calling convention.
    """

    def __call__(self, case_id: str, backend: str = ...) -> tuple[Case, Any]: ...


def _metrics_for(
    report: RcaReport | None, elapsed_s: float, error: str | None, case_id: str | None = None
) -> dict[str, Any]:
    if report is None:
        # case_id is in scope at the call site even when the agent blew up before
        # producing a report; carry it so failed cases are identifiable in the
        # summary CSV/JSON rather than getting a blank case_id column.
        return {
            "case_id": case_id,
            "status": "error",
            "error": error,
            "elapsed_s": round(elapsed_s, 1),
        }
    rc = report.root_cause
    # Per-tool-name breakdown; the predicate comes from scoring so the tool-call
    # detection rule lives in one place (the total is derived from this Counter).
    tool_calls = Counter(s.tool_name for s in report.steps if scoring.is_tool_call_step(s))
    return {
        "case_id": report.case_id,
        "task_id": report.task_id,
        "alert_title": report.alert_title,
        "status": report.status,
        "confidence": rc.confidence,
        "has_fault_type": scoring.has_fault_type(rc),
        "fault_type": rc.fault_type,
        "n_entities": scoring.n_entities(rc),
        "n_evidence": scoring.n_evidence(rc),
        "n_actions": len(rc.recommended_actions),
        "n_steps": len(report.steps),
        "n_tool_calls": sum(tool_calls.values()),
        "tool_calls": dict(tool_calls),
        "tokens": (report.token_usage or {}).get("total_tokens", 0),
        "elapsed_s": round(elapsed_s, 1),
    }


def _persist_report(report: RcaReport | None) -> None:
    """Best-effort write of the report to the persistent store.

    Storage is downstream of the eval result; a store failure must not abort
    the run. It is logged (structured) instead of silently swallowed so the
    failure is observable in production.
    """
    if report is None:
        return
    try:
        from ..store.mysql_store import MysqlStore

        MysqlStore().save_report(report)
    except Exception as exc:  # observability: surface, don't crash the eval
        logger.error(
            "run_eval: failed to persist report case_id=%s — %s; continuing",
            report.case_id,
            exc,
            extra={"case_id": report.case_id, "error": f"{type(exc).__name__}: {exc}"},
        )


def _default_factory(case_id: str, backend: str = "parquet") -> tuple[Case, Any]:
    """Default agent factory — wires the real provider/LLM/memory for a case."""
    return build_agent_for_case(case_id, backend=backend)


async def _drain(agent: Any, case: Case) -> RcaReport | None:
    """Run an agent to completion, returning the final report (or None).

    The agent yields RcaStep objects followed by one RcaReport; we keep the
    last report seen. The report carries its own ``steps`` list (populated by
    the agent), so there is no need to collect steps separately here — the
    runner reads ``len(report.steps)`` for the n_steps metric.
    """
    report: RcaReport | None = None
    async for ev in agent.run(case):
        if isinstance(ev, RcaReport):
            report = ev
    return report


async def run_eval(
    cases: list[str] | None = None,
    backend: str = "parquet",
    limit: int | None = None,
    out_dir: str = "runs",
    agent_factory: AgentFactory | None = None,
) -> list[dict[str, Any]]:
    """Run the agent over each case and record metrics to ``out_dir``.

    ``agent_factory`` overrides :func:`build_agent_for_case` (used by tests to
    inject a fake agent with no LLM/network). When omitted the real agent is
    built per case. The default path is unchanged.
    """
    factory: AgentFactory = agent_factory or _default_factory
    ids = list(cases) if cases else list_cases()
    if limit:
        ids = ids[:limit]
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    results: list[dict[str, Any]] = []
    for cid in ids:
        t0 = time.monotonic()
        report: RcaReport | None = None
        error: str | None = None
        try:
            case, agent = factory(cid, backend=backend)
            report = await _drain(agent, case)
        except Exception as e:  # one case failing must not abort the whole eval
            error = f"{type(e).__name__}: {e}"
        elapsed = time.monotonic() - t0
        m = _metrics_for(report, elapsed, error, case_id=cid)
        _persist_report(report)
        results.append(m)
        # also write the individual report json
        if report is not None:
            (out / f"{cid}.report.json").write_text(report.model_dump_json(indent=2))
        print(
            f"[{cid}] status={m.get('status')} conf={m.get('confidence')} "
            f"steps={m.get('n_steps', '-')} tools={m.get('n_tool_calls', '-')} "
            f"tok={m.get('tokens', '-')} {m.get('elapsed_s', '-')}s"
            + (f" ERR={error}" if error else ""),
            flush=True,
        )

    # Aggregate + persist.
    completed = [r for r in results if r.get("status") in ("completed", "truncated")]
    agg = {
        "n_cases": len(results),
        "n_completed": sum(1 for r in completed if r.get("status") == "completed"),
        "n_errors": sum(1 for r in results if r.get("status") == "error"),
        "avg_confidence": _avg([r.get("confidence", 0) for r in completed]),
        "avg_steps": _avg([r.get("n_steps", 0) for r in completed]),
        "avg_tokens": _avg([r.get("tokens", 0) for r in completed]),
        "avg_elapsed_s": _avg([r.get("elapsed_s", 0) for r in completed]),
        "pct_has_fault_type": _pct([r.get("has_fault_type") for r in completed]),
        "avg_entities": _avg([r.get("n_entities", 0) for r in completed]),
        "avg_evidence": _avg([r.get("n_evidence", 0) for r in completed]),
    }
    summary = {"aggregate": agg, "results": results}
    (out / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    with (out / "eval_summary.csv").open("w", newline="") as f:
        cols = ["case_id", "status", "confidence", "has_fault_type", "fault_type",
                "n_entities", "n_evidence", "n_steps", "n_tool_calls", "tokens", "elapsed_s", "error"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\nsummary -> {out / 'eval_summary.json'} (+ .csv)")
    return results


def _avg(xs: list[float | int | None]) -> float:
    vals = [float(x) for x in xs if x is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _pct(bs: list[bool | None]) -> float:
    vals = [b for b in bs if b is not None]
    return round(100.0 * sum(1 for b in vals if b) / len(vals), 1) if vals else 0.0


__all__ = ["run_eval", "AgentFactory"]
