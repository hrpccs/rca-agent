"""Benchmark / evaluation runner.

Runs the RCA agent over a set of cases and records structural + qualitative
metrics. The rca100 dataset is blind (no published ground truth yet), so we
score what can be measured objectively: whether the agent converged, confidence,
entity/evidence richness, tool usage, token cost, latency. When the benchmark's
prediction_schema / taxonomy is published, :mod:`scoring` can plug in exact
match / taxonomy scoring without changing the runner.
"""
from __future__ import annotations

import asyncio
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..agent import build_agent_for_case
from ..cases import list_cases
from ..contracts import RcaReport


def _metrics_for(report: RcaReport | None, elapsed_s: float, error: str | None) -> dict[str, Any]:
    if report is None:
        return {"status": "error", "error": error, "elapsed_s": round(elapsed_s, 1)}
    rc = report.root_cause
    tool_calls = Counter(
        s.tool_name for s in report.steps if str(s.step_kind) == "StepKind.TOOL_CALL" or getattr(s.step_kind, "value", None) == "tool_call"
    )
    n_steps = len(report.steps)
    return {
        "case_id": report.case_id,
        "task_id": report.task_id,
        "alert_title": report.alert_title,
        "status": report.status,
        "confidence": rc.confidence,
        "has_fault_type": bool(rc.fault_type),
        "fault_type": rc.fault_type,
        "n_entities": len(rc.entity_refs),
        "n_evidence": len(rc.evidence),
        "n_actions": len(rc.recommended_actions),
        "n_steps": n_steps,
        "n_tool_calls": sum(tool_calls.values()),
        "tool_calls": dict(tool_calls),
        "tokens": (report.token_usage or {}).get("total_tokens", 0),
        "elapsed_s": round(elapsed_s, 1),
    }


def _persist_report(report: RcaReport | None) -> None:
    if report is None:
        return
    try:
        from ..store.mysql_store import MysqlStore

        MysqlStore().save_report(report)
    except Exception:
        pass


async def run_eval(
    cases: list[str] | None = None,
    backend: str = "parquet",
    limit: int | None = None,
    out_dir: str = "runs",
) -> list[dict[str, Any]]:
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
            case, agent = build_agent_for_case(cid, backend=backend)
            async for ev in agent.run(case):
                if isinstance(ev, RcaReport):
                    report = ev
        except Exception as e:  # one case failing must not abort the whole eval
            error = f"{type(e).__name__}: {e}"
        elapsed = time.monotonic() - t0
        m = _metrics_for(report, elapsed, error)
        _persist_report(report)
        results.append(m)
        # also write the individual report json
        if report is not None:
            (out / f"{cid}.report.json").write_text(report.model_dump_json(indent=2))
        print(
            f"[{cid}] status={m.get('status')} conf={m.get('confidence')} "
            f"steps={m.get('n_steps','-')} tools={m.get('n_tool_calls','-')} "
            f"tok={m.get('tokens','-')} {m.get('elapsed_s','-')}s"
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
    print(f"\nsummary -> {out/'eval_summary.json'} (+ .csv)")
    return results


def _avg(xs: list[float | int | None]) -> float:
    vals = [float(x) for x in xs if x is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _pct(bs: list[bool | None]) -> float:
    vals = [b for b in bs if b is not None]
    return round(100.0 * sum(1 for b in vals if b) / len(vals), 1) if vals else 0.0


__all__ = ["run_eval"]
