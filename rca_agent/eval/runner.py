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

import asyncio
import csv
import json
import logging
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..agent import build_agent_for_case
from ..cases import list_cases
from ..contracts import Case, RcaReport, RcaStep, StepKind
from . import scoring

logger = logging.getLogger(__name__)

# Maps each builtin tool name to the modality it primarily exercises. Tools not
# listed here (or ad-hoc tools added later) fall under "other" so the modality
# breakdown never silently drops calls. This mirrors the data-query taxonomy
# exposed by the Provider contract (metrics/logs/traces/events/alerts) plus the
# topology/entity/memory investigation tools.
TOOL_MODALITY: dict[str, str] = {
    "query_metrics": "metrics",
    "query_logs": "logs",
    "query_traces": "traces",
    "query_events": "events",
    "query_alerts": "alerts",
    "get_topology": "topology",
    "inspect_entity": "entity",
    "store_observation": "memory",
}


def _modality_of(tool_name: str | None) -> str:
    """Modality bucket for a tool name; unknown names -> ``"other"``."""
    if not tool_name:
        return "other"
    return TOOL_MODALITY.get(tool_name, "other")


@dataclass
class DrainResult:
    """Result of running one agent to completion: final report + per-tool
    latency samples (monotonic seconds between each TOOL_CALL and the
    TOOL_RESULT that immediately follows it)."""

    report: RcaReport | None = None
    tool_latencies: dict[str, list[float]] = field(default_factory=dict)


class AgentFactory(Protocol):
    """Builds a ``(case, agent)`` pair for one case id.

    Matches :func:`build_agent_for_case`'s signature so the default and injected
    paths share one calling convention.
    """

    def __call__(self, case_id: str, backend: str = ...) -> tuple[Case, Any]: ...


def _metrics_for(
    report: RcaReport | None,
    elapsed_s: float,
    error: str | None,
    case_id: str | None = None,
    tool_latencies: dict[str, list[float]] | None = None,
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
    # Per-tool mean latency (monotonic seconds). Rounded for stable summaries;
    # a tool that was called but never produced a result has no latency sample
    # and is omitted (timing an unpaired call would be misleading). Samples
    # keyed under an empty tool_name (a malformed model tool_call with no
    # function name) are dropped — they would surface as a literal "" key in
    # eval_summary.json and inflate the "other" modality bucket without an
    # attributable tool.
    latencies = tool_latencies or {}
    tool_lat: dict[str, float] = {
        name: round(sum(ds) / len(ds), 4)
        for name, ds in latencies.items()
        if ds and name
    }
    # Modality breakdown: group the per-tool call counts by modality bucket so
    # bottlenecks (e.g. query_traces dominating) surface at a glance. The total
    # equals ``n_tool_calls`` only for known tools; unknown tools land in
    # "other". Counter handles the get-or-default so the bucketing expression
    # is evaluated exactly once per tool name.
    modality_counter: Counter[str] = Counter()
    for name, cnt in tool_calls.items():
        modality_counter[_modality_of(name)] += cnt
    modality_calls: dict[str, int] = dict(modality_counter)
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
        "tool_latencies": tool_lat,
        "modality_calls": modality_calls,
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


async def _drain(agent: Any, case: Case) -> DrainResult:
    """Run an agent to completion, returning the final report + per-tool
    latency samples.

    The agent yields RcaStep objects followed by one RcaReport; we keep the
    last report seen. The report carries its own ``steps`` list (populated by
    the agent), so there is no need to collect steps separately here — the
    runner reads ``len(report.steps)`` for the n_steps metric.

    Per-tool latency is derived WITHOUT touching the contracts: the agent
    emits a TOOL_CALL step immediately followed by its TOOL_RESULT step
    (see ``RcaAgent.run``), so we record ``time.monotonic()`` at the call and
    the elapsed at the matching result. A TOOL_CALL with no following result
    (e.g. agent interrupted mid-tool) simply yields no sample for that call.
    """
    report: RcaReport | None = None
    tool_latencies: dict[str, list[float]] = {}
    pending_call: tuple[str, float] | None = None
    async for ev in agent.run(case):
        if isinstance(ev, RcaReport):
            report = ev
            continue
        if not isinstance(ev, RcaStep):
            continue
        kind = ev.step_kind
        if kind == StepKind.TOOL_CALL:
            # Start timing this call. If two calls arrive back-to-back without
            # an intervening result (shouldn't happen with the current agent,
            # but is defensive), the earlier unpaired call is dropped.
            pending_call = (ev.tool_name or "", time.monotonic())
        elif kind == StepKind.TOOL_RESULT and pending_call is not None:
            name, t0 = pending_call
            pending_call = None
            tool_latencies.setdefault(name, []).append(time.monotonic() - t0)
    return DrainResult(report=report, tool_latencies=tool_latencies)


async def _run_one_case(
    cid: str, factory: AgentFactory, backend: str, out: Path
) -> dict[str, Any]:
    """Drive one case end-to-end and persist its per-case artifacts.

    Returns the per-case metrics dict. A case-level exception is captured into
    the metrics (``status="error"``) rather than propagated so a single bad
    case cannot abort the whole eval — including the concurrent path, where an
    unhandled raise inside a gathered task would otherwise cancel its siblings.
    Per-case report JSON is written here (best-effort) so concurrent runs do
    not need to coordinate on it.
    """
    t0 = time.monotonic()
    report: RcaReport | None = None
    tool_latencies: dict[str, list[float]] = {}
    error: str | None = None
    try:
        case, agent = factory(cid, backend=backend)
        drained = await _drain(agent, case)
        report = drained.report
        tool_latencies = drained.tool_latencies
    except Exception as e:  # one case failing must not abort the whole eval
        error = f"{type(e).__name__}: {e}"
    elapsed = time.monotonic() - t0
    m = _metrics_for(report, elapsed, error, case_id=cid, tool_latencies=tool_latencies)
    _persist_report(report)
    # also write the individual report json (best-effort; failure is non-fatal)
    if report is not None:
        try:
            (out / f"{cid}.report.json").write_text(report.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001 — per-case artifact I/O must not abort
            logger.warning("failed to write %s.report.json: %s", cid, exc)
    print(
        f"[{cid}] status={m.get('status')} conf={m.get('confidence')} "
        f"steps={m.get('n_steps', '-')} tools={m.get('n_tool_calls', '-')} "
        f"tok={m.get('tokens', '-')} {m.get('elapsed_s', '-')}s"
        + (f" ERR={error}" if error else ""),
        flush=True,
    )
    return m


def _build_aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the aggregate summary block from per-case metrics rows.

    Kept pure (no I/O) so the sequential and concurrent paths share one
    implementation — the aggregate is written exactly once regardless of
    concurrency, so concurrent runs never clobber ``eval_summary.json``.
    """
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
        # Per-tool mean latency averaged across the cases that called it. A tool
        # that appears in only some cases contributes only its own samples, so a
        # single slow call doesn't skew tools that were never invoked.
        "avg_tool_latency_by_tool": _aggregate_tool_latency(completed),
        # Share of total tool calls attributable to each modality bucket
        # (calls_per_modality / total_calls). Surfaces where the agent spent
        # its investigation budget (e.g. traces-heavy vs metrics-heavy).
        "modality_call_share": _modality_share(completed),
        # p90 of per-case MEAN latency per tool (one sample per completed case
        # that called the tool — not per individual call; see _tool_call_p90).
        # Empty when no latency samples were collected (e.g. all cases errored
        # pre-tool).
        "tool_call_p90": _tool_call_p90(completed),
    }
    return agg


def _collect_tool_latency_samples(
    completed: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Gather per-case ``tool_latencies`` means into per-tool sample lists.

    Shared by :func:`_aggregate_tool_latency` (mean) and :func:`_tool_call_p90`
    (p90) so the float-coercion + key handling lives in one place. Coercion
    failures (a non-numeric value sneaking into the metrics row) are skipped
    rather than crashing the whole aggregate.
    """
    sums: dict[str, list[float]] = {}
    for r in completed:
        for name, val in (r.get("tool_latencies") or {}).items():
            try:
                sums.setdefault(name, []).append(float(val))
            except (TypeError, ValueError):
                continue
    return sums


def _aggregate_tool_latency(completed: list[dict[str, Any]]) -> dict[str, float]:
    """Mean per-tool latency across completed cases (mean of per-case means)."""
    sums = _collect_tool_latency_samples(completed)
    return {name: round(sum(vs) / len(vs), 4) for name, vs in sums.items() if vs}


def _modality_share(completed: list[dict[str, Any]]) -> dict[str, float]:
    """Fraction of total tool calls per modality (0..1), rounded to 4 dp."""
    totals: Counter[str] = Counter()
    for r in completed:
        for mod, cnt in (r.get("modality_calls") or {}).items():
            try:
                totals[mod] += int(cnt)
            except (TypeError, ValueError):
                continue
    grand = sum(totals.values())
    if grand == 0:
        return {}
    return {mod: round(cnt / grand, 4) for mod, cnt in totals.items()}


def _tool_call_p90(completed: list[dict[str, Any]]) -> dict[str, float]:
    """p90 of per-case per-tool mean latency.

    Uses the per-case mean (already in ``tool_latencies``) as the sample unit
    — we don't retain raw per-call samples in the metrics row, so the per-case
    mean is the finest grain available at aggregate time. Tools with fewer
    than one sample are omitted.
    """
    sums = _collect_tool_latency_samples(completed)
    out: dict[str, float] = {}
    for name, vs in sums.items():
        if not vs:
            continue
        vs_sorted = sorted(vs)
        # Nearest-rank p90 (ceil(0.9*N)-1, clamped). Stable for small N.
        idx = max(0, math.ceil(0.9 * len(vs_sorted)) - 1)
        out[name] = round(vs_sorted[idx], 4)
    return out


def _write_summary(out: Path, results: list[dict[str, Any]], agg: dict[str, Any]) -> None:
    """Write ``eval_summary.{json,csv}`` once, after all cases finish."""
    summary = {"aggregate": agg, "results": results}
    (out / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str)
    )
    with (out / "eval_summary.csv").open("w", newline="") as f:
        cols = ["case_id", "status", "confidence", "has_fault_type", "fault_type",
                "n_entities", "n_evidence", "n_steps", "n_tool_calls", "tokens",
                "elapsed_s", "error"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)


async def run_eval(
    cases: list[str] | None = None,
    backend: str = "parquet",
    limit: int | None = None,
    out_dir: str = "runs",
    agent_factory: AgentFactory | None = None,
    concurrency: int = 1,
    sample: int | None = None,
) -> list[dict[str, Any]]:
    """Run the agent over each case and record metrics to ``out_dir``.

    ``agent_factory`` overrides :func:`build_agent_for_case` (used by tests to
    inject a fake agent with no LLM/network). When omitted the real agent is
    built per case. The default path is unchanged.

    ``concurrency`` (>1) runs cases concurrently under an
    :class:`asyncio.Semaphore` cap; the aggregate is written exactly once after
    all cases finish, so concurrent runs do not clobber ``eval_summary.json``.
    ``concurrency==1`` preserves the historical sequential behavior. A warning
    is logged for ``concurrency>3`` as a guardrail against overwhelming the
    GLM/DeepSeek gateway.

    ``sample`` (when set and ``cases`` is None) randomly picks that many case
    ids from :func:`list_cases` — handy for cheap spot-checks over the full
    rca100 set.
    """
    factory: AgentFactory = agent_factory or _default_factory
    if cases:
        ids = list(cases)
    else:
        all_ids = list_cases()
        if sample and sample > 0:
            # random.sample handles sample>len gracefully by raising; clamp so a
            # too-large --sample just returns everything.
            n = min(sample, len(all_ids))
            ids = random.sample(all_ids, n) if n else []
        else:
            ids = all_ids
    if limit:
        ids = ids[:limit]
    # Dedup while preserving order. Without this, `--cases a,a` (or any caller
    # passing duplicates) would run the same case twice; under concurrency>1
    # two tasks would then write the same {cid}.report.json path concurrently
    # and corrupt it (Path.write_text is open-truncate-write, not atomic).
    seen: set[str] = set()
    ids = [c for c in ids if not (c in seen or seen.add(c))]
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    if concurrency > 3:
        # Warn-only by design: the spec requires the Semaphore to bound at the
        # requested N (not clamp it), so operators stay in control. The message
        # deliberately does NOT claim a cap is applied — it just flags the risk.
        logger.warning(
            "run_eval: concurrency=%s exceeds the GLM/DeepSeek gateway safety "
            "threshold (3); 529s are likely. Proceeding uncapped at the "
            "requested concurrency (no automatic clamp).",
            concurrency,
        )

    if concurrency <= 1:
        # Sequential path — preserved exactly from the pre-concurrency behavior.
        results: list[dict[str, Any]] = []
        for cid in ids:
            results.append(await _run_one_case(cid, factory, backend, out))
    else:
        # Concurrent path: Semaphore bounds in-flight cases to the requested N.
        # Per-case exceptions are captured inside _run_one_case, so gather never
        # sees an exception that would cancel sibling tasks. Results come back
        # in submission order so the summary is deterministic.
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(cid: str) -> dict[str, Any]:
            async with sem:
                return await _run_one_case(cid, factory, backend, out)

        results = list(await asyncio.gather(*(_bounded(cid) for cid in ids)))

    agg = _build_aggregate(results)
    _write_summary(out, results, agg)
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


__all__ = ["run_eval", "AgentFactory", "TOOL_MODALITY", "DrainResult"]
