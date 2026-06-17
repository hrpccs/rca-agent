"""Command-line interface for the RCA agent.

Subcommands:
  cases                 list available benchmark cases
  run <case>            run the RCA agent on a case (parquet|clickhouse backend)
  runs                  list persisted RCA runs (the durable run + step trace)
  trace <run_id>        print one run's full ordered step trace
  llm ping              one real DeepSeek call (verifies thinking mode)
  data <case> <mod>     dump a sample of one data modality for a case
  import-case <case>    import a case into ClickHouse
  serve                 start the FastAPI SSE server
  eval                  run the agent over multiple cases (benchmark)

The CLI clears SOCKS/HTTP proxy env vars at startup because the dev machine's
shell profile exports a SOCKS proxy that breaks the openai/httpx client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# A stored report/run id is a 32-char hex uuid (MysqlStore mints uuid4().hex).
# Used to validate the positional id of ``trace`` before hitting the store.
_REPORT_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")

_PROXY_VARS = (
    "all_proxy",
    "ALL_PROXY",
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "socks_proxy",
    "SOCKS_PROXY",
)


def _clear_proxy_env() -> None:
    """Strip SOCKS/HTTP proxy env vars that break the openai/httpx client.

    Safe to call repeatedly; missing vars are ignored. A debug line is emitted
    so a misbehaving shell profile is easy to spot in logs.
    """
    removed = [v for v in _PROXY_VARS if os.environ.pop(v, None) is not None]
    if removed:
        logger.debug("cleared proxy env vars: %s", ", ".join(removed))


def _configure_logging() -> None:
    """Install a stderr handler on the root logger if none is configured.

    Without this the package's ``logger.debug``/``info``/``error`` calls are
    silently dropped (the root logger has no handlers by default), which would
    hide both the diagnosability lines added in this module and the
    ``exc_info=True`` traceback logged from :func:`main`'s top-level handler.

    The level is taken from ``RCA_LOG_LEVEL`` (default ``INFO``) so operators
    can opt into ``DEBUG`` without code changes. Existing handler setups
    (e.g. uvicorn configuring logging for ``serve``) are left untouched.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    level_name = os.environ.get("RCA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s"
    )


def _cmd_cases(args: argparse.Namespace) -> int:
    from .cases import list_cases

    ids = list_cases()
    print(f"{len(ids)} cases:")
    for cid in ids:
        print(f"  {cid}")
    return 0


def _print_step(step) -> None:
    """Render one agent step to stdout.

    Attribute access is defensive end-to-end: a malformed step object (missing
    or raising ``step_kind`` *or* payload attributes like ``thought`` /
    ``tool_args``) must not abort the whole run; the offending step is logged
    at WARNING and skipped instead.
    """
    try:
        raw_kind = getattr(step, "step_kind", None)
        kind = raw_kind.value if hasattr(raw_kind, "value") else str(raw_kind)
        if kind == "reasoning":
            print(f"\n[thought] {step.thought}"[:600], flush=True)
        elif kind == "tool_call":
            print(
                f"\n→ tool_call: {step.tool_name} {json.dumps(step.tool_args, ensure_ascii=False)}",
                flush=True,
            )
        elif kind == "tool_result":
            txt = (step.tool_result_text or "")[:700]
            print(f"← {step.tool_name}: {txt}", flush=True)
        elif kind == "conclude":
            print(f"\n[conclude] conf={step.confidence} :: {step.hypothesis}"[:800], flush=True)
        elif kind == "error":
            print(f"\n[error] {step.thought}", flush=True)
        else:
            logger.debug("unhandled step_kind %s; not rendered", kind)
    except Exception as exc:  # noqa: BLE001 — renderer must not crash the run
        logger.warning("failed to render step: %r (%s)", step, exc)


def _print_trace_step(idx: int, step) -> None:
    """Render one persisted step in the ``trace`` listing.

    Unlike :func:`_print_step` (which is tuned for the live ``run`` stream and
    silently skips unknown kinds), this renders *every* step_kind via this
    dedicated renderer so a stored trace never hides
    observe/hypothesize/investigate steps. Each line shows the 1-based index,
    kind, tool name (if any), a truncated thought / tool result / tool args,
    and the timestamp. Defensive: a malformed step is logged and skipped, never
    aborting the whole trace.
    """
    try:
        raw_kind = getattr(step, "step_kind", None)
        kind = raw_kind.value if hasattr(raw_kind, "value") else str(raw_kind)
        ts = getattr(step, "ts", None)
        ts_s = ts.isoformat() if hasattr(ts, "isoformat") else (ts if ts else "-")
        tool = getattr(step, "tool_name", None)
        head = f"#{idx:<3} {kind:<12}"
        if tool:
            head += f" tool={tool}"
        # Prefer the most informative text the step carries.
        parts: list[str] = []
        if kind == "conclude":
            conf = getattr(step, "confidence", None)
            hyp = getattr(step, "hypothesis", None)
            # confidence may be 0.0 (falsy!) — only drop it when truly absent.
            if conf is not None:
                parts.append(f"conf={conf}")
            if hyp:
                parts.append(str(hyp))
        else:
            txt = getattr(step, "thought", None) or getattr(step, "tool_result_text", None)
            if txt:
                parts.append(str(txt))
        # For a tool_call, surface the args too (truncated) when no result text.
        if kind == "tool_call":
            targs = getattr(step, "tool_args", None)
            if targs:
                parts.append(json.dumps(targs, ensure_ascii=False))
        detail = " ".join(parts)
        if detail:
            detail = detail.replace("\n", " ").strip()
            head += f" :: {detail[:240]}"
        print(f"{head}  [{ts_s}]", flush=True)
    except Exception as exc:  # noqa: BLE001 — renderer must not crash the trace
        logger.warning("failed to render trace step #%s: %r (%s)", idx, step, exc)


def _cmd_run(args: argparse.Namespace) -> int:
    from .agent import build_agent_for_case

    case, agent = build_agent_for_case(args.case, backend=args.backend)
    print(f"=== RCA run: {case.task.task_id} :: {case.task.alert_title} ===", flush=True)
    print(f"backend={args.backend} model={agent.model} max_steps={agent.max_steps}", flush=True)

    async def drive():
        report = None
        async for ev in agent.run(case):
            if ev.__class__.__name__ == "RcaReport":
                report = ev
            else:
                _print_step(ev)
        return report

    report = asyncio.run(drive())

    if report is None:
        logger.warning("run produced no RcaReport for case %s", args.case)
        print("\n(no report produced)", flush=True)
        return 1

    rc = report.root_cause
    print("\n" + "=" * 70, flush=True)
    print(f"STATUS: {report.status}  |  confidence: {rc.confidence}", flush=True)
    print(f"FAULT TYPE: {rc.fault_type or '(unspecified)'}", flush=True)
    print(f"ROOT CAUSE: {rc.summary}", flush=True)
    if rc.entity_refs:
        print(
            "ENTITIES:",
            ", ".join(
                (e.get("entity_name") or e.get("entity_id") or "?")
                if isinstance(e, dict)
                else str(e)
                for e in rc.entity_refs
            ),
            flush=True,
        )
    if rc.evidence:
        print("EVIDENCE:", flush=True)
        for ev in rc.evidence[:8]:
            print(f"  - {ev}", flush=True)
    if rc.recommended_actions:
        print("ACTIONS:", flush=True)
        for a in rc.recommended_actions[:6]:
            print(f"  - {a}", flush=True)
    print(f"TOKENS: {report.token_usage}", flush=True)
    print(f"STEPS: {len(report.steps)}", flush=True)

    if args.output:
        out = Path(args.output)
    else:
        Path("runs").mkdir(exist_ok=True)
        out = Path("runs") / f"{args.case}.report.json"
    out.write_text(report.model_dump_json(indent=2))
    print(f"\nreport written to {out}", flush=True)
    return 0


def _cmd_llm_ping(args: argparse.Namespace) -> int:
    from .contracts import LLMRequest
    from .llm.deepseek_client import default_client

    async def go():
        c = default_client()
        content, reasoning, _tc, usage = await c.complete(
            LLMRequest(
                messages=[{"role": "user", "content": "What is 7 * 6? Think briefly."}],
                max_tokens=2048,
            )
        )
        print("content:", content)
        print("reasoning present:", bool(reasoning))
        print("usage:", usage)

    asyncio.run(go())
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from .contracts import (
        AlertFilter,
        EventFilter,
        LogFilter,
        MetricFilter,
        TopologyFilter,
        TraceFilter,
    )
    from .providers.parquet_provider import ParquetProvider

    provider = ParquetProvider.from_case(args.case)
    w = provider.window
    mod = args.modality
    if mod == "alerts":
        rows = provider.query_alerts(AlertFilter(window=w))
        for r in rows[:10]:
            print(r.subject, r.severity, r.status)
        print(f"({len(rows)} alerts)")
    elif mod == "metrics":
        rows = provider.query_metrics(
            MetricFilter(window=w, services=[args.filter] if args.filter else None, limit=20)
        )
        for r in rows[:20]:
            print(r.entity_name, r.metric, r.summary_stats())
        print(f"({len(rows)} series)")
    elif mod == "logs":
        rows = provider.query_logs(LogFilter(window=w, contains=args.filter, limit=10))
        for r in rows[:10]:
            print(r.pod, "::", r.content[:160])
        print(f"({len(rows)} logs)")
    elif mod == "traces":
        rows = provider.query_traces(
            TraceFilter(window=w, service_names=[args.filter] if args.filter else None, limit=5)
        )
        for t in rows[:5]:
            try:
                sp = t.slowest_span()
            except Exception as exc:  # noqa: BLE001 — one bad trace must not abort the dump
                logger.warning(
                    "trace %s: slowest_span() failed: %s", getattr(t, "trace_id", "?"), exc
                )
                sp = None
            print(
                t.trace_id[:12],
                "spans:",
                len(t.spans),
                "slowest:",
                sp.name if sp else "-",
                sp.duration_ns if sp else "",
            )
        print(f"({len(rows)} traces)")
    elif mod == "events":
        rows = provider.query_events(EventFilter(window=w, limit=20))
        for r in rows[:20]:
            print(r.level, r.pod, "::", (r.reason or ""), (r.message or "")[:120])
        print(f"({len(rows)} events)")
    elif mod == "topology":
        sub = provider.query_topology(TopologyFilter())
        print(f"{len(sub.entities)} entities, {len(sub.edges)} edges")
        for e in sub.entities[:15]:
            print(" ", e.get("type"), e.get("name"))
    else:
        print(f"unknown modality: {mod}")
        return 2
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from .providers.loader import import_case

    r = import_case(args.case)
    logger.info("imported case %s: %s", args.case, r)
    print("imported:", r)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .config import get_settings

    s = get_settings()
    uvicorn.run(
        "rca_agent.server.app:app",
        host=args.host or s.server_host,
        port=args.port or s.server_port,
        reload=not args.no_reload,
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval.runner import run_eval

    cases = args.cases.split(",") if args.cases else None
    asyncio.run(
        run_eval(
            cases=cases,
            backend=args.backend,
            limit=args.limit,
            out_dir=args.out_dir,
            concurrency=args.concurrency,
            sample=args.sample,
        )
    )
    return 0


def _new_store():
    """Construct the production MySQL store.

    Imported lazily so importing ``rca_agent.cli`` (e.g. by the server) does not
    build a SQLAlchemy engine at module load. Mirrors the lazy construction used
    elsewhere in this module (``build_agent_for_case``, ``default_client``).
    """
    from .store.mysql_store import MysqlStore

    return MysqlStore()


def _fmt_dt(v: object) -> str:
    """Render a datetime/str/None timestamp compactly for CLI output."""
    if v is None:
        return "-"
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _cmd_runs(args: argparse.Namespace) -> int:
    """List persisted RCA runs (the durable per-case run + step trace).

    Prefers the Wave-4 trace store (``rca_runs`` + per-run ``step_count``),
    which lists EVERY run — including ones that errored or were truncated, not
    just successful ones — and exposes the ``run_id`` that ``trace`` consumes.
    When the trace API is unavailable (e.g. an older store), falls back to
    ``list_reports`` so the CLI always works. ``--case`` filters, ``--limit``
    caps the page (default 50). Errors surface as a clear stderr message +
    non-zero exit, never a traceback.
    """
    from .store.mysql_store import StoreError

    try:
        store = _new_store()
        if hasattr(store, "list_runs"):
            runs = store.list_runs(case_id=args.case, limit=args.limit)
            return _render_runs_summaries(runs, args)
        reports = store.list_reports(case_id=args.case, limit=args.limit)
    except StoreError as exc:
        print(f"error: could not list runs: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI surface; never crash with a traceback
        logger.error("runs: store list failed: %s", exc, exc_info=True)
        print(f"error: could not list runs: {exc}", file=sys.stderr)
        return 1

    # Report-fallback path (no trace store): a persisted run here is one
    # rca_reports row. RcaReport carries no report_id, so we can't offer a
    # traceable id in this view — use the server / the /runs API for that.
    if not reports:
        where = f" for case {args.case}" if args.case else ""
        print(f"no runs found{where}.", flush=True)
        return 0

    print(f"{len(reports)} run(s):", flush=True)
    for r in reports:
        cid = getattr(r, "case_id", "?")
        status = getattr(r, "status", "?")
        model = getattr(r, "model", None) or "-"
        steps = len(getattr(r, "steps", []) or [])
        print(f"  case={cid}  status={status}  model={model}  steps={steps}", flush=True)
    print(
        "\n(tip: report-backed runs have no run_id here; use the server "
        "(`POST /rca`) or the `/runs` API for run_id-based traces.)",
        flush=True,
    )
    return 0


def _render_runs_summaries(runs: list, args: argparse.Namespace) -> int:
    """Render ``list_runs`` summary dicts (the Wave-4 trace-store path)."""
    if not runs:
        where = f" for case {args.case}" if args.case else ""
        print(f"no runs found{where}.", flush=True)
        return 0
    print(f"{len(runs)} run(s):", flush=True)
    for d in runs:
        rid = str(d.get("run_id") or "?")
        cid = str(d.get("case_id") or "?")
        status = str(d.get("status") or "?")
        model = str(d.get("model") or "-")
        steps = int(d.get("step_count") or 0)
        line = (
            f"  run={rid[:12]}  case={cid}  status={status}  model={model}  steps={steps}"
        )
        if d.get("started_at"):
            line += f"  started={_fmt_dt(d['started_at'])}"
        print(line, flush=True)
    print(
        "\n(tip: `rca-agent trace <run_id>` prints a run's full step trace.)",
        flush=True,
    )
    return 0


def _kind_of(step: object) -> str:
    """Best-effort step_kind string for a persisted/live step."""
    k = getattr(step, "step_kind", None)
    return k.value if hasattr(k, "value") else str(k)


def _cmd_trace(args: argparse.Namespace) -> int:
    """Print one run's full ordered step trace.

    The positional id may be a ``run_id`` (Wave-4 trace store: ``rca_steps``)
    OR a ``report_id`` (``rca_reports``, e.g. produced by ``run -o`` / eval) —
    both are 32-char hex uuids. The trace store is tried first (it covers
    server runs, including ones that errored or were truncated and never
    produced a report); if the id isn't a known run, we fall back to the report
    lookup. Unknown id / store error / bad id all surface as a clear stderr
    message + non-zero exit, never a traceback.
    """
    from .store.mysql_store import StoreError

    rid = args.report_id
    if not _REPORT_ID_RE.match(rid):
        print(
            f"error: '{rid}' is not a valid run/report id (expected 32-char hex uuid)",
            file=sys.stderr,
        )
        return 2

    try:
        store = _new_store()
        # Trace-store path: run_id -> rca_steps.
        if hasattr(store, "get_run"):
            summary = store.get_run(rid)
            if summary is not None:
                steps = store.list_steps(rid) if hasattr(store, "list_steps") else []
                return _render_trace_from_summary(rid, summary, steps)
        # Report path: report_id -> rca_reports (covers `run -o` / eval runs).
        report = store.get_report(rid)
    except StoreError as exc:
        print(f"error: could not read run {rid}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI surface; never crash with a traceback
        logger.error("trace: store get failed for %s: %s", rid, exc, exc_info=True)
        print(f"error: could not read run {rid}: {exc}", file=sys.stderr)
        return 1

    if report is None:
        print(f"error: no such run: {rid}", file=sys.stderr)
        return 1

    rc = report.root_cause
    print(f"=== trace {rid} :: case={report.case_id} ===", flush=True)
    print(
        f"status={report.status}  model={report.model or '-'}  "
        f"steps={len(report.steps)}  confidence={rc.confidence}",
        flush=True,
    )
    print(f"alert: {report.alert_title}", flush=True)
    for i, step in enumerate(report.steps):
        _print_trace_step(i + 1, step)
    print("\n" + "=" * 70, flush=True)
    print(f"ROOT CAUSE: {rc.summary}", flush=True)
    if rc.fault_type:
        print(f"FAULT TYPE: {rc.fault_type}", flush=True)
    if report.token_usage:
        print(f"TOKENS: {report.token_usage}", flush=True)
    return 0


def _render_trace_from_summary(rid: str, summary: dict, steps: list) -> int:
    """Render a run from its trace-store summary + persisted ``RcaStep`` rows.

    The terminal root cause lives in the final ``conclude`` step of the trace
    (it carries ``hypothesis`` + ``confidence``), so no report lookup is needed.
    """
    cid = str(summary.get("case_id") or "?")
    status = str(summary.get("status") or "?")
    model = str(summary.get("model") or "-")
    print(f"=== trace {rid} :: case={cid} ===", flush=True)
    head = f"status={status}  model={model}  steps={len(steps)}"
    if summary.get("started_at"):
        head += f"  started={_fmt_dt(summary['started_at'])}"
    if summary.get("finished_at"):
        head += f"  finished={_fmt_dt(summary['finished_at'])}"
    print(head, flush=True)
    if summary.get("token_usage"):
        print(f"TOKENS: {summary['token_usage']}", flush=True)
    for i, step in enumerate(steps):
        _print_trace_step(i + 1, step)
    conclude = next((s for s in reversed(steps) if _kind_of(s) == "conclude"), None)
    if conclude is not None:
        print("\n" + "=" * 70, flush=True)
        hyp = getattr(conclude, "hypothesis", None) or "(no hypothesis)"
        print(f"ROOT CAUSE: {hyp}", flush=True)
        conf = getattr(conclude, "confidence", None)
        if conf is not None:
            print(f"CONFIDENCE: {conf}", flush=True)
    return 0


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on missing/bad values.

    A non-numeric value (e.g. ``RCA_RUNS_LIMIT=unlimited``) logs a warning and
    uses the default rather than crashing ``build_parser()`` — which runs before
    ``main()``'s try/except, so an uncaught ``ValueError`` would surface as a raw
    traceback.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("env %s=%r is not an int; using default %s", name, raw, default)
        return default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rca-agent", description="LLM-core RCA agent CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cases", help="list benchmark cases").set_defaults(func=_cmd_cases)

    pruns = sub.add_parser("runs", help="list persisted RCA runs")
    pruns.add_argument("--case", default=None, help="filter by case_id")
    pruns.add_argument(
        "--limit",
        type=int,
        default=_env_int("RCA_RUNS_LIMIT", 50),
        help="max runs to list (env RCA_RUNS_LIMIT, default 50)",
    )
    pruns.set_defaults(func=_cmd_runs)

    ptrace = sub.add_parser("trace", help="print a run's full ordered step trace")
    ptrace.add_argument(
        "report_id", metavar="run_id", help="32-char hex run_id (or report_id); see `runs`"
    )
    ptrace.set_defaults(func=_cmd_trace)

    pr = sub.add_parser("run", help="run the RCA agent on a case")
    pr.add_argument("case")
    pr.add_argument("--backend", default=os.getenv("RCA_DATA_BACKEND", "parquet"))
    pr.add_argument("--output", "-o", default=None)
    pr.set_defaults(func=_cmd_run)

    pp = sub.add_parser("llm", help="LLM utilities")
    pps = pp.add_subparsers(dest="llm_cmd", required=True)
    pps.add_parser("ping", help="one real DeepSeek call").set_defaults(func=_cmd_llm_ping)

    pd = sub.add_parser("data", help="dump a sample of a data modality")
    pd.add_argument("case")
    pd.add_argument(
        "modality", choices=["alerts", "metrics", "logs", "traces", "events", "topology"]
    )
    pd.add_argument(
        "--filter", default=None, help="service / pod / substring depending on modality"
    )
    pd.set_defaults(func=_cmd_data)

    pi = sub.add_parser("import-case", help="import a case into ClickHouse")
    pi.add_argument("case")
    pi.set_defaults(func=_cmd_import)

    ps = sub.add_parser("serve", help="start the FastAPI SSE server")
    ps.add_argument("--host", default=None)
    ps.add_argument("--port", type=int, default=None)
    ps.add_argument("--no-reload", action="store_true")
    ps.set_defaults(func=_cmd_serve)

    pe = sub.add_parser(
        "eval",
        help="benchmark the agent over cases",
        description=(
            "Run the RCA agent over a set of benchmark cases and record "
            "structural + qualitative metrics (per-case + aggregate) to "
            "<out-dir>/eval_summary.{json,csv}. Use --cases/--sample to scope "
            "the run, --concurrency to run cases in parallel (capped)."
        ),
    )
    pe.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    pe.add_argument("--backend", default="parquet")
    pe.add_argument("--limit", type=int, default=None)
    pe.add_argument(
        "--out-dir",
        default="runs",
        help="directory for eval_summary.{json,csv} + per-case reports (default: runs)",
    )
    pe.add_argument(
        "--sample",
        type=int,
        default=None,
        help="randomly pick N cases from the full set (ignored when --cases is given)",
    )
    pe.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="run N cases concurrently (default 1 = sequential). A warning is "
        "logged for N>3 (GLM/DeepSeek gateway safety).",
    )
    pe.set_defaults(func=_cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    _clear_proxy_env()
    _configure_logging()
    args = build_parser().parse_args(argv)
    # build_parser uses subparsers(required=True), so args.func is always set
    # for any argv that survives parse_args (missing subcommand -> SystemExit 2).
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("interrupted by user")
        raise
    except Exception as exc:  # noqa: BLE001 — top-level surface; preserves exit-1 behavior
        # SystemExit (argparse / uvicorn) is a BaseException subclass, so it is
        # NOT caught here and propagates with its original code unchanged.
        logger.error("command %r failed: %s", getattr(args, "cmd", "?"), exc, exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
