"""Command-line interface for the RCA agent.

Subcommands:
  cases                 list available benchmark cases
  run <case>            run the RCA agent on a case (parquet|clickhouse backend)
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
import os
import sys
from pathlib import Path

_PROXY_VARS = (
    "all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
    "https_proxy", "HTTPS_PROXY", "socks_proxy", "SOCKS_PROXY",
)


def _clear_proxy_env() -> None:
    for v in _PROXY_VARS:
        os.environ.pop(v, None)


def _cmd_cases(args: argparse.Namespace) -> int:
    from .cases import list_cases

    ids = list_cases()
    print(f"{len(ids)} cases:")
    for cid in ids:
        print(f"  {cid}")
    return 0


def _print_step(step) -> None:
    kind = step.step_kind.value if hasattr(step.step_kind, "value") else str(step.step_kind)
    if kind == "reasoning":
        print(f"\n[thought] {step.thought}"[:600], flush=True)
    elif kind == "tool_call":
        print(f"\n→ tool_call: {step.tool_name} {json.dumps(step.tool_args, ensure_ascii=False)}", flush=True)
    elif kind == "tool_result":
        txt = (step.tool_result_text or "")[:700]
        print(f"← {step.tool_name}: {txt}", flush=True)
    elif kind == "conclude":
        print(f"\n[conclude] conf={step.confidence} :: {step.hypothesis}"[:800], flush=True)
    elif kind == "error":
        print(f"\n[error] {step.thought}", flush=True)


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
        print("\n(no report produced)", flush=True)
        return 1

    rc = report.root_cause
    print("\n" + "=" * 70, flush=True)
    print(f"STATUS: {report.status}  |  confidence: {rc.confidence}", flush=True)
    print(f"FAULT TYPE: {rc.fault_type or '(unspecified)'}", flush=True)
    print(f"ROOT CAUSE: {rc.summary}", flush=True)
    if rc.entity_refs:
        print("ENTITIES:", ", ".join(
            (e.get("entity_name") or e.get("entity_id") or "?") if isinstance(e, dict) else str(e)
            for e in rc.entity_refs
        ), flush=True)
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
    from .contracts import AlertFilter, EventFilter, LogFilter, MetricFilter, TopologyFilter, TraceFilter
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
        rows = provider.query_metrics(MetricFilter(window=w, services=[args.filter] if args.filter else None, limit=20))
        for r in rows[:20]:
            print(r.entity_name, r.metric, r.summary_stats())
        print(f"({len(rows)} series)")
    elif mod == "logs":
        rows = provider.query_logs(LogFilter(window=w, contains=args.filter, limit=10))
        for r in rows[:10]:
            print(r.pod, "::", r.content[:160])
        print(f"({len(rows)} logs)")
    elif mod == "traces":
        rows = provider.query_traces(TraceFilter(window=w, service_names=[args.filter] if args.filter else None, limit=5))
        for t in rows[:5]:
            sp = t.slowest_span()
            print(t.trace_id[:12], "spans:", len(t.spans), "slowest:", sp.name if sp else "-", sp.duration_ns if sp else "")
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
    asyncio.run(run_eval(cases=cases, backend=args.backend, limit=args.limit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rca-agent", description="LLM-core RCA agent CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cases", help="list benchmark cases").set_defaults(func=_cmd_cases)

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
    pd.add_argument("modality", choices=["alerts", "metrics", "logs", "traces", "events", "topology"])
    pd.add_argument("--filter", default=None, help="service / pod / substring depending on modality")
    pd.set_defaults(func=_cmd_data)

    pi = sub.add_parser("import-case", help="import a case into ClickHouse")
    pi.add_argument("case")
    pi.set_defaults(func=_cmd_import)

    ps = sub.add_parser("serve", help="start the FastAPI SSE server")
    ps.add_argument("--host", default=None)
    ps.add_argument("--port", type=int, default=None)
    ps.add_argument("--no-reload", action="store_true")
    ps.set_defaults(func=_cmd_serve)

    pe = sub.add_parser("eval", help="benchmark the agent over cases")
    pe.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    pe.add_argument("--backend", default="parquet")
    pe.add_argument("--limit", type=int, default=None)
    pe.set_defaults(func=_cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    _clear_proxy_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
