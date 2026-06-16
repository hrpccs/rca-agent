"""Builtin SRE investigation tools (args models + handlers).

Each tool is a thin wrapper over :class:`DataProvider`: the LLM passes simple
filter arguments (strings/ints), the handler derives the full
:class:`TimeWindow` from ``provider.window`` (the case alert window), calls the
matching provider query, and returns ``{"tool", "count", "text", "raw"}``.

The ``text`` field is structured-text evidence written to be information-dense
yet concise — it is what the LLM actually reads. ``raw`` carries a truncated
slice of the result models for tracing/debugging.

Handlers are pure functions ``(args, provider, memory) -> dict`` so they can be
unit-tested against a fake provider with no I/O.
"""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from ..contracts import (
    AlertFilter,
    CloudEvent,
    EventFilter,
    K8sEvent,
    LogFilter,
    LogLine,
    MemoryItem,
    MetricFilter,
    MetricSeries,
    Span,
    TopologyFilter,
    TopologySubgraph,
    Trace,
    TraceFilter,
)

# Cap on the size of the ``raw`` payload echoed back to the LLM, to keep tool
# messages cheap. Full data stays in the provider.
_RAW_CAP = 20


def _cap(items: list[Any]) -> list[Any]:
    return items[:_RAW_CAP]


def _ts(dt: Any) -> str:
    """Compact timestamp for rendering (handles None)."""
    if dt is None:
        return "-"
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        return str(dt)


# --------------------------------------------------------------------------- #
# Args models (one per tool). These double as the OpenAI JSON-schema source.
# --------------------------------------------------------------------------- #
class QueryAlertsArgs(BaseModel):
    """Fetch alert (CloudEvent) rows in the case window."""

    limit: int = Field(default=20, ge=1, le=200, description="Max alerts to return.")


class QueryEventsArgs(BaseModel):
    """Fetch Kubernetes events in the case window."""

    pod: str | None = Field(default=None, description="Filter by pod name (substring/exact by backend).")
    level: str | None = Field(default=None, description="Event level, e.g. 'Warning' or 'Normal'.")
    limit: int = Field(default=100, ge=1, le=1000, description="Max events to return.")


class QueryMetricsArgs(BaseModel):
    """Fetch metric series in the case window."""

    service: str | None = Field(default=None, description="Service name filter.")
    metric: str | None = Field(default=None, description="Metric name filter (e.g. 'cpu_usage').")
    entity_name: str | None = Field(default=None, description="Entity name filter.")
    domain: str | None = Field(default=None, description="Metric domain, e.g. 'k8s' or 'apm'.")
    limit: int = Field(default=500, ge=1, le=5000, description="Max series to return.")


class QueryLogsArgs(BaseModel):
    """Fetch log lines in the case window."""

    pod: str | None = Field(default=None, description="Pod name filter.")
    namespace: str | None = Field(default=None, description="Kubernetes namespace.")
    container: str | None = Field(default=None, description="Container name.")
    host: str | None = Field(default=None, description="Node/host name.")
    contains: str | None = Field(default=None, description="Substring to match in log content.")
    level_hint: str | None = Field(default=None, description="Level heuristic, e.g. 'ERROR' or 'WARN'.")
    limit: int = Field(default=50, ge=1, le=1000, description="Max log lines to return.")


class QueryTracesArgs(BaseModel):
    """Fetch traces in the case window."""

    service: str | None = Field(default=None, description="Service name filter.")
    span: str | None = Field(default=None, description="Span name filter.")
    status: str | None = Field(default=None, description="Span status code, e.g. 'ERROR'.")
    min_duration_ms: int | None = Field(default=None, ge=0, description="Minimum span duration in milliseconds.")
    limit: int = Field(default=10, ge=1, le=200, description="Max traces to return.")


class GetTopologyArgs(BaseModel):
    """Fetch a topology subgraph around an entity/neighborhood."""

    entity_name: str | None = Field(default=None, description="Center entity name (optional).")
    entity_type: str | None = Field(default=None, description="Entity type filter, e.g. 'apm.service'.")
    hops: int = Field(default=1, ge=0, le=5, description="Neighborhood radius (0 = exact entities).")
    limit: int = Field(default=200, ge=1, le=2000, description="Max entities to return.")


class InspectEntityArgs(BaseModel):
    """Look up a single topology entity and its neighbors."""

    entity_id: str | None = Field(default=None, description="Entity id to inspect.")
    entity_name: str | None = Field(default=None, description="Entity name to inspect (used if id omitted).")
    hops: int = Field(default=1, ge=0, le=3, description="Neighborhood radius for neighbors.")


class StoreObservationArgs(BaseModel):
    """Persist a note/observation to agent memory for later retrieval."""

    content: str = Field(description="The observation text to store.")
    kind: str = Field(default="evidence", description="Memory kind: evidence|hypothesis|note|metric_obs|log_obs.")
    entities: list[str] = Field(default_factory=list, description="Entity names/ids this observation relates to.")


# --------------------------------------------------------------------------- #
# Renderers — concise structured text for the LLM.
# --------------------------------------------------------------------------- #
def _render_alert(ev: CloudEvent) -> str:
    parts = [
        f"[{_ts(ev.ts)}] {ev.type or 'ALERT'}",
        f"sev={ev.severity or '-'}",
        f"subtype={ev.subtype or '-'}",
    ]
    if ev.subject:
        parts.append(f"subject={ev.subject}")
    if ev.status:
        parts.append(f"status={ev.status}")
    head = " ".join(parts)
    data = ev.data or {}
    if data:
        # Keep the data hint short: keys + small values only.
        kvs = ", ".join(f"{k}={v}" for k, v in list(data.items())[:6])
        return f"{head} | {kvs}"
    return head


def _render_event(ev: K8sEvent) -> str:
    parts = [f"[{_ts(ev.ts)}] lvl={ev.level or '-'}"]
    if ev.pod:
        parts.append(f"pod={ev.pod}")
    if ev.hostname:
        parts.append(f"host={ev.hostname}")
    if ev.reason:
        parts.append(f"reason={ev.reason}")
    head = " ".join(parts)
    if ev.message:
        head += f" | {ev.message}"
    return head


def _render_metric(s: MetricSeries) -> str:
    st = s.summary_stats()
    if st.get("count", 0) == 0:
        return f"{s.entity_name or s.entity_id} {s.metric}: <no points>"
    return (
        f"{s.entity_name or s.entity_id} ({s.domain}/{s.entity_type}) "
        f"{s.metric}: n={st['count']} min={st['min']:.4g} max={st['max']:.4g} "
        f"avg={st['avg']:.4g} last={st['last']:.4g}"
    )


def _render_log(line: LogLine) -> str:
    where = ":".join(p for p in [line.namespace, line.pod, line.container] if p)
    host = f" @host={line.host}" if line.host else ""
    return f"[{_ts(line.ts)}] {where}{host} | {line.content}"


def _render_span(sp: Span, depth: int = 0) -> str:
    dur_ms = (sp.duration_ns / 1e6) if sp.duration_ns is not None else None
    dur = f"{dur_ms:.1f}ms" if dur_ms is not None else "-"
    svc = f"[{sp.service}]" if sp.service else ""
    status = f" <{sp.status_code}>" if sp.status_code else ""
    msg = f" :: {sp.status_message}" if sp.status_message else ""
    return f"{'  ' * depth}- {sp.name}{svc} {dur}{status}{msg}"


def _render_trace(tr: Trace) -> str:
    slow = tr.slowest_span()
    if slow is not None and slow.duration_ns is not None:
        slow_txt = f"slowest={slow.name} {slow.duration_ns / 1e6:.1f}ms"
    else:
        slow_txt = "slowest=-"
    err_count = sum(1 for s in tr.spans if (s.status_code or "").upper() == "ERROR")
    err_txt = f"errors={err_count}" if err_count else "errors=0"
    head = f"trace={tr.trace_id} spans={len(tr.spans)} {slow_txt} {err_txt}"
    body = "\n".join(_render_span(s, 0) for s in tr.spans[:12])
    return head + ("\n" + body if body else "")


def _entity_label(e: dict[str, Any]) -> str:
    return str(e.get("name") or e.get("id") or "?")


def _entity_kind(e: dict[str, Any]) -> str:
    return str(e.get("type") or e.get("entity_type") or "-")


def _render_topology(sub: TopologySubgraph, max_entities: int = 60) -> str:
    ents = sub.entities[:max_entities]
    ent_lines = [f"- {_entity_label(e)} ({_entity_kind(e)}) id={e.get('id', '-')}" for e in ents]
    more = f"\n... (+{len(sub.entities) - len(ents)} more entities)" if len(sub.entities) > len(ents) else ""
    edges = sub.edges[:60]
    edge_lines = []
    for ed in edges:
        src = ed.get("source") or ed.get("from") or "-"
        dst = ed.get("target") or ed.get("to") or "-"
        rel = ed.get("relation") or ed.get("type") or "-"
        edge_lines.append(f"- {src} --{rel}--> {dst}")
    out = [f"entities({len(sub.entities)}):", *ent_lines, more]
    if sub.edges:
        out += [f"edges({len(sub.edges)}):", *edge_lines]
    return "\n".join(x for x in out if x)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def query_alerts(args: QueryAlertsArgs, provider: Any, memory: Any) -> dict[str, Any]:
    f = AlertFilter(window=provider.window, limit=args.limit)
    rows: list[CloudEvent] = provider.query_alerts(f)
    text = "\n".join(_render_alert(r) for r in rows) or "(no alerts in window)"
    raw = [r.model_dump() for r in _cap(rows)]
    return {"tool": "query_alerts", "count": len(rows), "text": text, "raw": raw}


def query_events(args: QueryEventsArgs, provider: Any, memory: Any) -> dict[str, Any]:
    f = EventFilter(
        window=provider.window,
        pod_names=[args.pod] if args.pod else None,
        levels=[args.level] if args.level else None,
        limit=args.limit,
    )
    rows: list[K8sEvent] = provider.query_events(f)
    text = "\n".join(_render_event(r) for r in rows) or "(no k8s events in window)"
    raw = [r.model_dump() for r in _cap(rows)]
    return {"tool": "query_events", "count": len(rows), "text": text, "raw": raw}


def query_metrics(args: QueryMetricsArgs, provider: Any, memory: Any) -> dict[str, Any]:
    f = MetricFilter(
        window=provider.window,
        services=[args.service] if args.service else None,
        metrics=[args.metric] if args.metric else None,
        entity_names=[args.entity_name] if args.entity_name else None,
        domains=[args.domain] if args.domain else None,
        limit=args.limit,
    )
    rows: list[MetricSeries] = provider.query_metrics(f)
    text = "\n".join(_render_metric(r) for r in rows) or "(no metric series in window)"
    # raw: strip the (potentially large) points arrays; keep summary stats.
    raw = [
        {"entity": r.entity_name or r.entity_id, "metric": r.metric, "stats": r.summary_stats()}
        for r in _cap(rows)
    ]
    return {"tool": "query_metrics", "count": len(rows), "text": text, "raw": raw}


def query_logs(args: QueryLogsArgs, provider: Any, memory: Any) -> dict[str, Any]:
    f = LogFilter(
        window=provider.window,
        pod_names=[args.pod] if args.pod else None,
        namespaces=[args.namespace] if args.namespace else None,
        containers=[args.container] if args.container else None,
        hosts=[args.host] if args.host else None,
        contains=args.contains,
        level_hint=args.level_hint,
        limit=args.limit,
    )
    rows: list[LogLine] = provider.query_logs(f)
    text = "\n".join(_render_log(r) for r in rows) or "(no logs in window)"
    raw = [{"ts": _ts(r.ts), "content": r.content} for r in _cap(rows)]
    return {"tool": "query_logs", "count": len(rows), "text": text, "raw": raw}


def query_traces(args: QueryTracesArgs, provider: Any, memory: Any) -> dict[str, Any]:
    f = TraceFilter(
        window=provider.window,
        service_names=[args.service] if args.service else None,
        span_names=[args.span] if args.span else None,
        status_codes=[args.status] if args.status else None,
        min_duration_ns=args.min_duration_ms * 1_000_000 if args.min_duration_ms is not None else None,
        limit=args.limit,
    )
    rows: list[Trace] = provider.query_traces(f)
    text = "\n".join(_render_trace(r) for r in rows) or "(no traces in window)"
    raw = [r.model_dump() for r in _cap(rows)]
    return {"tool": "query_traces", "count": len(rows), "text": text, "raw": raw}


def get_topology(args: GetTopologyArgs, provider: Any, memory: Any) -> dict[str, Any]:
    f = TopologyFilter(
        entity_names=[args.entity_name] if args.entity_name else None,
        entity_types=[args.entity_type] if args.entity_type else None,
        hops=args.hops,
        limit=args.limit,
    )
    sub: TopologySubgraph = provider.query_topology(f)
    text = _render_topology(sub)
    return {
        "tool": "get_topology",
        "count": len(sub.entities),
        "text": text,
        "raw": {"entities": _cap(sub.entities), "edges": _cap(sub.edges)},
    }


def inspect_entity(args: InspectEntityArgs, provider: Any, memory: Any) -> dict[str, Any]:
    if not args.entity_id and not args.entity_name:
        return {
            "tool": "inspect_entity",
            "count": 0,
            "text": "(inspect_entity requires entity_id or entity_name)",
            "raw": None,
        }
    f = TopologyFilter(
        entity_ids=[args.entity_id] if args.entity_id else None,
        entity_names=[args.entity_name] if args.entity_name else None,
        hops=args.hops,
        limit=500,
    )
    sub: TopologySubgraph = provider.query_topology(f)
    key = args.entity_id or args.entity_name
    target = next(
        (e for e in sub.entities if e.get("id") == key or e.get("name") == key), None
    )
    if target is None:
        text = f"(entity not found: {key}; window returned {len(sub.entities)} entities)"
        return {"tool": "inspect_entity", "count": 0, "text": text, "raw": None}
    neighbors = [e for e in sub.entities if e is not target]
    # Render props compactly.
    props = {k: v for k, v in target.items() if k not in ("id", "name", "type")}
    prop_str = ", ".join(f"{k}={v}" for k, v in list(props.items())[:12]) or "(none)"
    text = f"entity {_entity_label(target)} ({_entity_kind(target)}) id={target.get('id', '-')}\n"
    text += f"props: {prop_str}\n"
    text += f"neighbors({len(neighbors)}): " + (
        ", ".join(f"{_entity_label(n)}({_entity_kind(n)})" for n in neighbors[:20]) or "(none)"
    )
    return {
        "tool": "inspect_entity",
        "count": len(sub.entities),
        "text": text,
        "raw": {"entity": target, "neighbors": _cap(neighbors), "edges": _cap(sub.edges)},
    }


def store_observation(args: StoreObservationArgs, provider: Any, memory: Any) -> dict[str, Any]:
    case_id = getattr(provider, "case_id", "__global__")
    item = MemoryItem(
        id=f"obs-{uuid.uuid4().hex[:12]}",
        case_id=case_id,
        content=args.content,
        kind=args.kind,
        source_tool="store_observation",
        entities=list(args.entities),
    )
    if memory is not None:
        memory.index([item])
    return {"tool": "store_observation", "count": 1, "text": f"stored ({args.kind}): {args.content}", "raw": {"stored": True, "id": item.id}}


__all__ = [
    "QueryAlertsArgs",
    "QueryEventsArgs",
    "QueryMetricsArgs",
    "QueryLogsArgs",
    "QueryTracesArgs",
    "GetTopologyArgs",
    "InspectEntityArgs",
    "StoreObservationArgs",
    "query_alerts",
    "query_events",
    "query_metrics",
    "query_logs",
    "query_traces",
    "get_topology",
    "inspect_entity",
    "store_observation",
]
