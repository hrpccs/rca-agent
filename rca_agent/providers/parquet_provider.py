"""Parquet-backed :class:`DataProvider` for the rca100 benchmark.

Reads the per-case ``.parquet`` files (metrics/logs/traces/events/alerts) with
pyarrow, applying column selection + predicate pushdown so only the needed
columns/rows are materialized. Every ``query_*`` method maps the raw rows into
the frozen result models from :mod:`rca_agent.contracts.provider`, so the agent
tools never see backend-specific shapes.

The same files exist on disk for every case; if a file is missing or empty the
corresponding query returns ``[]`` (robustness over loud failures). All type
coercion is defensive: several columns are stored as JSON strings or numeric
strings and a few cells are ``None``.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..cases import load_case
from ..contracts._primitives import Modality, TimeWindow
from ..contracts.dataset import Case
from ..contracts.provider import (
    AlertFilter,
    CloudEvent,
    EventFilter,
    K8sEvent,
    LogFilter,
    LogLine,
    MetricFilter,
    MetricSeries,
    Span,
    TopologyFilter,
    TopologySubgraph,
    Trace,
    TraceFilter,
)

__all__ = ["ParquetProvider", "render"]

logger = logging.getLogger(__name__)

# Bounded LRU table cache size (env-tunable, default 64). 0 is treated as the
# default so a misconfigured env var never silently disables caching.
_CACHE_MAX_ENV = "RCA_PARQUET_CACHE_MAX"
_CACHE_MAX_DEFAULT = 64

# Pyarrow / parquet IO + parse errors we treat as "this file is unusable".
# ArrowException is the base class of all pyarrow errors.
_PARQUET_IO_ERRORS: tuple[type[BaseException], ...] = (
    pa.ArrowException,
    OSError,
    ValueError,
)

# Errors that the in-memory Table -> Python conversion (to_pylist) can raise
# that are NOT pyarrow IO errors. to_pylist is pure data conversion (e.g. it
# raises KeyError on map columns with duplicate keys, TypeError on
# non-convertible cells), so we keep a wider net here than on the read path.
_PYLIST_ERRORS: tuple[type[BaseException], ...] = (
    pa.ArrowException,
    ValueError,
    KeyError,
    TypeError,
)


# --------------------------------------------------------------------------- #
# Small coercion helpers
# --------------------------------------------------------------------------- #
def _as_int(v: Any) -> int | None:
    """Best-effort int coercion (handles str/float/None/numpy scalars)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return None
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return None
    return None


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v) if not math.isnan(v) else None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            f = float(s)
            return f if not math.isnan(f) else None
        except ValueError:
            return None
    return None


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v)
    return s if s else None


def _parse_json_obj(v: Any) -> dict[str, Any]:
    """Parse a JSON-string column into a dict; never raises."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            return {}
        return obj if isinstance(obj, dict) else {}
    return {}


def _parse_iso(v: Any) -> datetime | None:
    """Parse an ISO 8601 string (with optional Z / +0800 offsets) to datetime."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _window_us(window: TimeWindow) -> tuple[int, int] | None:
    """Return (start_us, end_us) for a window; None if unbounded (epoch-based)."""
    start = window.start_us
    end = window.end_us
    if start is None:
        start = int(window.start.timestamp() * 1_000_000) if window.start else None
    if end is None:
        end = int(window.end.timestamp() * 1_000_000) if window.end else None
    if start is None or end is None:
        return None
    return start, end


def _in_range_us(ts_us: int | None, lo: int, hi: int) -> bool:
    return ts_us is not None and lo <= ts_us < hi


def _in_range_dt(dt: datetime | None, lo: datetime, hi: datetime) -> bool:
    return dt is not None and lo <= dt < hi


class ParquetProvider:
    """Reads benchmark parquet files into the contract result models."""

    def __init__(self, case: Case) -> None:
        self.case: Case = case
        self.case_id: str = case.task.task_id
        self.window: TimeWindow = case.task.alert_window
        self.case_dir: Path = Path(case.case_dir)
        # Bounded LRU table cache keyed by (filename, frozenset(columns)).
        # Caching is read-through: a miss reads from disk, an evicted entry is
        # dropped least-recently-used. Parsed ONCE at init from the env var; the
        # default preserves the original caching benefit while bounding memory
        # for long-lived providers.
        self._table_cache: OrderedDict[tuple[str, frozenset[str]], Any] = OrderedDict()
        self._cache_max: int = self._parse_cache_max(
            os.environ.get(_CACHE_MAX_ENV, str(_CACHE_MAX_DEFAULT))
        )

    @staticmethod
    def _parse_cache_max(raw: str) -> int:
        """Parse ``RCA_PARQUET_CACHE_MAX`` once at init.

        Non-positive or unparseable values resolve to the default so a
        misconfigured env var never silently disables caching.
        """
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            return _CACHE_MAX_DEFAULT
        return value if value > 0 else _CACHE_MAX_DEFAULT

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_case(cls, case_id: str, cases_dir: str | Path | None = None) -> ParquetProvider:
        return cls(load_case(case_id, cases_dir))

    # -- modalities --------------------------------------------------------- #
    def modalities(self) -> list[Modality]:
        return list(self.case.modalities) if self.case.modalities else list(Modality)

    # -- parquet IO --------------------------------------------------------- #
    def _read(self, name: str, columns: list[str]) -> list[dict[str, Any]]:
        """Read selected columns of a case parquet file into list-of-dicts.

        Returns [] if the file is missing/empty/unreadable. Columns absent
        from the file are dropped from the projection so missing columns never
        raise. Results are cached in a bounded LRU keyed by (name, columns).
        """
        path = self.case_dir / name
        if not path.exists():
            return []
        key = (name, frozenset(columns))
        table = self._table_cache.get(key)
        if table is not None:
            # Cache hit: mark most-recently-used (move to end) for LRU eviction.
            self._table_cache.move_to_end(key)
        else:
            try:
                schema = pq.ParquetFile(path).schema_arrow
                present = [c for c in columns if c in schema.names]
                table = pq.read_table(path, columns=present) if present else pq.read_table(path)
            except _PARQUET_IO_ERRORS as exc:
                logger.warning(
                    "parquet read failed case=%s file=%s error=%s",
                    self.case_id, name, exc,
                )
                return []
            self._table_cache[key] = table
            if len(self._table_cache) > self._cache_max:
                self._table_cache.popitem(last=False)  # evict least-recently-used
        try:
            return table.to_pylist()
        except _PYLIST_ERRORS as exc:
            logger.warning(
                "parquet to_pylist failed case=%s file=%s error=%s",
                self.case_id, name, exc,
            )
            return []

    # ------------------------------------------------------------------ #
    # metrics
    # ------------------------------------------------------------------ #
    def query_metrics(self, f: MetricFilter) -> list[MetricSeries]:
        cols = [
            "time", "domain", "entity_id", "entity_name", "metric", "value",
            "metric_set_id", "service",
        ]
        rows = self._read("metrics.parquet", cols)

        win = _window_us(f.window)
        e_ids = set(f.entity_ids) if f.entity_ids else None
        e_names = set(f.entity_names) if f.entity_names else None
        domains = set(f.domains) if f.domains else None
        services = set(f.services) if f.services else None
        metrics = set(f.metrics) if f.metrics else None
        # entity_types is not a column in the parquet (it lives in topology);
        # match against the domain/dataset when callers pass types like "k8s".
        e_types = set(f.entity_types) if f.entity_types else None

        grouped: dict[tuple[str, str], MetricSeries] = {}
        limit = f.limit if f.limit and f.limit > 0 else 5000

        for r in rows:
            t_us = _as_int(r.get("time"))
            if win is not None and not _in_range_us(t_us, win[0], win[1]):
                continue
            domain = _as_str(r.get("domain")) or ""
            ent_id = _as_str(r.get("entity_id")) or ""
            ent_name = _as_str(r.get("entity_name")) or ""
            metric = _as_str(r.get("metric")) or ""
            service = _as_str(r.get("service"))

            if domains is not None and domain not in domains:
                continue
            if e_ids is not None and ent_id not in e_ids:
                continue
            if e_names is not None and ent_name not in e_names:
                continue
            if metrics is not None and metric not in metrics:
                continue
            if services is not None and (service is None or service not in services):
                continue
            if e_types is not None and domain not in e_types:
                continue

            val = _as_float(r.get("value"))
            if val is None or t_us is None:
                continue

            key = (ent_id, metric)
            ms = grouped.get(key)
            if ms is None:
                ms = MetricSeries(
                    entity_id=ent_id,
                    entity_name=ent_name,
                    entity_type=domain,
                    service=service,
                    domain=domain,
                    metric=metric,
                    metric_set_id=_as_str(r.get("metric_set_id")),
                    points=[],
                )
                grouped[key] = ms
            if len(ms.points) < limit:
                ms.points.append((t_us, val))

        # Bound overall series count too.
        result = list(grouped.values())
        if len(result) > limit:
            result = result[:limit]
        # Sort each series' points by time for stable downstream use.
        for ms in result:
            ms.points.sort(key=lambda p: p[0])
        return result

    # ------------------------------------------------------------------ #
    # logs
    # ------------------------------------------------------------------ #
    def query_logs(self, f: LogFilter) -> list[LogLine]:
        cols = [
            "content", "_time_", "_container_name_", "_pod_name_", "_namespace_",
            "__tag__:__hostname__",
        ]
        rows = self._read("logs.parquet", cols)

        pods = set(f.pod_names) if f.pod_names else None
        namespaces = set(f.namespaces) if f.namespaces else None
        containers = set(f.containers) if f.containers else None
        hosts = set(f.hosts) if f.hosts else None
        contains = f.contains.lower() if f.contains else None
        level = f.level_hint.upper() if f.level_hint else None
        lo, hi = f.window.start, f.window.end

        out: list[LogLine] = []
        limit = f.limit if f.limit and f.limit > 0 else 200
        for r in rows:
            content = _as_str(r.get("content")) or ""
            ts = _parse_iso(r.get("_time_"))
            if not _in_range_dt(ts, lo, hi):
                continue
            pod = _as_str(r.get("_pod_name_"))
            ns = _as_str(r.get("_namespace_"))
            container = _as_str(r.get("_container_name_"))
            host = _as_str(r.get("__tag__:__hostname__"))

            if pods is not None and (pod is None or pod not in pods):
                continue
            if namespaces is not None and (ns is None or ns not in namespaces):
                continue
            if containers is not None and (container is None or container not in containers):
                continue
            if hosts is not None and (host is None or host not in hosts):
                continue
            if contains is not None and contains not in content.lower():
                continue
            if level is not None and level not in content.upper():
                continue

            out.append(LogLine(ts=ts, pod=pod, namespace=ns, container=container, host=host, content=content))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------ #
    # traces
    # ------------------------------------------------------------------ #
    def query_traces(self, f: TraceFilter) -> list[Trace]:
        cols = [
            "traceId", "spanId", "parentSpanId", "kind", "spanName",
            "startTime", "endTime", "duration", "serviceName", "statusCode",
            "statusMessage", "resources", "attributes",
        ]
        rows = self._read("traces.parquet", cols)

        trace_ids = set(f.trace_ids) if f.trace_ids else None
        services = set(f.service_names) if f.service_names else None
        span_names = set(f.span_names) if f.span_names else None
        status_codes = set(f.status_codes) if f.status_codes else None
        min_dur = f.min_duration_ns

        win = _window_us(f.window)
        lo_ns = win[0] * 1_000 if win else None
        hi_ns = win[1] * 1_000 if win else None

        grouped: dict[str, Trace] = {}
        # preserve first-seen order of traces
        order: list[str] = []
        limit = f.limit if f.limit and f.limit > 0 else 50

        for r in rows:
            tid = _as_str(r.get("traceId")) or ""
            if not tid:
                continue
            if trace_ids is not None and tid not in trace_ids:
                continue
            start_ns = _as_int(r.get("startTime"))
            if (
                lo_ns is not None
                and hi_ns is not None
                and (start_ns is None or not (lo_ns <= start_ns < hi_ns))
            ):
                continue
            svc = _as_str(r.get("serviceName"))
            sname = _as_str(r.get("spanName"))
            sc = _as_str(r.get("statusCode"))
            if services is not None and (svc is None or svc not in services):
                continue
            if span_names is not None and (sname is None or sname not in span_names):
                continue
            if status_codes is not None and (sc is None or sc not in status_codes):
                continue
            dur = _as_int(r.get("duration"))
            if min_dur is not None and (dur is None or dur < min_dur):
                continue

            span = Span(
                trace_id=tid,
                span_id=_as_str(r.get("spanId")) or "",
                parent_span_id=_as_str(r.get("parentSpanId")),
                kind=_as_str(r.get("kind")),
                name=sname or "",
                service=svc,
                start_ns=start_ns,
                end_ns=_as_int(r.get("endTime")),
                duration_ns=dur,
                status_code=sc,
                status_message=_as_str(r.get("statusMessage")),
                attributes=_parse_json_obj(r.get("attributes")),
                resources=_parse_json_obj(r.get("resources")),
            )
            tr = grouped.get(tid)
            if tr is None:
                tr = Trace(trace_id=tid)
                grouped[tid] = tr
                order.append(tid)
            tr.spans.append(span)
            # NOTE: do NOT break early — span rows are interleaved across
            # traces in the file (verified: ~9 traceId changes per unique
            # trace), so a trace's spans appear scattered throughout. We must
            # scan the whole (filtered) set, then truncate by first-seen order.

        # Limit by number of traces (first-seen order).
        return [grouped[t] for t in order[:limit]]

    # ------------------------------------------------------------------ #
    # events (k8s)
    # ------------------------------------------------------------------ #
    def query_events(self, f: EventFilter) -> list[K8sEvent]:
        cols = ["eventId", "hostname", "level", "pod_name", "clusterId", "clusterName"]
        rows = self._read("events.parquet", cols)

        pods = set(f.pod_names) if f.pod_names else None
        levels = set(f.levels) if f.levels else None
        clusters = set(f.cluster_ids) if f.cluster_ids else None
        hosts = set(f.hosts) if f.hosts else None
        lo, hi = f.window.start, f.window.end

        out: list[K8sEvent] = []
        limit = f.limit if f.limit and f.limit > 0 else 200
        for r in rows:
            level = _as_str(r.get("level"))
            pod = _as_str(r.get("pod_name"))
            host = _as_str(r.get("hostname"))
            cluster = _as_str(r.get("clusterId"))

            if pods is not None and (pod is None or pod not in pods):
                continue
            if levels is not None and (level is None or level not in levels):
                continue
            if clusters is not None and (cluster is None or cluster not in clusters):
                continue
            if hosts is not None and (host is None or host not in hosts):
                continue

            meta = _parse_json_obj(r.get("eventId"))
            reason = _as_str(meta.get("reason"))
            message = _as_str(meta.get("message"))
            # creationTimestamp lives under metadata.metadata; use it as ts.
            inner = meta.get("metadata")
            ts_raw = None
            if isinstance(inner, dict):
                ts_raw = inner.get("creationTimestamp")
            ts = _parse_iso(ts_raw)
            # Window filter: only drop rows whose timestamp is parseable and
            # outside the window (k8s events may legitimately predate it).
            if ts is not None and not _in_range_dt(ts, lo, hi):
                continue

            out.append(K8sEvent(
                ts=ts, level=level, pod=pod, cluster_id=cluster,
                hostname=host, metadata=meta, reason=reason, message=message,
            ))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------ #
    # alerts (cloud events)
    # ------------------------------------------------------------------ #
    def query_alerts(self, f: AlertFilter) -> list[CloudEvent]:
        cols = [
            "id", "type", "subtype", "source", "time", "timestamp", "subject",
            "severity", "status", "resource", "labels", "annotations", "data",
            "dataschema", "datacontenttype", "time_s", "specversion",
        ]
        rows = self._read("alerts.parquet", cols)

        severities = set(f.severities) if f.severities else None
        subtypes = set(f.subtypes) if f.subtypes else None
        subjects = set(f.subjects) if f.subjects else None
        lo, hi = f.window.start, f.window.end

        out: list[CloudEvent] = []
        limit = f.limit if f.limit and f.limit > 0 else 50
        for r in rows:
            ts = _parse_iso(r.get("time"))
            if ts is None:
                # fall back to time_s / timestamp(ms)
                t_s = _as_int(r.get("time_s"))
                if t_s is not None:
                    ts = datetime.fromtimestamp(t_s, tz=UTC)
            if not _in_range_dt(ts, lo, hi):
                continue

            sev = _as_str(r.get("severity"))
            sub = _as_str(r.get("subtype"))
            subj = _as_str(r.get("subject"))
            if severities is not None and (sev is None or sev not in severities):
                continue
            if subtypes is not None and (sub is None or sub not in subtypes):
                continue
            if subjects is not None and (subj is None or subj not in subjects):
                continue

            out.append(CloudEvent(
                id=_as_str(r.get("id")) or "",
                type=_as_str(r.get("type")) or "",
                subtype=sub,
                severity=sev,
                status=_as_str(r.get("status")),
                subject=subj,
                ts=ts,
                resource=_parse_json_obj(r.get("resource")),
                labels=_parse_json_obj(r.get("labels")),
                annotations=_parse_json_obj(r.get("annotations")),
                data=_parse_json_obj(r.get("data")),
            ))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------ #
    # topology
    # ------------------------------------------------------------------ #
    def query_topology(self, f: TopologyFilter) -> TopologySubgraph:
        topo = self.case.topology
        entities = [e.model_dump() for e in topo.entities]
        edges = [e.model_dump() for e in topo.edges]

        e_ids = set(f.entity_ids) if f.entity_ids else None
        e_types = set(f.entity_types) if f.entity_types else None
        e_names = set(f.entity_names) if f.entity_names else None
        relations = set(f.relations) if f.relations else None
        hops = f.hops if f.hops is not None else 1
        limit = f.limit if f.limit and f.limit > 0 else 500

        # If no selector is given, return the whole (bounded) graph.
        if e_ids is None and e_types is None and e_names is None:
            sel_entities = entities
            sel_edges = edges
        else:
            # Seed: entities matching any selector.
            seed_ids: set[str] = set()
            for e in entities:
                eid = e.get("id")
                if eid is None:
                    continue
                if e_ids is not None and eid in e_ids:
                    seed_ids.add(eid)
                    continue
                if e_types is not None and e.get("type") in e_types:
                    seed_ids.add(eid)
                    continue
                if e_names is not None and e.get("name") in e_names:
                    seed_ids.add(eid)
                    continue
            # BFS over edges (undirected neighborhood).
            keep = set(seed_ids)
            frontier = set(seed_ids)
            for _ in range(max(hops, 0)):
                if not frontier:
                    break
                nxt: set[str] = set()
                for edge in edges:
                    s, d = edge.get("src"), edge.get("dst")
                    if s in frontier and d not in keep:
                        nxt.add(d)
                    if d in frontier and s not in keep:
                        nxt.add(s)
                keep |= nxt
                frontier = nxt
            sel_entities = [e for e in entities if e.get("id") in keep]
            sel_edges = [
                e for e in edges
                if e.get("src") in keep and e.get("dst") in keep
            ]

        if relations is not None:
            sel_edges = [e for e in sel_edges if e.get("relation") in relations]

        if len(sel_entities) > limit:
            sel_entities = sel_entities[:limit]
        return TopologySubgraph(entities=sel_entities, edges=sel_edges)


# --------------------------------------------------------------------------- #
# render(): turn query results into concise structured text for the LLM
# --------------------------------------------------------------------------- #
def _fmt_dt(dt: datetime | None) -> str:
    return dt.isoformat() if dt is not None else "-"


def render_metrics(series: list[MetricSeries], max_series: int = 40, max_points: int = 12) -> str:
    if not series:
        return "metrics: (none)"
    lines = [f"metrics: {len(series)} series"]
    for ms in series[:max_series]:
        st = ms.summary_stats()
        pts = ms.points[:max_points]
        pts_s = ", ".join(f"{t}:{v:g}" for t, v in pts) or "-"
        lines.append(
            f"  - entity={ms.entity_id or ms.entity_name or '?'} "
            f"metric={ms.metric} domain={ms.domain or '-'} "
            f"service={ms.service or '-'} "
            f"count={int(st.get('count',0))} min={st.get('min','-')} "
            f"max={st.get('max','-')} avg={st.get('avg','-')} "
            f"last={st.get('last','-')} pts=[{pts_s}]"
        )
    if len(series) > max_series:
        lines.append(f"  ... ({len(series) - max_series} more series)")
    return "\n".join(lines)


def render_logs(logs: list[LogLine], max_lines: int = 60, width: int = 300) -> str:
    if not logs:
        return "logs: (none)"
    lines = [f"logs: {len(logs)} lines"]
    for lg in logs[:max_lines]:
        content = lg.content if len(lg.content) <= width else lg.content[:width] + "..."
        lines.append(
            f"  - [{_fmt_dt(lg.ts)}] pod={lg.pod or '-'} ns={lg.namespace or '-'} "
            f"cnt={lg.container or '-'} host={lg.host or '-'} :: {content}"
        )
    if len(logs) > max_lines:
        lines.append(f"  ... ({len(logs) - max_lines} more lines)")
    return "\n".join(lines)


def render_traces(traces: list[Trace], max_traces: int = 30, max_spans: int = 20) -> str:
    if not traces:
        return "traces: (none)"
    lines = [f"traces: {len(traces)}"]
    for tr in traces[:max_traces]:
        slow = tr.slowest_span()
        slow_s = (
            f"slowest={slow.name} dur={slow.duration_ns}ns svc={slow.service or '-'}"
            if slow is not None else "slowest=-"
        )
        lines.append(f"  - trace={tr.trace_id} spans={len(tr.spans)} {slow_s}")
        for sp in tr.spans[:max_spans]:
            lines.append(
                f"      * span={sp.span_id} parent={sp.parent_span_id or '-'} "
                f"name={sp.name} svc={sp.service or '-'} kind={sp.kind or '-'} "
                f"dur={sp.duration_ns}ns status={sp.status_code or '-'}"
            )
        if len(tr.spans) > max_spans:
            lines.append(f"      ... ({len(tr.spans) - max_spans} more spans)")
    if len(traces) > max_traces:
        lines.append(f"  ... ({len(traces) - max_traces} more traces)")
    return "\n".join(lines)


def render_events(events: list[K8sEvent], max_events: int = 60, width: int = 300) -> str:
    if not events:
        return "events: (none)"
    lines = [f"events: {len(events)}"]
    for ev in events[:max_events]:
        msg = ev.message or ""
        msg = msg if len(msg) <= width else msg[:width] + "..."
        lines.append(
            f"  - [{_fmt_dt(ev.ts)}] level={ev.level or '-'} "
            f"pod={ev.pod or '-'} host={ev.hostname or '-'} "
            f"reason={ev.reason or '-'} :: {msg}"
        )
    if len(events) > max_events:
        lines.append(f"  ... ({len(events) - max_events} more events)")
    return "\n".join(lines)


def render_alerts(alerts: list[CloudEvent], max_alerts: int = 50, width: int = 300) -> str:
    if not alerts:
        return "alerts: (none)"
    lines = [f"alerts: {len(alerts)}"]
    for al in alerts[:max_alerts]:
        subj = al.subject or ""
        subj = subj if len(subj) <= width else subj[:width] + "..."
        res = al.resource.get("entity", {}) if isinstance(al.resource, dict) else {}
        ent = ""
        if isinstance(res, dict):
            ent = res.get("entity_id") or res.get("entity_type") or ""
        lines.append(
            f"  - [{_fmt_dt(al.ts)}] id={al.id} sev={al.severity or '-'} "
            f"sub={al.subtype or '-'} status={al.status or '-'} "
            f"entity={ent} :: {subj}"
        )
    if len(alerts) > max_alerts:
        lines.append(f"  ... ({len(alerts) - max_alerts} more alerts)")
    return "\n".join(lines)


def render_topology(sub: TopologySubgraph, max_entities: int = 120, max_edges: int = 200) -> str:
    lines = [f"topology: {len(sub.entities)} entities, {len(sub.edges)} edges"]
    shown_e = sub.entities[:max_entities]
    for e in shown_e:
        props = e.get("props", {}) or {}
        extra = ""
        if isinstance(props, dict):
            extra = " ".join(f"{k}={v}" for k, v in list(props.items())[:3])
        lines.append(
            f"  - {e.get('type', '?')} {e.get('name', '?')} id={e.get('id', '?')} {extra}"
        )
    if len(sub.entities) > max_entities:
        lines.append(f"  ... ({len(sub.entities) - max_entities} more entities)")
    shown_ed = sub.edges[:max_edges]
    for ed in shown_ed:
        lines.append(
            f"  ~ {ed.get('src_type', '?')}:{ed.get('src', '?')} "
            f"--[{ed.get('relation', '?')}]--> "
            f"{ed.get('dst_type', '?')}:{ed.get('dst', '?')}"
        )
    if len(sub.edges) > max_edges:
        lines.append(f"  ... ({len(sub.edges) - max_edges} more edges)")
    return "\n".join(lines)


def render(result: Any) -> str:
    """Dispatch render based on the result type."""
    if isinstance(result, list):
        if not result:
            return "(empty)"
        first = result[0]
        if isinstance(first, MetricSeries):
            return render_metrics(result)
        if isinstance(first, LogLine):
            return render_logs(result)
        if isinstance(first, Trace):
            return render_traces(result)
        if isinstance(first, K8sEvent):
            return render_events(result)
        if isinstance(first, CloudEvent):
            return render_alerts(result)
        # Fallback: stringify list items.
        return "\n".join(str(x) for x in result[:200])
    if isinstance(result, TopologySubgraph):
        return render_topology(result)
    return str(result)
