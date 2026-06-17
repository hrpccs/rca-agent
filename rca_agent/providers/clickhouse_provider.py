"""ClickHouse :class:`DataProvider` implementation.

Mirrors the parquet provider's result models but pulls the same data from the
``rca`` ClickHouse database after import. Every query is parameterised, filtered
by ``case_id`` AND the filter fields AND the time window, and capped by the
filter ``limit``. JSON-string columns (resources/attributes/props/eventId/
alert resource/labels/annotations/data) are parsed into the contract result
models.

Construction:
    ClickhouseProvider(case_id)           # uses get_settings().clickhouse_dsn()
    ClickhouseProvider.from_case(case_id) # classmethod alternative

A query/socket timeout (``RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC``, default 30s) is
applied at client creation via clickhouse-connect's ``send_receive_timeout``
(the HTTP read/socket timeout). A failure on any modality query is caught,
logged structurally, and surfaced as an empty result rather than crashing the
agent loop — the other modalities still run.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import clickhouse_connect

from rca_agent.config import get_settings
from rca_agent.contracts import (
    AlertFilter,
    CloudEvent,
    EventFilter,
    K8sEvent,
    LogFilter,
    LogLine,
    MetricFilter,
    MetricSeries,
    Modality,
    Span,
    TimeWindow,
    TopologyFilter,
    TopologySubgraph,
    Trace,
    TraceFilter,
)

logger = logging.getLogger(__name__)

# Default socket/read timeout (seconds) handed to clickhouse-connect as
# ``send_receive_timeout``. Overridable via env without touching config.py.
_DEFAULT_QUERY_TIMEOUT_SEC = 30

_SCHEMA_PATH = Path(__file__).with_name("clickhouse_schema.sql")


def _strip_line_comments(sql: str) -> str:
    """Remove ``--`` line comments so statement splitting isn't fooled by
    comment-only fragments (ClickHouse rejects empty queries)."""
    out_lines = []
    for line in sql.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("--"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def _split_sql_statements(sql: str) -> list[str]:
    """Split a multi-statement SQL script into executable statements.

    Splits on ``;`` after dropping ``--`` line comments and filters out
    statements that are empty once comments are removed.
    """
    cleaned = _strip_line_comments(sql)
    return [s.strip() for s in cleaned.split(";") if s.strip()]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_json(s: Any) -> dict[str, Any]:
    """Parse a JSON-string column to a dict; return {} on any failure."""
    if not s:
        return {}
    if isinstance(s, dict):
        return s
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {"value": out}
    except (ValueError, TypeError):
        return {}


def _window_us(window: TimeWindow) -> tuple[int, int]:
    """Return [start, end] epoch microseconds for a window."""
    start_us = window.start_us if window.start_us is not None else int(
        window.start.timestamp() * 1_000_000
    )
    end_us = window.end_us if window.end_us is not None else int(
        window.end.timestamp() * 1_000_000
    )
    return start_us, end_us


def _window_ns(window: TimeWindow) -> tuple[int, int]:
    """Return [start, end] epoch nanoseconds for a window."""
    s, e = _window_us(window)
    return s * 1_000, e * 1_000


def _window_datetimes(window: TimeWindow) -> tuple[datetime, datetime]:
    """Return tz-aware [start, end] datetimes for DateTime columns."""
    start = window.start
    end = window.end
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return start, end


def _in_clause(values: list[str] | None) -> tuple[str | None, list[str]]:
    """Build a `col IN (...)` placeholder sequence. Returns (placeholder, params).

    Returns (None, []) when there is nothing to filter on so callers can skip the
    clause entirely.
    """
    if not values:
        return None, []
    ph = ", ".join(["%s"] * len(values))
    return ph, list(values)


def _query_timeout_sec() -> int:
    """Return the ClickHouse query/socket timeout in seconds.

    Read from ``RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC`` (env); falls back to the
    module default (30). Invalid values fall back to the default rather than
    raising so a misconfigured env can't brick the provider.
    """
    raw = os.environ.get("RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC")
    if not raw:
        return _DEFAULT_QUERY_TIMEOUT_SEC
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_QUERY_TIMEOUT_SEC
    return v if v > 0 else _DEFAULT_QUERY_TIMEOUT_SEC


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #
class ClickhouseProvider:
    """Production :class:`DataProvider` backed by ClickHouse (db ``rca``)."""

    def __init__(
        self,
        case_id: str,
        dsn: dict[str, Any] | None = None,
        window: TimeWindow | None = None,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.case_id = case_id
        # The investigation tools read `provider.window` (the case alert window)
        # to scope every query by default. ParquetProvider sets this from the
        # Case; mirror it here so the ClickHouse backend isn't blind to time.
        # Optional — standalone/schema-only construction leaves it unset.
        self.window: TimeWindow | None = window
        if client is not None:
            # Direct injection (tests / DI). Bypass env-derived construction.
            self._client = client
            return
        cfg = dict(dsn if dsn is not None else get_settings().clickhouse_dsn())
        # Apply the query/socket timeout at client creation. clickhouse-connect
        # accepts ``send_receive_timeout`` (HTTP read + socket timeout). If the
        # caller already supplied it via dsn, respect their value.
        cfg.setdefault("send_receive_timeout", _query_timeout_sec())
        factory = client_factory or clickhouse_connect.get_client
        # clickhouse_connect uses port 8123 (HTTP) by default; the dsn from
        # settings already carries the configured port.
        self._client = factory(**cfg)

    # -- Protocol construction alternative ------------------------------- #
    @classmethod
    def from_case(cls, case_id: str) -> ClickhouseProvider:
        return cls(case_id)

    # -- Internal: run a query, swallowing connection errors -------------- #
    def _query_rows(self, sql: str, params: list[Any]) -> list[tuple]:
        """Execute ``self._client.query(sql, params)`` returning ``result_rows``.

        Any exception from the client (connection refused, timeout, socket
        error, ClickHouse server error) is caught, logged structurally with the
        modality context, and an empty row list is returned — the caller then
        maps that to an empty result. This keeps a single backend failure from
        aborting the whole RCA loop; the other modalities still run.
        """
        try:
            rows = self._client.query(sql, params).result_rows
            # result_rows is already a concrete list in clickhouse-connect; the
            # `or []` only guarantees the empty-list contract if a client ever
            # returns None.
            return rows or []
        except Exception as exc:  # noqa: BLE001 — provider must not crash callers
            logger.error(
                "clickhouse.query_failed",
                extra={
                    "case_id": self.case_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "sql_head": sql[:120],
                },
            )
            return []

    # -- Schema bootstrap (convenience; not part of the Protocol) -------- #
    def ensure_schema(self) -> None:
        """Apply clickhouse_schema.sql if tables are missing (idempotent).

        Statements that become empty after stripping ``--`` line comments are
        skipped (ClickHouse rejects "empty query").
        """
        for stmt in _split_sql_statements(_SCHEMA_PATH.read_text()):
            self._client.command(stmt)

    # ------------------------------------------------------------------ #
    # Protocol: modality discovery
    # ------------------------------------------------------------------ #
    def modalities(self) -> list[Modality]:
        return [m for m in Modality]

    # ------------------------------------------------------------------ #
    # Protocol: queries
    # ------------------------------------------------------------------ #
    def query_metrics(self, f: MetricFilter) -> list[MetricSeries]:
        start_us, end_us = _window_us(f.window)
        where = ["case_id = %s", "time >= %s", "time < %s"]
        params: list[Any] = [self.case_id, start_us, end_us]

        for col, vals in (
            ("entity_id", f.entity_ids),
            ("entity_name", f.entity_names),
            ("service", f.services),
            ("metric", f.metrics),
            ("domain", f.domains),
        ):
            ph, p = _in_clause(vals)
            if ph:
                where.append(f"{col} IN ({ph})")
                params.extend(p)

        # entity_types isn't a direct column; it lives in entity_set/domain. We
        # treat it as a soft filter on entity_name contains when provided, but
        # since the column is absent we skip it rather than guess. (No-op guard.)
        sql = (
            "SELECT entity_id, entity_name, entity_set, service, domain, metric, "
            "metric_set_id, time, value "
            "FROM metrics WHERE "
            + " AND ".join(where)
            + " ORDER BY entity_id, metric, time "
            f"LIMIT {int(f.limit)}"
        )
        rows = self._query_rows(sql, params)

        # Group points into MetricSeries.
        series: dict[tuple[str, str], MetricSeries] = {}
        for entity_id, entity_name, entity_set, service, domain, metric, metric_set_id, t, value in rows:
            key = (entity_id or entity_name, metric)
            ms = series.get(key)
            if ms is None:
                ms = MetricSeries(
                    entity_id=entity_id or "",
                    entity_name=entity_name or "",
                    entity_type=entity_set or "",
                    service=service,
                    domain=domain or "",
                    metric=metric,
                    metric_set_id=metric_set_id,
                )
                series[key] = ms
            ms.points.append((int(t), float(value)))
        return list(series.values())

    def query_logs(self, f: LogFilter) -> list[LogLine]:
        start_dt, end_dt = _window_datetimes(f.window)
        where = ["case_id = %s", "`_time_` >= %s", "`_time_` < %s"]
        params: list[Any] = [self.case_id, start_dt, end_dt]

        for col, vals in (
            ("`_pod_name_`", f.pod_names),
            ("`_namespace_`", f.namespaces),
            ("`_container_name_`", f.containers),
            ("`__hostname__`", f.hosts),
        ):
            ph, p = _in_clause(vals)
            if ph:
                where.append(f"{col} IN ({ph})")
                params.extend(p)
        if f.contains:
            where.append("positionCaseInsensitive(content, %s) > 0")
            params.append(f.contains)
        if f.level_hint:
            where.append("positionCaseInsensitive(content, %s) > 0")
            params.append(f.level_hint)

        sql = (
            "SELECT content, `_time_`, `_pod_name_`, `_namespace_`, `_container_name_`, `__hostname__` "
            "FROM logs WHERE "
            + " AND ".join(where)
            + " ORDER BY `_time_` "
            f"LIMIT {int(f.limit)}"
        )
        rows = self._query_rows(sql, params)
        out: list[LogLine] = []
        for content, t, pod, ns, ctr, host in rows:
            out.append(
                LogLine(
                    ts=t if isinstance(t, datetime) else None,
                    pod=pod or None,
                    namespace=ns or None,
                    container=ctr or None,
                    host=host or None,
                    content=content or "",
                )
            )
        return out

    def query_traces(self, f: TraceFilter) -> list[Trace]:
        start_ns, end_ns = _window_ns(f.window)
        where = ["case_id = %s", "startTime >= %s", "startTime < %s"]
        params: list[Any] = [self.case_id, start_ns, end_ns]

        for col, vals in (
            ("traceId", f.trace_ids),
            ("serviceName", f.service_names),
            ("spanName", f.span_names),
            ("statusCode", f.status_codes),
        ):
            ph, p = _in_clause(vals)
            if ph:
                where.append(f"{col} IN ({ph})")
                params.extend(p)
        if f.min_duration_ns:
            where.append("duration >= %s")
            params.append(int(f.min_duration_ns))

        sql = (
            "SELECT traceId, spanId, parentSpanId, kind, spanName, startTime, "
            "endTime, duration, serviceName, statusCode, statusMessage, resources, attributes "
            "FROM traces WHERE "
            + " AND ".join(where)
            + " ORDER BY traceId, startTime "
            f"LIMIT {int(f.limit)}"
        )
        rows = self._query_rows(sql, params)

        traces: dict[str, Trace] = {}
        for (
            trace_id, span_id, parent, kind, name, st, et, dur, svc, sc, sm, res, attrs
        ) in rows:
            span = Span(
                trace_id=trace_id or "",
                span_id=span_id or "",
                parent_span_id=parent or None,
                kind=kind or None,
                name=name or "",
                service=svc or None,
                start_ns=int(st) if st is not None else None,
                end_ns=int(et) if et is not None else None,
                duration_ns=int(dur) if dur is not None else None,
                status_code=sc or None,
                status_message=sm or None,
                resources=_safe_json(res),
                attributes=_safe_json(attrs),
            )
            traces.setdefault(span.trace_id, Trace(trace_id=span.trace_id)).spans.append(span)
        return list(traces.values())

    def query_events(self, f: EventFilter) -> list[K8sEvent]:
        start_dt, end_dt = _window_datetimes(f.window)
        where = ["case_id = %s"]
        params: list[Any] = [self.case_id]
        # `_time_` may be zero/epoch for legacy imports; only filter when the
        # column is populated by treating absent values via >= start anyway.
        where.append("(`_time_` = toDateTime(0) OR (`_time_` >= %s AND `_time_` < %s))")
        params.extend([start_dt, end_dt])

        for col, vals in (
            ("pod_name", f.pod_names),
            ("level", f.levels),
            ("clusterId", f.cluster_ids),
            ("hostname", f.hosts),
        ):
            ph, p = _in_clause(vals)
            if ph:
                where.append(f"{col} IN ({ph})")
                params.extend(p)

        sql = (
            "SELECT eventId, hostname, level, pod_name, clusterId, `_time_` "
            "FROM events WHERE "
            + " AND ".join(where)
            + " ORDER BY `_time_` "
            f"LIMIT {int(f.limit)}"
        )
        rows = self._query_rows(sql, params)
        out: list[K8sEvent] = []
        for event_id, host, level, pod, cluster_id, t in rows:
            meta = _safe_json(event_id)
            reason = meta.get("reason")
            message = (
                meta.get("message")
                or (meta.get("note") if isinstance(meta.get("note"), str) else None)
            )
            out.append(
                K8sEvent(
                    ts=t if isinstance(t, datetime) else None,
                    level=level or None,
                    pod=pod or None,
                    cluster_id=cluster_id or None,
                    hostname=host or None,
                    metadata=meta,
                    reason=reason if isinstance(reason, str) else None,
                    message=message,
                )
            )
        return out

    def query_alerts(self, f: AlertFilter) -> list[CloudEvent]:
        start_dt, end_dt = _window_datetimes(f.window)
        where = ["case_id = %s", "(`_time_` = toDateTime(0) OR (`_time_` >= %s AND `_time_` < %s))"]
        params: list[Any] = [self.case_id, start_dt, end_dt]

        for col, vals in (
            ("severity", f.severities),
            ("subtype", f.subtypes),
            ("subject", f.subjects),
        ):
            ph, p = _in_clause(vals)
            if ph:
                where.append(f"{col} IN ({ph})")
                params.extend(p)

        sql = (
            "SELECT id, type, subtype, severity, status, subject, `_time_`, "
            "resource, labels, annotations, data "
            "FROM alerts WHERE "
            + " AND ".join(where)
            + " ORDER BY `_time_` "
            f"LIMIT {int(f.limit)}"
        )
        rows = self._query_rows(sql, params)
        out: list[CloudEvent] = []
        for (
            aid, atype, asub, sev, status, subject, t, resource, labels, annotations_col, data
        ) in rows:
            out.append(
                CloudEvent(
                    id=aid or "",
                    type=atype or "",
                    subtype=asub or None,
                    severity=sev or None,
                    status=status or None,
                    subject=subject or None,
                    ts=t if isinstance(t, datetime) else None,
                    resource=_safe_json(resource),
                    labels=_safe_json(labels),
                    annotations=_safe_json(annotations_col),
                    data=_safe_json(data),
                )
            )
        return out

    def query_topology(self, f: TopologyFilter) -> TopologySubgraph:
        # Base entity selection by exact ids / types / names.
        ent_where = ["case_id = %s"]
        ent_params: list[Any] = [self.case_id]
        for col, vals in (
            ("id", f.entity_ids),
            ("type", f.entity_types),
            ("name", f.entity_names),
        ):
            ph, p = _in_clause(vals)
            if ph:
                ent_where.append(f"{col} IN ({ph})")
                ent_params.extend(p)

        ent_sql = (
            "SELECT id, type, name, first_observed, last_observed, props "
            "FROM topology_entities WHERE "
            + " AND ".join(ent_where)
            + f" LIMIT {int(f.limit)}"
        )
        ent_rows = self._query_rows(ent_sql, ent_params)

        entities: list[dict[str, Any]] = []
        seed_ids: list[str] = []
        for eid, etype, name, fo, lo, props in ent_rows:
            entities.append(
                {
                    "id": eid,
                    "type": etype,
                    "name": name,
                    "first_observed": fo,
                    "last_observed": lo,
                    "props": _safe_json(props),
                }
            )
            if eid:
                seed_ids.append(eid)

        # Neighborhood expansion (hops). BFS over edges in-process to keep the
        # edge query simple and parameter-free beyond case_id.
        edges_sql = "SELECT src, src_type, dst, dst_type, relation FROM topology_edges WHERE case_id = %s"
        if f.relations:
            ph, p = _in_clause(f.relations)
            if ph:
                edges_sql += f" AND relation IN ({ph})"
                edge_rows = self._query_rows(edges_sql, [self.case_id, *p])
            else:
                edge_rows = self._query_rows(edges_sql, [self.case_id])
        else:
            edge_rows = self._query_rows(edges_sql, [self.case_id])

        all_edges = [
            {
                "src": s,
                "src_type": st,
                "dst": d,
                "dst_type": dt,
                "relation": rel,
            }
            for s, st, d, dt, rel in edge_rows
        ]

        adj: dict[str, set[str]] = {}
        for e in all_edges:
            adj.setdefault(e["src"], set()).add(e["dst"])
            adj.setdefault(e["dst"], set()).add(e["src"])

        # BFS up to `hops` from the seed ids (0 = exact only).
        hops = max(0, int(f.hops))
        reachable: set[str] = set(seed_ids)
        if hops > 0 and seed_ids:
            frontier = set(seed_ids)
            for _ in range(hops):
                nxt: set[str] = set()
                for node in frontier:
                    for nbr in adj.get(node, ()):
                        if nbr not in reachable:
                            nxt.add(nbr)
                if not nxt:
                    break
                reachable |= nxt
                frontier = nxt

        # If no seeds were specified (pure type/name filter), keep the direct
        # entity set without neighborhood expansion and return all matching edges.
        keep_ids = reachable if seed_ids else {e["id"] for e in entities}
        entities = [e for e in entities if e["id"] in keep_ids or not seed_ids]
        # When a neighborhood expanded the set beyond the seed filter, fetch the
        # extra entity metadata so the subgraph is complete.
        if seed_ids and hops > 0:
            extra_ids = reachable - {e["id"] for e in entities}
            if extra_ids:
                ph = ", ".join(["%s"] * len(extra_ids))
                extra_rows = self._query_rows(
                    "SELECT id, type, name, first_observed, last_observed, props "
                    f"FROM topology_entities WHERE case_id = %s AND id IN ({ph})",
                    [self.case_id, *extra_ids],
                )
                for eid, etype, name, fo, lo, props in extra_rows:
                    entities.append(
                        {
                            "id": eid,
                            "type": etype,
                            "name": name,
                            "first_observed": fo,
                            "last_observed": lo,
                            "props": _safe_json(props),
                        }
                    )

        keep = keep_ids | {e["id"] for e in entities}
        edges_out = [e for e in all_edges if e["src"] in keep and e["dst"] in keep]
        # Respect the limit on entities.
        if len(entities) > f.limit:
            entities = entities[: int(f.limit)]
        return TopologySubgraph(entities=entities, edges=edges_out)

    # ------------------------------------------------------------------ #
    # Structured-text rendering helpers (LLM context)
    # ------------------------------------------------------------------ #
    def render_metrics(self, series: list[MetricSeries]) -> str:
        if not series:
            return "## Metrics\n(none)\n"
        lines = ["## Metrics"]
        for ms in series:
            stats = ms.summary_stats()
            ident = ms.entity_name or ms.entity_id
            lines.append(
                f"- [{ms.domain}] {ms.metric} | {ident}"
                f" (svc={ms.service or '-'}, type={ms.entity_type or '-'})"
                f" n={int(stats.get('count', 0))}"
            )
            if "min" in stats:
                lines.append(
                    f"    min={stats['min']:.4g} max={stats['max']:.4g} "
                    f"avg={stats['avg']:.4g} last={stats['last']:.4g}"
                )
        return "\n".join(lines) + "\n"

    def render_logs(self, logs: list[LogLine]) -> str:
        if not logs:
            return "## Logs\n(none)\n"
        lines = ["## Logs"]
        for lg in logs:
            ts = lg.ts.isoformat() if lg.ts else "?"
            who = lg.pod or lg.host or "?"
            lines.append(f"[{ts}] {who}: {lg.content}")
        return "\n".join(lines) + "\n"

    def render_traces(self, traces: list[Trace]) -> str:
        if not traces:
            return "## Traces\n(none)\n"
        lines = ["## Traces"]
        for tr in traces:
            slow = tr.slowest_span()
            slow_ms = (slow.duration_ns / 1e6) if (slow and slow.duration_ns) else 0
            lines.append(
                f"- trace {tr.trace_id} spans={len(tr.spans)} "
                f"slowest={slow_ms:.1f}ms ({slow.name if slow else '-'})"
            )
            for sp in tr.spans:
                dur_ms = (sp.duration_ns / 1e6) if sp.duration_ns else 0
                flag = f" [{sp.status_code}]" if sp.status_code else ""
                lines.append(
                    f"    - {sp.service or '-'}::{sp.name} dur={dur_ms:.1f}ms{flag}"
                )
        return "\n".join(lines) + "\n"

    def render_events(self, events: list[K8sEvent]) -> str:
        if not events:
            return "## Events\n(none)\n"
        lines = ["## Events"]
        for ev in events:
            ts = ev.ts.isoformat() if ev.ts else "?"
            who = ev.pod or ev.hostname or "?"
            lvl = ev.level or "?"
            lines.append(f"[{ts}] {lvl} {who}: {ev.reason or ''} {ev.message or ''}".rstrip())
        return "\n".join(lines) + "\n"

    def render_alerts(self, alerts: list[CloudEvent]) -> str:
        if not alerts:
            return "## Alerts\n(none)\n"
        lines = ["## Alerts"]
        for a in alerts:
            ts = a.ts.isoformat() if a.ts else "?"
            lines.append(
                f"- [{ts}] {a.severity or '-'} {a.type or '-'} ({a.subtype or '-'}) "
                f"{a.subject or ''} [{a.status or '-'}]"
            )
            if a.data:
                lines.append(f"    data={json.dumps(a.data, ensure_ascii=False)[:300]}")
        return "\n".join(lines) + "\n"

    def render_topology(self, sub: TopologySubgraph) -> str:
        if not sub.entities and not sub.edges:
            return "## Topology\n(none)\n"
        lines = ["## Topology", "Entities:"]
        for e in sub.entities:
            lines.append(f"- {e['type']}: {e['name']} ({e['id']})")
        lines.append("Edges:")
        for e in sub.edges:
            lines.append(f"- {e['src']} --{e['relation']}--> {e['dst']}")
        return "\n".join(lines) + "\n"


__all__ = ["ClickhouseProvider"]
