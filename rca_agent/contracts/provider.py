"""Data-provider abstraction over observable data.

Two backends implement :class:`DataProvider`:
  * parquet  — reads benchmark ``.parquet``/``.json`` files directly (dev/benchmark)
  * clickhouse — queries the same data after import (production)

Both return *identical* result models, so the agent tools never branch on the
backend. This is the single most important integration contract in the system.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ._primitives import Modality, TimeWindow


# --------------------------------------------------------------------------- #
# Filter models (one per modality)
# --------------------------------------------------------------------------- #
class _BaseFilter(BaseModel):
    window: TimeWindow
    limit: int = 5000


class MetricFilter(_BaseFilter):
    entity_ids: list[str] | None = None
    entity_names: list[str] | None = None
    entity_types: list[str] | None = None
    services: list[str] | None = None
    metrics: list[str] | None = None
    domains: list[str] | None = None  # "k8s" | "apm"
    limit: int = 5000


class LogFilter(_BaseFilter):
    pod_names: list[str] | None = None
    namespaces: list[str] | None = None
    containers: list[str] | None = None
    hosts: list[str] | None = None
    contains: str | None = None  # substring match on content
    level_hint: str | None = None  # ERROR/WARN heuristic
    limit: int = 200


class TraceFilter(_BaseFilter):
    trace_ids: list[str] | None = None
    service_names: list[str] | None = None
    span_names: list[str] | None = None
    status_codes: list[str] | None = None  # e.g. ["ERROR"]
    min_duration_ns: int | None = None
    limit: int = 50


class EventFilter(_BaseFilter):
    pod_names: list[str] | None = None
    levels: list[str] | None = None  # ["Warning", "Normal"]
    cluster_ids: list[str] | None = None
    hosts: list[str] | None = None
    limit: int = 200


class AlertFilter(_BaseFilter):
    severities: list[str] | None = None
    subtypes: list[str] | None = None
    subjects: list[str] | None = None
    limit: int = 50


class TopologyFilter(BaseModel):
    """Topology is a graph; filtering is by entity set + neighborhood hops."""

    entity_ids: list[str] | None = None
    entity_types: list[str] | None = None
    entity_names: list[str] | None = None
    hops: int = 1  # neighborhood radius (0 = exact entities only)
    relations: list[str] | None = None
    limit: int = 500


# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #
class MetricSeries(BaseModel):
    entity_id: str = ""
    entity_name: str = ""
    entity_type: str = ""
    service: str | None = None
    domain: str = ""
    metric: str
    metric_set_id: str | None = None
    points: list[tuple[int, float]] = Field(default_factory=list)  # (epoch_micros, value)

    def summary_stats(self) -> dict[str, float]:
        vals = [v for _, v in self.points]
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "avg": sum(vals) / len(vals),
            "last": vals[-1],
        }


class LogLine(BaseModel):
    ts: datetime | None = None
    pod: str | None = None
    namespace: str | None = None
    container: str | None = None
    host: str | None = None
    content: str


class Span(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    kind: str | None = None
    name: str
    service: str | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    duration_ns: int | None = None
    status_code: str | None = None
    status_message: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    trace_id: str
    spans: list[Span] = Field(default_factory=list)

    def slowest_span(self) -> Span | None:
        return max(self.spans, key=lambda s: s.duration_ns or 0, default=None)


class K8sEvent(BaseModel):
    ts: datetime | None = None
    level: str | None = None
    pod: str | None = None
    cluster_id: str | None = None
    hostname: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # parsed eventId JSON
    reason: str | None = None  # k8s event reason, if present in metadata
    message: str | None = None


class CloudEvent(BaseModel):
    """One alert row (CNCF CloudEvents format)."""

    id: str
    type: str
    subtype: str | None = None
    severity: str | None = None
    status: str | None = None
    subject: str | None = None
    ts: datetime | None = None
    resource: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class TopologySubgraph(BaseModel):
    entities: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# The Protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class DataProvider(Protocol):
    """Abstraction over all observable data (parquet OR clickhouse behind it).

    Implementations are constructed per-case (they know the case_id / case_dir /
    clickhouse database scope). Every query takes a filter model and returns the
    matching result models — never raw rows.
    """

    case_id: str

    def query_metrics(self, f: MetricFilter) -> list[MetricSeries]: ...
    def query_logs(self, f: LogFilter) -> list[LogLine]: ...
    def query_traces(self, f: TraceFilter) -> list[Trace]: ...
    def query_events(self, f: EventFilter) -> list[K8sEvent]: ...
    def query_alerts(self, f: AlertFilter) -> list[CloudEvent]: ...
    def query_topology(self, f: TopologyFilter) -> TopologySubgraph: ...
    def modalities(self) -> list[Modality]: ...


__all__ = [
    "MetricFilter",
    "LogFilter",
    "TraceFilter",
    "EventFilter",
    "AlertFilter",
    "TopologyFilter",
    "MetricSeries",
    "LogLine",
    "Span",
    "Trace",
    "K8sEvent",
    "CloudEvent",
    "TopologySubgraph",
    "DataProvider",
]
