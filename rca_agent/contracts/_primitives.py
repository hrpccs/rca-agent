"""Shared primitive types for the RCA agent contracts.

These are intentionally dependency-free (only stdlib + pydantic) so that every
module can import them without creating import cycles. Do not import any
non-contract module from here.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Modality(StrEnum):
    """Observable-data modalities available to the RCA agent."""

    METRICS = "metrics"
    LOGS = "logs"
    TRACES = "traces"
    EVENTS = "events"
    ALERTS = "alerts"
    TOPOLOGY = "topology"


class Severity(StrEnum):
    INFO = "INFO"
    WARN = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TimeWindow(BaseModel):
    """A half-open [start, end) time window. Timestamps are tz-aware ISO 8601.

    The dataset stores epoch microseconds natively (``*_us``); providers use
    those when present to avoid repeated parsing.
    """

    start: datetime
    end: datetime
    start_us: int | None = None
    end_us: int | None = None


class EntityRef(BaseModel):
    """A reference to a topology entity. Fields may be None when unknown
    (the benchmark leaves ~19%% of alert entities fully null)."""

    entity_id: str | None = None
    entity_name: str | None = None
    entity_type: str | None = None  # e.g. "apm.operation"
    entity_domain: str | None = None  # e.g. "apm"


def utcnow() -> datetime:
    """tz-aware UTC now. Use as ``Field(default_factory=utcnow)``."""
    from datetime import timezone

    return datetime.now(timezone.utc)


__all__ = ["Modality", "Severity", "TimeWindow", "EntityRef", "utcnow"]
