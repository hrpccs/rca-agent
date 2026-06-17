"""SSE event schema shared by the server and the frontend.

The server emits a stream of :class:`SSEEvent` over the ``GET /rca/{case_id}/stream``
endpoint; the frontend consumes the identical shape (TS types generated from the
JSON schema). This guarantees the two never drift.
"""
from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Union

from pydantic import BaseModel

from .rca import RcaReport, RcaStep


class SSEEventKind(StrEnum):
    STEP = "step"
    DELTA = "delta"
    REPORT = "report"
    ERROR = "error"
    DONE = "done"
    PING = "ping"


class SSEDelta(BaseModel):
    """Fine-grained streaming token (optional; the agent may emit only STEP events)."""

    kind: str  # text | reasoning | tool_call
    text: str | None = None
    step_id: str | None = None


# `data` is one of the structured payloads; kept as a broad union plus dict for
# error/ping cases.
SSEData = Union[RcaStep, RcaReport, SSEDelta, dict[str, Any]]  # noqa: UP007


class SSEEvent(BaseModel):
    event: SSEEventKind
    case_id: str
    data: SSEData = {}
    seq: int = 0


def sse_format(ev: SSEEvent) -> str:
    """Serialize to the SSE wire format:

    ``event: <kind>\\ndata: <json>\\n\\n``
    """
    payload = ev.model_dump(mode="json")
    data_json = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {ev.event.value}\ndata: {data_json}\n\n"


__all__ = [
    "SSEEventKind",
    "SSEDelta",
    "SSEEvent",
    "sse_format",
]
