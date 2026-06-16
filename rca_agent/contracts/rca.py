"""RCA execution schema shared by agent core, server, eval, and frontend."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ._primitives import utcnow


class StepKind(StrEnum):
    OBSERVE = "observe"
    HYPOTHESIZE = "hypothesize"
    INVESTIGATE = "investigate"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    CONCLUDE = "conclude"
    ERROR = "error"


class RcaStep(BaseModel):
    step_id: str
    case_id: str
    step_kind: StepKind
    thought: str | None = None  # reasoning_content excerpt (optional, for display)
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    tool_result_text: str | None = None  # pre-rendered text fed back to the LLM
    hypothesis: str | None = None
    confidence: float | None = None  # 0..1
    entities: list[str] = Field(default_factory=list)
    ts: datetime = Field(default_factory=utcnow)


class RootCause(BaseModel):
    summary: str  # 1-3 sentence root cause
    entity_refs: list[dict[str, Any]] = Field(default_factory=list)
    fault_type: str | None = None  # e.g. "k8s.pod_crashloop"
    evidence: list[str] = Field(default_factory=list)  # pointers to observations / step_ids
    confidence: float = 0.0
    contributing_factors: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class RcaReport(BaseModel):
    case_id: str
    task_id: str
    alert_title: str
    root_cause: RootCause
    steps: list[RcaStep] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    model: str | None = None
    token_usage: dict[str, Any] | None = None
    status: str = "completed"  # completed | error | truncated


class RcaTrace(BaseModel):
    """Streaming envelope: a sequence of these is the full run."""

    case_id: str
    steps: list[RcaStep] = Field(default_factory=list)
    report: RcaReport | None = None
    final: bool = False


__all__ = ["StepKind", "RcaStep", "RootCause", "RcaReport", "RcaTrace"]
