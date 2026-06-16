"""Agent memory contract.

Memory stores the kind of knowledge a human SRE consults during triage:
app-specific docs, runbooks/SOPs, and business-agnostic domain facts (e.g. "RTT
rise may come from CPU contention or network latency"). The key property is
efficient, on-demand retrieval — NOT stuffing everything into the LLM context.
The interface leaves storage (in-process dict, vector DB, external service) and
retrieval method (keyword, TF-IDF, embedding) pluggable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ._primitives import utcnow


class MemoryItem(BaseModel):
    id: str
    case_id: str = "__global__"  # "__global__" for cross-case / domain knowledge
    content: str
    kind: str = "note"  # runbook | sop | domain_fact | metric_obs | log_obs | hypothesis | evidence
    source_tool: str | None = None
    entities: list[str] = Field(default_factory=list)
    score: float | None = None
    created_at: datetime = Field(default_factory=utcnow)
    meta: dict = Field(default_factory=dict)


class MemoryQuery(BaseModel):
    case_id: str = "__global__"
    text: str | None = None
    kind: str | None = None
    entities: list[str] | None = None
    top_k: int = 8


@runtime_checkable
class MemoryStore(Protocol):
    """Pluggable memory store. Implementations may use keyword, TF-IDF, or
    embedding retrieval."""

    def index(self, items: list[MemoryItem]) -> None: ...
    def retrieve(self, q: MemoryQuery) -> list[MemoryItem]: ...

    def retrieve_for_context(
        self, case_id: str, query: str, top_k: int = 8
    ) -> list[MemoryItem]:
        """Convenience: build a MemoryQuery and call retrieve()."""
        ...

    def clear(self, case_id: str | None = None) -> None: ...


__all__ = ["MemoryItem", "MemoryQuery", "MemoryStore"]
