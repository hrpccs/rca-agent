"""LLM client contract (DeepSeek thinking mode, streaming).

DeepSeek-specific knobs (``extra_body={"thinking": {"type": "enabled"}}``,
``reasoning_effort``, base_url) live ONLY in the implementation
(:mod:`rca_agent.llm`), never in this contract. The contract exposes a
backend-neutral streaming chat interface that normalizes the raw SSE into typed
:class:`LLMStreamDelta` events.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class DeltaKind(StrEnum):
    TEXT = "text"  # final answer content
    REASONING = "reasoning"  # thinking / reasoning_content
    TOOL_CALL = "tool_call"  # a tool-call fragment
    USAGE = "usage"  # token usage (once, at end)
    DONE = "done"
    ERROR = "error"


class LLMStreamDelta(BaseModel):
    kind: DeltaKind
    text: str | None = None
    reasoning: str | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_args_fragment: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None


class LLMRequest(BaseModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    model: str = "deepseek-reasoner"
    reasoning_effort: str = "high"
    thinking_enabled: bool = True
    stream: bool = True
    max_tokens: int | None = None
    # NOTE: no temperature/top_p when thinking_enabled (DeepSeek constraint).


@runtime_checkable
class LLMClient(Protocol):
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMStreamDelta]:
        """Async generator normalizing the OpenAI/DeepSeek SSE stream.

        MUST:
          * emit REASONING deltas from ``reasoning_content``
          * accumulate tool_calls across chunks, emitting TOOL_CALL deltas
          * emit a single USAGE delta at the end if available
          * emit DONE then stop; emit ERROR (then stop) on non-2xx.
        """
        ...

    async def complete(
        self, req: LLMRequest
    ) -> tuple[str | None, str | None, list[dict[str, Any]] | None, dict[str, Any] | None]:
        """Non-streaming convenience. Returns
        ``(content, reasoning_content, tool_calls, usage)`` by consuming
        :meth:`stream`."""
        ...


__all__ = ["DeltaKind", "LLMStreamDelta", "LLMRequest", "LLMClient"]
