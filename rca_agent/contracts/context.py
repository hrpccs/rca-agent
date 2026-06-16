"""Context-manager contract.

Owns the DeepSeek ``reasoning_content`` echo invariant: for any assistant turn
that produced ``tool_calls``, the matching ``reasoning_content`` MUST be present
in the messages sent on every subsequent request, or the API returns HTTP 400.
This invariant is enforced entirely inside :meth:`ContextManager.assemble_turn`
and :meth:`ContextManager.compress`; the agent loop never manipulates
``reasoning_content`` directly.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolMessage(BaseModel):
    tool_call_id: str
    name: str
    content: str  # JSON string of the ToolResult


class TurnRecord(BaseModel):
    """One assistant turn, persisted so reasoning_content can be replayed."""

    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None  # DeepSeek thinking trace
    tool_calls: list[dict[str, Any]] | None = None  # openai-shaped tool_calls


class ContextState(BaseModel):
    case_id: str
    system: str
    messages: list[dict[str, Any]] = Field(
        default_factory=list
    )  # openai chat messages, ready to send
    turns: list[TurnRecord] = Field(default_factory=list)  # raw history incl. reasoning
    token_estimate: int = 0


@runtime_checkable
class ContextManager(Protocol):
    def init(self, case_id: str, system_prompt: str) -> ContextState: ...

    def append_assistant(
        self,
        state: ContextState,
        content: str | None,
        reasoning_content: str | None,
        tool_calls: list[dict[str, Any]] | None,
    ) -> ContextState:
        """Record an assistant turn.

        If ``tool_calls`` is present, the assistant message appended to
        ``state.messages`` MUST carry ``reasoning_content`` (the DeepSeek echo
        invariant). If there are no tool calls, ``reasoning_content`` is stored
        in ``turns`` for display but omitted from ``messages`` (the API ignores
        it between non-tool user turns).
        """
        ...

    def append_tool_result(
        self, state: ContextState, results: list[ToolMessage]
    ) -> ContextState:
        """Append ``{"role": "tool", "tool_call_id", "content"}`` messages."""
        ...

    def assemble_turn(
        self, state: ContextState, new_user: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the messages array to send to the LLM this turn.

        Guarantees: every assistant message in the array that carries
        ``tool_calls`` also carries its ``reasoning_content``. The system prompt
        is always first.
        """
        ...

    def compress(self, state: ContextState, max_tokens: int) -> ContextState:
        """Summarize the oldest turns to fit ``max_tokens``.

        MUST preserve the reasoning_content echo for any retained tool-bearing
        turn (never drop reasoning from a turn whose tool_calls survive).
        """
        ...


__all__ = ["ToolMessage", "TurnRecord", "ContextState", "ContextManager"]
