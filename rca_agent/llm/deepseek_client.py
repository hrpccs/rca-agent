"""DeepSeek thinking-mode streaming LLM client (U3).

Implements :class:`rca_agent.contracts.llm.LLMClient` on top of the OpenAI SDK
pointed at DeepSeek's OpenAI-compatible endpoint with thinking mode enabled.

DeepSeek-specific knobs that live here (never in the contract):

* ``base_url`` defaults to ``https://api.deepseek.com``
* model defaults to ``deepseek-reasoner``
* ``reasoning_effort`` (default ``"high"``) is forwarded to the API
* ``extra_body={"thinking": {"type": "enabled"}}`` turns on thinking mode, which
  emits ``delta.reasoning_content`` (the thinking trace) alongside the normal
  content / tool-call deltas. Thinking mode does NOT accept ``temperature`` /
  ``top_p`` and this client never sends them.

The multi-turn ``reasoning_content`` echo-back rule (an assistant message that
carries ``tool_calls`` must echo its ``reasoning_content`` in later turns) is
enforced by the context layer; this client only faithfully relays whatever it
receives — it never mutates ``req.messages``.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from rca_agent.config import get_settings
from rca_agent.contracts.llm import (
    DeltaKind,
    LLMClient,
    LLMRequest,
    LLMStreamDelta,
)

__all__ = ["DeepSeekClient", "default_client"]


class DeepSeekClient(LLMClient):
    """Streaming DeepSeek chat client with thinking mode enabled."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        s = get_settings()
        self.model = model or s.deepseek_model
        self.reasoning_effort = reasoning_effort or s.reasoning_effort
        self._client = AsyncOpenAI(
            api_key=api_key or s.deepseek_api_key,
            base_url=base_url or s.deepseek_base_url,
        )

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMStreamDelta]:
        """Normalize the raw SSE stream into typed :class:`LLMStreamDelta`.

        Emits REASONING / TEXT / TOOL_CALL deltas as they arrive, a single
        USAGE delta at the end (if present), then a terminal DONE. On an API
        error emits ERROR then stops.
        """
        tools = req.tools if req.tools else NOT_GIVEN
        tool_choice = "auto" if req.tools else NOT_GIVEN
        max_tokens = req.max_tokens if req.max_tokens else NOT_GIVEN
        model = req.model or self.model
        reasoning_effort = req.reasoning_effort or self.reasoning_effort

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=req.messages,
                stream=True,
                stream_options={"include_usage": True},
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                extra_body={"thinking": {"type": "enabled"}},
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001 - surface any API error to caller
            yield LLMStreamDelta(kind=DeltaKind.ERROR, error=str(e))
            return

        try:
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                delta: ChoiceDelta = choices[0].delta if choices else None

                # Emit content deltas first so that, if the usage-bearing final
                # chunk also carries a content/tool-call delta, USAGE strictly
                # follows the content (contract: "USAGE at the end").
                if delta is not None:
                    # Thinking trace (DeepSeek extension; not a typed OpenAI field).
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        yield LLMStreamDelta(kind=DeltaKind.REASONING, reasoning=reasoning)

                    if delta.content:
                        yield LLMStreamDelta(kind=DeltaKind.TEXT, text=delta.content)

                    # Tool-call fragments. DeepSeek (like OpenAI) streams a single
                    # tool call across many chunks keyed by ``index``. We emit one
                    # typed delta per fragment; the caller (complete()) reassembles.
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            name = tc.function.name if tc.function else None
                            args_frag = tc.function.arguments if tc.function else None
                            yield LLMStreamDelta(
                                kind=DeltaKind.TOOL_CALL,
                                tool_call_index=idx,
                                tool_call_id=tc.id,
                                tool_call_name=name,
                                tool_call_args_fragment=args_frag,
                            )

                # Usage (if any) rides on the final chunk. Emitted AFTER any
                # content/tool delta on the same chunk so it stays terminal.
                if getattr(chunk, "usage", None) is not None:
                    yield LLMStreamDelta(
                        kind=DeltaKind.USAGE,
                        usage=chunk.usage.model_dump(),
                    )

            yield LLMStreamDelta(kind=DeltaKind.DONE)
        except Exception as e:  # noqa: BLE001 - mid-stream API error
            yield LLMStreamDelta(kind=DeltaKind.ERROR, error=str(e))

    # ------------------------------------------------------------------ #
    # Non-streaming convenience
    # ------------------------------------------------------------------ #
    async def complete(
        self, req: LLMRequest
    ) -> tuple[
        str | None,
        str | None,
        list[dict[str, Any]] | None,
        dict[str, Any] | None,
    ]:
        """Consume :meth:`stream` and return the assembled result.

        Returns ``(content, reasoning_content, tool_calls, usage)`` where
        ``tool_calls`` is OpenAI-shaped::

            [{"id": ..., "type": "function",
              "function": {"name": ..., "arguments": ...}}]
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # index -> {"id", "name", "arguments"}
        tool_acc: dict[int, dict[str, str | None]] = {}
        usage: dict[str, Any] | None = None

        async for delta in self.stream(req):
            if delta.kind is DeltaKind.TEXT and delta.text:
                text_parts.append(delta.text)
            elif delta.kind is DeltaKind.REASONING and delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            elif delta.kind is DeltaKind.TOOL_CALL and delta.tool_call_index is not None:
                slot = tool_acc.setdefault(
                    delta.tool_call_index,
                    {"id": None, "name": None, "arguments": ""},
                )
                if delta.tool_call_id:
                    slot["id"] = delta.tool_call_id
                if delta.tool_call_name:
                    slot["name"] = delta.tool_call_name
                if delta.tool_call_args_fragment:
                    slot["arguments"] += delta.tool_call_args_fragment
            elif delta.kind is DeltaKind.USAGE and delta.usage:
                usage = delta.usage
            elif delta.kind is DeltaKind.ERROR:
                raise RuntimeError(delta.error or "DeepSeek API error")

        content = "".join(text_parts) if text_parts else None
        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        tool_calls: list[dict[str, Any]] | None = None
        if tool_acc:
            tool_calls = [
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {
                        "name": slot["name"],
                        "arguments": slot["arguments"] or "",
                    },
                }
                for _, slot in sorted(tool_acc.items())
            ]

        return content, reasoning, tool_calls, usage


def default_client() -> DeepSeekClient:
    """Build a :class:`DeepSeekClient` from application settings."""
    return DeepSeekClient()
