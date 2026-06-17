"""DeepSeek thinking-mode streaming LLM client (U3, hardened U7).

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

Resilience (U7)
---------------
The single network round-trip that establishes the SSE connection
(``chat.completions.create(stream=True)``) is wrapped in a bounded retry loop
with jittered exponential backoff. Only TRANSIENT errors are retried:

* :class:`httpx.TimeoutException`
* :class:`httpx.TransportError` (covers :class:`httpx.ConnectError`)
* :class:`openai.APIStatusError` whose ``status_code`` is in
  ``{408, 409, 429, 500, 502, 503, 504}``

All other errors (notably other 4xx) are surfaced immediately as a terminal
``ERROR`` delta. Mid-stream failures (after the SSE connection is established)
are NOT retried — the contract requires ``ERROR`` to be terminal, and partial
output has already been emitted.

The OpenAI SDK's own built-in retry is disabled (``max_retries=0``) so that this
layer is the single owner of retry policy; otherwise both layers would retry the
same transient error, doubling the wall-clock cost and confusing call-count
expectations.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import NOT_GIVEN, APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from rca_agent.config import get_settings
from rca_agent.contracts.llm import (
    DeltaKind,
    LLMClient,
    LLMRequest,
    LLMStreamDelta,
)

__all__ = ["DeepSeekClient", "default_client"]

logger = logging.getLogger(__name__)

# HTTP status codes that are considered transient and therefore retryable.
# 408 Request Timeout, 409 Conflict (transient), 429 Too Many Requests,
# 500/502/503/504 server/gateway errors.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})

# Cap on the total cumulative backoff across all retries (seconds), so a
# misconfigured base / large max_retries cannot stall the agent for minutes.
_BACKOFF_CAP_SECONDS = 8.0


def _retry_tunables() -> tuple[int, float]:
    """Read retry knobs from the environment (with defaults).

    Returns ``(max_retries, base_delay_seconds)``. Read live on every call so
    tests can ``monkeypatch.setenv`` and have the next ``stream()`` honor the
    new value without rebuilding the client.
    """
    raw_retries = os.environ.get("RCA_LLM_MAX_RETRIES", "3")
    raw_base = os.environ.get("RCA_LLM_RETRY_BASE", "0.5")
    try:
        max_retries = max(0, int(raw_retries))
    except (TypeError, ValueError):
        max_retries = 3
    try:
        base_delay = max(0.0, float(raw_base))
    except (TypeError, ValueError):
        base_delay = 0.5
    return max_retries, base_delay


def _is_transient(exc: BaseException) -> bool:
    """Classify whether ``exc`` is a transient error worth retrying.

    Covers the httpx transport/timeout bases the OpenAI SDK wraps
    (:class:`openai.APIConnectionError` — including its
    :class:`openai.APITimeoutError` subclass — is the SDK's representation of
    an :class:`httpx.TransportError` / :class:`httpx.ConnectError` /
    :class:`httpx.TimeoutException`), as well as the SDK's own
    :class:`APIStatusError` mapped from a retryable HTTP status. The underlying
    httpx types are also matched directly as a defense-in-depth in case a
    future SDK version surfaces them unwrapped.
    """
    # The OpenAI SDK wraps every httpx transport/timeout failure in
    # APIConnectionError (APITimeoutError is its timeout subclass).
    if isinstance(exc, APIConnectionError):
        return True
    # httpx-level transport/timeout errors surfaced unwrapped.
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    return False


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
        # Disable the SDK's own retry loop: this client owns retry policy (U7).
        self._client = AsyncOpenAI(
            api_key=api_key or s.deepseek_api_key,
            base_url=base_url or s.deepseek_base_url,
            max_retries=0,
        )

    # ------------------------------------------------------------------ #
    # Retry-wrapped single round-trip
    # ------------------------------------------------------------------ #
    async def _open_stream(self, req: LLMRequest) -> Any:
        """Open the SSE connection with bounded retry on transient errors.

        Retries only the connection-establishing ``create()`` call. On success
        returns the live ``AsyncStream``; the caller then iterates chunks WITHOUT
        retry (mid-stream failures are terminal, per the streaming contract).
        Raises the last error if all attempts are exhausted or a non-retryable
        error is hit.
        """
        max_retries, base_delay = _retry_tunables()
        tools = req.tools if req.tools else NOT_GIVEN
        tool_choice = "auto" if req.tools else NOT_GIVEN
        max_tokens = req.max_tokens if req.max_tokens else NOT_GIVEN
        model = req.model or self.model
        reasoning_effort = req.reasoning_effort or self.reasoning_effort

        last_exc: BaseException | None = None
        cumulative_backoff = 0.0
        for attempt in range(max_retries + 1):
            try:
                return await self._client.chat.completions.create(
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
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                if not _is_transient(exc):
                    # Non-retryable (e.g. 4xx auth/validation): surface now.
                    logger.warning(
                        "deepseek.create.non_retryable",
                        extra={
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "status_code": getattr(exc, "status_code", None),
                            "error": str(exc),
                        },
                    )
                    raise
                if attempt >= max_retries:
                    # Exhausted retries on a transient error.
                    logger.warning(
                        "deepseek.create.retries_exhausted",
                        extra={
                            "attempts": attempt + 1,
                            "error_type": type(exc).__name__,
                            "status_code": getattr(exc, "status_code", None),
                        },
                    )
                    raise
                # Compute jittered exponential backoff: base * 2**attempt,
                # capped so cumulative never exceeds the cap.
                delay = min(
                    base_delay * (2**attempt),
                    _BACKOFF_CAP_SECONDS - cumulative_backoff,
                )
                delay = max(0.0, delay)
                # Full jitter: sleep a uniform random fraction of the delay.
                jitter = random.uniform(0.0, delay) if delay > 0 else 0.0
                logger.info(
                    "deepseek.create.retry",
                    extra={
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "delay_s": round(jitter, 3),
                        "error_type": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                    },
                )
                if jitter > 0:
                    await asyncio.sleep(jitter)
                cumulative_backoff += jitter
        # Unreachable: the loop either returns or raises.
        assert last_exc is not None  # pragma: no cover
        raise last_exc  # pragma: no cover

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #
    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMStreamDelta]:
        """Normalize the raw SSE stream into typed :class:`LLMStreamDelta`.

        Emits REASONING / TEXT / TOOL_CALL deltas as they arrive, a single
        USAGE delta at the end (if present), then a terminal DONE. On an API
        error emits ERROR then stops.

        Only the connection-establishing call is retried (transient errors
        only). Mid-stream errors are terminal.
        """
        try:
            stream = await self._open_stream(req)
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
            logger.warning(
                "deepseek.stream.mid_stream_error",
                extra={"error_type": type(e).__name__, "error": str(e)},
            )
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
