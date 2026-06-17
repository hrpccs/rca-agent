"""Mocked unit tests for :class:`rca_agent.llm.deepseek_client.DeepSeekClient`.

Uses ``respx`` to intercept the OpenAI SDK's httpx transport and replay canned
SSE streams. A ``live``-marked test at the bottom makes one real call.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from rca_agent.contracts.llm import DeltaKind, LLMRequest
from rca_agent.llm.deepseek_client import DeepSeekClient, default_client
from tests.llm.conftest import _resp, install_sequence


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sse(events: list[dict | str]) -> str:
    """Build a DeepSeek/OpenAI SSE body from a list of payload dicts.

    A bare ``str`` (e.g. ``"[DONE]"``) is emitted verbatim.
    """
    out: list[str] = []
    for ev in events:
        body = ev if isinstance(ev, str) else json.dumps(ev)
        out.append(f"data: {body}\n\n")
    return "".join(out)


def _chunk(
    *,
    reasoning: str | None = None,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    role: str | None = None,
    usage: dict | None = None,
) -> dict[str, Any]:
    """One chat.completion.chunk payload."""
    delta: dict[str, Any] = {}
    if role:
        delta["role"] = role
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    chunk: dict[str, Any] = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-reasoner",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
        if (delta or usage is None)
        else [],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _install_route(
    mock: Any,
    body: str,
    status: int = 200,
    content_type: str = "text/event-stream",
) -> Any:
    return mock.post("/chat/completions").respond(
        status, text=body, headers={"content-type": content_type}
    )


def _usage(**kw: int) -> dict[str, Any]:
    """The SDK's CompletionUsage.model_dump() always includes the *_details keys."""
    base = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    base.update(kw)
    base.setdefault("completion_tokens_details", None)
    base.setdefault("prompt_tokens_details", None)
    return base


# --------------------------------------------------------------------------- #
# stream()
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_stream_text_and_reasoning(base_url: str) -> None:
    body = _sse(
        [
            _chunk(role="assistant"),
            _chunk(reasoning="thinking hard"),
            _chunk(reasoning=" more"),
            _chunk(content="Hello"),
            _chunk(content=" world"),
            _chunk(
                usage=_usage(
                    prompt_tokens=4,
                    completion_tokens=3,
                    total_tokens=7,
                )
            ),
            "[DONE]",
        ]
    )
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        _install_route(mock, body)
        deltas = [d async for d in client.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}]))]

    kinds = [d.kind for d in deltas]
    assert DeltaKind.DONE in kinds
    reasoning = "".join(d.reasoning for d in deltas if d.kind is DeltaKind.REASONING and d.reasoning)
    text = "".join(d.text for d in deltas if d.kind is DeltaKind.TEXT and d.text)
    assert reasoning == "thinking hard more"
    assert text == "Hello world"
    usages = [d for d in deltas if d.kind is DeltaKind.USAGE]
    assert len(usages) == 1
    assert usages[0].usage == _usage(
        prompt_tokens=4, completion_tokens=3, total_tokens=7
    )
    # DONE is terminal.
    assert kinds[-1] is DeltaKind.DONE


@pytest.mark.asyncio
async def test_stream_tool_call_fragments(base_url: str) -> None:
    body = _sse(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_42",
                        "type": "function",
                        "function": {"name": "add", "arguments": ""},
                    }
                ]
            ),
            _chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"a":1'}}]),
            _chunk(tool_calls=[{"index": 0, "function": {"arguments": ',"b":2}'}}]),
            "[DONE]",
        ]
    )
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        _install_route(mock, body)
        deltas = [d async for d in client.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}]))]

    tc = [d for d in deltas if d.kind is DeltaKind.TOOL_CALL]
    assert len(tc) == 3
    assert tc[0].tool_call_id == "call_42"
    assert tc[0].tool_call_name == "add"
    assert tc[0].tool_call_index == 0
    args = "".join(d.tool_call_args_fragment for d in tc if d.tool_call_args_fragment)
    assert json.loads(args) == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_stream_emits_error_on_api_failure(base_url: str) -> None:
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        _install_route(
            mock,
            '{"error":{"message":"bad key","type":"invalid_request_error"}}',
            status=400,
            content_type="application/json",
        )
        deltas = [d async for d in client.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}]))]

    errs = [d for d in deltas if d.kind is DeltaKind.ERROR]
    assert len(errs) == 1
    assert "bad key" in errs[0].error
    # ERROR is terminal: no DONE after it.
    assert DeltaKind.DONE not in [d.kind for d in deltas]


@pytest.mark.asyncio
async def test_stream_done_even_when_no_usage(base_url: str) -> None:
    body = _sse([_chunk(content="hi"), "[DONE]"])
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        _install_route(mock, body)
        deltas = [d async for d in client.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}]))]
    assert deltas[-1].kind is DeltaKind.DONE
    assert not any(d.kind is DeltaKind.USAGE for d in deltas)


# --------------------------------------------------------------------------- #
# complete()
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_complete_aggregates_text_reasoning_and_usage(base_url: str) -> None:
    body = _sse(
        [
            _chunk(reasoning="plan"),
            _chunk(content="42"),
            _chunk(usage=_usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
            "[DONE]",
        ]
    )
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        _install_route(mock, body)
        content, reasoning, tool_calls, usage = await client.complete(
            LLMRequest(messages=[{"role": "user", "content": "1+1"}])
        )
    assert content == "42"
    assert reasoning == "plan"
    assert tool_calls is None
    assert usage == _usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)


@pytest.mark.asyncio
async def test_complete_assembles_tool_calls_openai_shape(base_url: str) -> None:
    body = _sse(
        [
            _chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": ""},
                    }
                ]
            ),
            _chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"a":1'}}]),
            _chunk(tool_calls=[{"index": 0, "function": {"arguments": ',"b":2}'}}]),
            _chunk(
                tool_calls=[
                    {
                        "index": 1,
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "mul", "arguments": '{"x":3}'},
                    }
                ]
            ),
            "[DONE]",
        ]
    )
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        _install_route(mock, body)
        content, reasoning, tool_calls, usage = await client.complete(
            LLMRequest(messages=[{"role": "user", "content": "hi"}])
        )
    assert content is None
    assert reasoning is None
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "add"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"a": 1, "b": 2}
    assert tool_calls[1]["function"]["name"] == "mul"
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"x": 3}


@pytest.mark.asyncio
async def test_complete_raises_on_api_error(base_url: str) -> None:
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        _install_route(
            mock,
            '{"error":{"message":"rate limited","type":"rate_limit_error"}}',
            status=429,
            content_type="application/json",
        )
        with pytest.raises(RuntimeError, match="rate limited"):
            await client.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))


# --------------------------------------------------------------------------- #
# Retry (U7)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_complete_retries_transient_then_succeeds(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient 429 on attempt 0 must be retried; the retried request must
    still succeed and echo reasoning_content. Exactly 2 HTTP calls."""
    monkeypatch.setenv("RCA_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("RCA_LLM_RETRY_BASE", "0.001")  # near-zero backoff
    ok_body = _sse([_chunk(reasoning="plan"), _chunk(content="42"), "[DONE]"])
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/", assert_all_called=False) as mock:
        route = install_sequence(
            mock,
            [
                _resp(
                    429,
                    '{"error":{"message":"rate limited","type":"rate_limit_error"}}',
                ),
                _resp(200, ok_body, content_type="text/event-stream"),
            ],
        )
        content, reasoning, _tc, _u = await client.complete(
            LLMRequest(messages=[{"role": "user", "content": "hi"}])
        )
    assert route.call_count == 2
    assert content == "42"
    # reasoning_content is still echoed on the retried (successful) request.
    assert reasoning == "plan"


@pytest.mark.asyncio
async def test_complete_retries_timeout_then_succeeds(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An httpx connect timeout is transient: retried once, then succeeds."""
    monkeypatch.setenv("RCA_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("RCA_LLM_RETRY_BASE", "0.001")
    ok_body = _sse([_chunk(content="ok"), "[DONE]"])
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/", assert_all_called=False) as mock:
        route = install_sequence(
            mock,
            [
                httpx.ConnectTimeout("simulated timeout"),
                _resp(200, ok_body, content_type="text/event-stream", base=base_url),
            ],
        )
        content, _reasoning, _tc, _u = await client.complete(
            LLMRequest(messages=[{"role": "user", "content": "hi"}])
        )
    assert route.call_count == 2
    assert content == "ok"


@pytest.mark.asyncio
async def test_complete_exhausts_retries_on_persistent_503(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent 503 must be retried up to max_retries then raised. The
    terminal ERROR surfaces through complete() as RuntimeError."""
    monkeypatch.setenv("RCA_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("RCA_LLM_RETRY_BASE", "0.001")
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/", assert_all_called=False) as mock:
        route = install_sequence(
            mock,
            [
                _resp(503, '{"error":{"message":"down","type":"server_error"}}'),
                _resp(503, '{"error":{"message":"down","type":"server_error"}}'),
                _resp(503, '{"error":{"message":"down","type":"server_error"}}'),
            ],
        )
        with pytest.raises(RuntimeError, match="down"):
            await client.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    # 1 initial + 2 retries = 3 attempts.
    assert route.call_count == 3


@pytest.mark.asyncio
async def test_complete_does_not_retry_non_retryable_400(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 400 (not in the retryable set) must NOT be retried: exactly 1 call."""
    monkeypatch.setenv("RCA_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("RCA_LLM_RETRY_BASE", "0.001")
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/", assert_all_called=False) as mock:
        route = install_sequence(
            mock,
            [_resp(400, '{"error":{"message":"bad request","type":"invalid_request_error"}}')],
        )
        with pytest.raises(RuntimeError, match="bad request"):
            await client.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_complete_no_error_single_call_no_sleep(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a clean 200 the client makes exactly 1 call and never sleeps."""
    monkeypatch.setenv("RCA_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("RCA_LLM_RETRY_BASE", "0.001")
    slept: list[float] = []
    # Monkeypatch the module's asyncio.sleep to detect any backoff.
    import rca_agent.llm.deepseek_client as mod

    real_sleep = mod.asyncio.sleep

    async def fake_sleep(d: float) -> None:
        slept.append(d)
        await real_sleep(0)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    ok_body = _sse([_chunk(content="hi"), "[DONE]"])
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/", assert_all_called=False) as mock:
        route = install_sequence(mock, [_resp(200, ok_body, content_type="text/event-stream")])
        content, _r, _tc, _u = await client.complete(
            LLMRequest(messages=[{"role": "user", "content": "hi"}])
        )
    assert route.call_count == 1
    assert content == "hi"
    assert slept == [], f"expected no backoff sleep on clean call, got {slept}"


@pytest.mark.asyncio
async def test_stream_emits_error_on_non_retryable_401(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stream() surfaces a non-retryable APIStatusError (401, not in the
    retryable set) as a single terminal ERROR delta (no DONE), honoring the
    streaming contract. Exactly 1 call regardless of the retry budget."""
    monkeypatch.setenv("RCA_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("RCA_LLM_RETRY_BASE", "0.001")
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/", assert_all_called=False) as mock:
        route = install_sequence(
            mock,
            [_resp(401, '{"error":{"message":"bad key","type":"invalid_request_error"}}')],
        )
        deltas = [d async for d in client.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}]))]
    assert route.call_count == 1
    errs = [d for d in deltas if d.kind is DeltaKind.ERROR]
    assert len(errs) == 1
    assert "bad key" in errs[0].error
    assert DeltaKind.DONE not in [d.kind for d in deltas]


def test_retry_tunables_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without env overrides the defaults are max_retries=3, base=0.5s."""
    from rca_agent.llm.deepseek_client import _retry_tunables

    monkeypatch.delenv("RCA_LLM_MAX_RETRIES", raising=False)
    monkeypatch.delenv("RCA_LLM_RETRY_BASE", raising=False)
    assert _retry_tunables() == (3, 0.5)


def test_retry_tunables_invalid_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garbage env values fall back to defaults rather than crashing."""
    from rca_agent.llm.deepseek_client import _retry_tunables

    monkeypatch.setenv("RCA_LLM_MAX_RETRIES", "not-a-number")
    monkeypatch.setenv("RCA_LLM_RETRY_BASE", "oops")
    assert _retry_tunables() == (3, 0.5)


# --------------------------------------------------------------------------- #
# Constructor / helpers
# --------------------------------------------------------------------------- #
def test_constructor_overrides_win(base_url: str) -> None:
    c = DeepSeekClient(
        api_key="sk-override",
        base_url=base_url,
        model="custom-model",
        reasoning_effort="low",
    )
    assert c.model == "custom-model"
    assert c.reasoning_effort == "low"


def test_default_client_uses_settings() -> None:
    c = default_client()
    assert c.model  # populated from settings
    assert c.reasoning_effort


@pytest.mark.asyncio
async def test_complete_relays_request_params(base_url: str) -> None:
    """The client must forward model / reasoning_effort / max_tokens / tools."""
    body = _sse([_chunk(content="ok"), "[DONE]"])
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
    async with respx.mock(base_url=base_url + "/") as mock:
        route = _install_route(mock, body)
        await client.complete(
            LLMRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="my-model",
                reasoning_effort="medium",
                max_tokens=123,
                tools=tools,
            )
        )
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "my-model"
    assert sent["reasoning_effort"] == "medium"
    assert sent["max_tokens"] == 123
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"
    assert sent["thinking"] == {"type": "enabled"}
    assert sent["stream"] is True
    # Streaming usage must be requested so the final chunk carries it.
    assert sent["stream_options"] == {"include_usage": True}
    # Thinking mode must not send temperature/top_p.
    assert "temperature" not in sent
    assert "top_p" not in sent


@pytest.mark.asyncio
async def test_stream_usage_follows_content_on_final_chunk(base_url: str) -> None:
    """If the final chunk carries both a content delta and usage, USAGE must
    be emitted AFTER the content delta (contract: USAGE is terminal)."""
    # One chunk carrying both content and usage.
    combined = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek-reasoner",
        "choices": [{"index": 0, "delta": {"content": "tail"}, "finish_reason": "stop"}],
        "usage": _usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    }
    body = _sse([combined, "[DONE]"])
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        _install_route(mock, body)
        deltas = [d async for d in client.stream(LLMRequest(messages=[{"role": "user", "content": "hi"}]))]

    # Find positions of the trailing TEXT and USAGE.
    text_pos = next(i for i, d in enumerate(deltas) if d.kind is DeltaKind.TEXT)
    usage_pos = next(i for i, d in enumerate(deltas) if d.kind is DeltaKind.USAGE)
    done_pos = next(i for i, d in enumerate(deltas) if d.kind is DeltaKind.DONE)
    assert text_pos < usage_pos < done_pos


@pytest.mark.asyncio
async def test_complete_omits_tools_when_none(base_url: str) -> None:
    body = _sse([_chunk(content="ok"), "[DONE]"])
    client = DeepSeekClient(api_key="sk-test", base_url=base_url)
    async with respx.mock(base_url=base_url + "/") as mock:
        route = _install_route(mock, body)
        await client.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    sent = json.loads(route.calls[0].request.content)
    assert "tools" not in sent
    assert "tool_choice" not in sent


# --------------------------------------------------------------------------- #
# Live (real DeepSeek call)
# --------------------------------------------------------------------------- #
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_thinking_mode_produces_reasoning() -> None:
    client = default_client()
    content, reasoning, _tool_calls, usage = await client.complete(
        LLMRequest(messages=[{"role": "user", "content": "What is 7*6? Think briefly."}], max_tokens=2048)
    )
    assert content
    assert reasoning, "expected non-empty reasoning_content (thinking mode on)"
    assert usage is not None
    assert "42" in (content or "")
