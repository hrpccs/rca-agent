"""Local fixtures for the LLM client tests.

Clears proxy environment variables so the OpenAI SDK's underlying httpx client
can be constructed (and so respx can intercept the transport) regardless of any
shell-level SOCKS/HTTP proxy configuration on the developer machine.
"""
from __future__ import annotations

import httpx
import pytest
import respx

_PROXY_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
)


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    # Tell httpx not to read proxies from the environment at all.
    monkeypatch.setenv("NO_PROXY", "*")


@pytest.fixture
def base_url() -> str:
    return "https://api.deepseek.com"


def _resp(
    status: int,
    body: str,
    content_type: str = "application/json",
    base: str = "https://api.deepseek.com",
) -> httpx.Response:
    """Build a respx/httpx Response with a request attached (required for the
    OpenAI SDK to raise APIStatusError rather than APIConnectionError)."""
    return httpx.Response(
        status,
        text=body,
        headers={"content-type": content_type},
        request=httpx.Request("POST", base + "/chat/completions"),
    )


def install_sequence(
    mock: respx.MockRouter,
    responses: list[httpx.Response | Exception],
) -> respx.Route:
    """Register a single ``POST /chat/completions`` route that returns the given
    responses in order (responses[0] on the first call, popped on each call).

    Each entry may be an :class:`httpx.Response` (returned verbatim) or an
    :class:`Exception` instance/type (raised by respx as a transport error),
    so a retry test can interleave failures and successes."""
    route = mock.post("/chat/completions")
    # respx consumes a list side_effect entry-by-entry: Exception instances are
    # raised, Responses are returned. Assign directly (do not wrap in a
    # callable, since callable results must be Responses only).
    route.side_effect = list(responses)
    return route


# Re-exported for tests that build raw SSE bodies without going through respx.
__all__ = ["base_url", "install_sequence", "_resp"]
