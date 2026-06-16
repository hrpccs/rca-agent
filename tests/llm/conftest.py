"""Local fixtures for the LLM client tests.

Clears proxy environment variables so the OpenAI SDK's underlying httpx client
can be constructed (and so respx can intercept the transport) regardless of any
shell-level SOCKS/HTTP proxy configuration on the developer machine.
"""
from __future__ import annotations

import os

import pytest

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
