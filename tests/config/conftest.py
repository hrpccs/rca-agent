"""Shared fixtures for the rca_agent.config unit tests.

Hosts the ``get_settings`` cache-reset fixture so every test in this package
starts and ends with an empty lru_cache (env overrides via monkeypatch then
flow through to the next ``get_settings()`` / ``Settings()`` call and don't
leak across tests). ``tests/conftest.py`` (the root) is intentionally not
touched — the cache-clear is specific to config-touching tests.
"""
from __future__ import annotations

import pytest

from rca_agent.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
