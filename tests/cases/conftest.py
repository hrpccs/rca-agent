"""Shared fixtures for the rca_agent.cases unit tests.

Hosts the ``get_settings`` cache-reset fixture: several ``rca_agent.cases``
helpers fall back to ``get_settings().cases_dir`` when ``cases_dir`` is omitted,
so each test must start/end with an empty lru_cache to avoid inheriting another
test's ``RCA_CASES_DIR`` env override. The root ``tests/conftest.py`` is
intentionally not touched (it is frozen); this local conftest mirrors the
``tests/llm/conftest.py`` precedent.
"""
from __future__ import annotations

import pytest

from rca_agent.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
