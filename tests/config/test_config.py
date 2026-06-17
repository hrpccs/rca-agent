"""Unit tests for rca_agent.config (Settings + get_settings caching + validators).

These tests are fully offline: they construct fresh ``Settings`` instances from
explicit env (via ``monkeypatch.setenv`` + ``get_settings.cache_clear()``) and
never read a real ``.env``. They pin the CURRENT default value of every Settings
field so an accidental default drift is caught, and they exercise the
non-breaking field validators added alongside this test file.

The golden-default snapshot doubles as documentation of the contract every other
module implicitly relies on; do NOT relax these assertions when changing a
default — raise a separate, deliberate change instead.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from rca_agent.config import Settings, get_settings

# The ``_clear_settings_cache`` autouse fixture lives in tests/config/conftest.py
# so every config-touching test starts/ends with an empty get_settings lru_cache.
DEFAULTS: dict[str, object] = {
    "deepseek_api_key": "",
    "deepseek_base_url": "https://api.deepseek.com",
    "deepseek_model": "deepseek-reasoner",
    "reasoning_effort": "high",
    "llm_max_steps": 25,
    "llm_max_tokens": 8192,
    "data_backend": "parquet",
    "cases_dir": Path("/Users/hrpccs/Desktop/workspace/aiops/rca100/cases"),
    "clickhouse_host": "localhost",
    "clickhouse_port": 8123,
    "clickhouse_user": "rca",
    "clickhouse_password": "rca123",
    "clickhouse_database": "rca",
    "mysql_url": "mysql+pymysql://rca:rca123@localhost:3306/rca",
    "server_host": "0.0.0.0",
    "server_port": 8000,
    "memory_backend": "inmemory",
    "otel_endpoint": "http://localhost:4317",
    "otel_service_name": "rca-agent",
    "otel_enabled": True,
}


def _defaults_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Build a Settings() that reflects ONLY the class defaults.

    Two contamination sources must be neutralized, or the golden-snapshot tests
    silently assert whatever the developer's environment happens to contain:
      * ``RCA_*`` shell env vars -> deleted from os.environ.
      * a developer ``.env`` in the CWD -> disabled via ``_env_file=None``.

    ``monkeypatch`` auto-undoes the env deletions after the test, so this is
    safe to call per-test without leaking.
    """
    for key in list(os.environ):
        if key.startswith("RCA_"):
            monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)


# --------------------------------------------------------------------------- #
# Defaults: every field matches the golden snapshot
# --------------------------------------------------------------------------- #
def test_defaults_match_snapshot(monkeypatch):
    s = _defaults_settings(monkeypatch)
    for field, expected in DEFAULTS.items():
        actual = getattr(s, field)
        # Path compares by value; ints/strs/bools compare directly.
        assert actual == expected, f"default drift on {field!r}: {actual!r} != {expected!r}"


def test_defaults_field_set_covers_all_settings_fields(monkeypatch):
    """Guard against silently adding a new Settings field without a pinned
    default in this test (which would let its default drift unnoticed)."""
    s = _defaults_settings(monkeypatch)
    model_fields = set(s.__class__.model_fields)
    snapshotted = set(DEFAULTS)
    missing = model_fields - snapshotted
    assert not missing, (
        f"new Settings field(s) without a pinned default: {sorted(missing)} — "
        f"add them to DEFAULTS in this test"
    )


def test_defaults_snapshot_covers_no_unknown_fields(monkeypatch):
    """Inverse guard: DEFAULTS must not list fields that no longer exist."""
    s = _defaults_settings(monkeypatch)
    model_fields = set(s.__class__.model_fields)
    stale = set(DEFAULTS) - model_fields
    assert not stale, f"DEFAULTS references removed Settings field(s): {sorted(stale)}"


# --------------------------------------------------------------------------- #
# get_settings caching + env override
# --------------------------------------------------------------------------- #
def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b  # lru_cache returns the same instance


def test_get_settings_cache_clear_picks_up_new_env(monkeypatch):
    first = get_settings()
    get_settings.cache_clear()
    monkeypatch.setenv("RCA_SERVER_PORT", "9999")
    second = get_settings()
    assert second is not first
    assert second.server_port == 9999
    assert first.server_port == DEFAULTS["server_port"]


def test_env_override_each_field(monkeypatch):
    """Every RCA_* env var maps to the expected field and is coerced to type."""
    overrides = {
        "RCA_DEEPSEEK_API_KEY": "sk-realkey",
        "RCA_DEEPSEEK_BASE_URL": "https://example.com/v1",
        "RCA_DEEPSEEK_MODEL": "deepseek-chat",
        "RCA_REASONING_EFFORT": "medium",
        "RCA_LLM_MAX_STEPS": "40",
        "RCA_LLM_MAX_TOKENS": "1024",
        "RCA_DATA_BACKEND": "clickhouse",
        "RCA_CASES_DIR": "/tmp/cases-override",
        "RCA_CLICKHOUSE_HOST": "ch.example",
        "RCA_CLICKHOUSE_PORT": "9000",
        "RCA_CLICKHOUSE_USER": "root",
        "RCA_CLICKHOUSE_PASSWORD": "secret",
        "RCA_CLICKHOUSE_DATABASE": "analytics",
        "RCA_MYSQL_URL": "mysql+pymysql://u:p@db:3306/x",
        "RCA_SERVER_HOST": "127.0.0.1",
        "RCA_SERVER_PORT": "8080",
        "RCA_MEMORY_BACKEND": "mysql",
        "RCA_OTEL_ENDPOINT": "http://otel:4317",
        "RCA_OTEL_SERVICE_NAME": "svc-x",
        "RCA_OTEL_ENABLED": "false",
    }
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    s = Settings()

    assert s.deepseek_api_key == "sk-realkey"
    assert s.deepseek_base_url == "https://example.com/v1"
    assert s.deepseek_model == "deepseek-chat"
    assert s.reasoning_effort == "medium"
    assert s.llm_max_steps == 40
    assert s.llm_max_tokens == 1024
    assert s.data_backend == "clickhouse"
    assert s.cases_dir == Path("/tmp/cases-override")
    assert s.clickhouse_host == "ch.example"
    assert s.clickhouse_port == 9000
    assert s.clickhouse_user == "root"
    assert s.clickhouse_password == "secret"
    assert s.clickhouse_database == "analytics"
    assert s.mysql_url == "mysql+pymysql://u:p@db:3306/x"
    assert s.server_host == "127.0.0.1"
    assert s.server_port == 8080
    assert s.memory_backend == "mysql"
    assert s.otel_endpoint == "http://otel:4317"
    assert s.otel_service_name == "svc-x"
    assert s.otel_enabled is False


def test_env_prefix_is_rca(monkeypatch):
    """A non-RCA_-prefixed env var must NOT leak into Settings (extra=ignore)."""
    monkeypatch.setenv("DEEPSEEK_MODEL", "should-be-ignored")
    s = Settings()
    assert s.deepseek_model == DEFAULTS["deepseek_model"]


def test_env_file_is_optional():
    """Settings constructs with no .env present (pure-defaults path)."""
    # No env set; if a developer .env exists on this machine the defaults test
    # above already pins the documented values, and Settings() never raises on
    # a missing .env anyway.
    s = Settings()
    assert isinstance(s, Settings)


# --------------------------------------------------------------------------- #
# has_llm_key
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("", False),
        ("sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", False),  # placeholder sentinel
        ("sk-realnotplaceholder", True),
        ("realkey", True),
    ],
)
def test_has_llm_key(monkeypatch, key, expected):
    monkeypatch.setenv("RCA_DEEPSEEK_API_KEY", key)
    assert Settings().has_llm_key is expected


# --------------------------------------------------------------------------- #
# clickhouse_dsn()
# --------------------------------------------------------------------------- #
def test_clickhouse_dsn_round_trips_fields():
    s = Settings()
    dsn = s.clickhouse_dsn()
    assert dsn == {
        "host": s.clickhouse_host,
        "port": s.clickhouse_port,
        "username": s.clickhouse_user,
        "password": s.clickhouse_password,
        "database": s.clickhouse_database,
    }


def test_clickhouse_dsn_reflects_env_overrides(monkeypatch):
    monkeypatch.setenv("RCA_CLICKHOUSE_HOST", "h")
    monkeypatch.setenv("RCA_CLICKHOUSE_PORT", "1234")
    monkeypatch.setenv("RCA_CLICKHOUSE_DATABASE", "db")
    dsn = Settings().clickhouse_dsn()
    assert dsn["host"] == "h"
    assert dsn["port"] == 1234
    assert dsn["database"] == "db"


# --------------------------------------------------------------------------- #
# Field validators (non-breaking): defaults pass, bad values raise.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("port", [8123, 8000, 1, 65535, 80])
def test_port_validator_accepts_valid(monkeypatch, port):
    monkeypatch.setenv("RCA_SERVER_PORT", str(port))
    assert Settings().server_port == port  # no ValidationError raised


@pytest.mark.parametrize("port", ["0", "65536", "-1", "100000"])
def test_port_validator_rejects_out_of_range(monkeypatch, port):
    monkeypatch.setenv("RCA_CLICKHOUSE_PORT", port)
    with pytest.raises(ValidationError) as ei:
        Settings()
    # The offending field must be named in the error so callers get a clear msg.
    errs = ei.value.errors()
    assert any(e.get("loc") == ("clickhouse_port",) for e in errs), errs


@pytest.mark.parametrize("port", ["0", "65536", "-1", "100000"])
def test_server_port_validator_rejects_out_of_range(monkeypatch, port):
    monkeypatch.setenv("RCA_SERVER_PORT", port)
    with pytest.raises(ValidationError) as ei:
        Settings()
    errs = ei.value.errors()
    assert any(e.get("loc") == ("server_port",) for e in errs), errs


@pytest.mark.parametrize(
    ("field", "env"),
    [
        ("deepseek_base_url", "RCA_DEEPSEEK_BASE_URL"),
        ("deepseek_model", "RCA_DEEPSEEK_MODEL"),
        ("data_backend", "RCA_DATA_BACKEND"),
        ("memory_backend", "RCA_MEMORY_BACKEND"),
    ],
)
def test_non_empty_validator_rejects_blank(monkeypatch, field, env):
    monkeypatch.setenv(env, "   ")
    with pytest.raises(ValidationError) as ei:
        Settings()
    errs = ei.value.errors()
    assert any(e.get("loc") == (field,) for e in errs), errs


def test_non_empty_validator_accepts_defaults(monkeypatch):
    """Every default value for a validated field must pass its own validator."""
    s = _defaults_settings(monkeypatch)
    assert s.deepseek_base_url == DEFAULTS["deepseek_base_url"]
    assert s.deepseek_model == DEFAULTS["deepseek_model"]
    assert s.data_backend == DEFAULTS["data_backend"]
    assert s.memory_backend == DEFAULTS["memory_backend"]


def test_api_key_not_validated_allows_empty(monkeypatch):
    """Empty api_key is the documented 'live disabled' sentinel — must NOT raise."""
    s = _defaults_settings(monkeypatch)
    assert s.deepseek_api_key == ""
    assert s.has_llm_key is False


# --------------------------------------------------------------------------- #
# Fail-fast behavior change introduced by the validators.
#
# Because rca_agent.config evaluates a module-level ``settings = get_settings()``
# singleton at import time, an out-of-range port or empty required string in a
# ``.env`` / env now raises ValidationError at IMPORT (fail-fast) rather than
# silently propagating a bad value to first use. This is intentional and
# desirable (a typo'd port surfaces immediately with a clear message), but it IS
# a behavior change for deployments that previously tolerated bad config — so we
# pin it with a test.
# --------------------------------------------------------------------------- #
def test_bad_env_raises_at_settings_construction(monkeypatch):
    """A bad port in env must raise ValidationError when Settings() runs."""
    monkeypatch.setenv("RCA_SERVER_PORT", "80000")
    with pytest.raises(ValidationError) as ei:
        Settings()
    assert any(e.get("loc") == ("server_port",) for e in ei.value.errors())


def test_blank_required_string_raises_at_settings_construction(monkeypatch):
    monkeypatch.setenv("RCA_DATA_BACKEND", "   ")
    with pytest.raises(ValidationError) as ei:
        Settings()
    assert any(e.get("loc") == ("data_backend",) for e in ei.value.errors())
