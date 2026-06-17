"""Application settings (env-driven via pydantic-settings).

All knobs are namespaced ``RCA_*`` (see .env.example). Importing this module
never requires a key — live features check :attr:`Settings.has_llm_key`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RCA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM (DeepSeek thinking mode) ----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-reasoner"
    reasoning_effort: str = "high"
    llm_max_steps: int = 25
    llm_max_tokens: int = 8192

    # ---- Data backend ----
    data_backend: str = "parquet"  # parquet | clickhouse
    cases_dir: Path = Path("/Users/hrpccs/Desktop/workspace/aiops/rca100/cases")

    # ---- ClickHouse ----
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "rca"
    clickhouse_password: str = "rca123"
    clickhouse_database: str = "rca"

    # ---- MySQL (app persistence) ----
    mysql_url: str = "mysql+pymysql://rca:rca123@localhost:3306/rca"

    # ---- Server ----
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # ---- Memory ----
    memory_backend: str = "inmemory"

    # ---- OpenTelemetry ----
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "rca-agent"
    otel_enabled: bool = True

    # ------------------------------------------------------------------ #
    # Field validators guarding env overrides.
    #
    # Every default value satisfies its own validator, so these never reject the
    # documented defaults. They are intentionally narrow: only ports get a range
    # check and only the enum-like routing knobs (data/memory backend, model,
    # base url) get a non-empty check. ``deepseek_api_key`` is deliberately NOT
    # validated — empty is the "live features disabled" sentinel.
    #
    # Behavior note: ``rca_agent.config`` builds a module-level ``settings``
    # singleton at import time, so a bad port / blank required string in ``.env``
    # or the shell environment now raises ``ValidationError`` at *import* (fail
    # fast) instead of silently propagating a bad value to first use. This is
    # intentional — a typo'd port surfaces immediately with a clear message —
    # but it is a behavior change for deployments that previously tolerated bad
    # config values silently.
    # ------------------------------------------------------------------ #
    @field_validator("clickhouse_port", "server_port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        # 1..65535 per IANA; defaults 8123 / 8000 are well within range.
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be in 1..65535, got {v}")
        return v

    @field_validator("deepseek_base_url", "deepseek_model", "data_backend", "memory_backend")
    @classmethod
    def _non_empty_str(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @property
    def has_llm_key(self) -> bool:
        return bool(self.deepseek_api_key and not self.deepseek_api_key.startswith("sk-x"))

    def clickhouse_dsn(self) -> dict:
        return {
            "host": self.clickhouse_host,
            "port": self.clickhouse_port,
            "username": self.clickhouse_user,
            "password": self.clickhouse_password,
            "database": self.clickhouse_database,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Backwards-friendly alias used across modules.
settings = get_settings()


__all__ = ["Settings", "get_settings", "settings"]
