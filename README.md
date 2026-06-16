# RCA Agent

A production-grade, LLM-core **Root Cause Analysis (RCA)** agent. It investigates
observability data (alerts / events / metrics / logs / traces / topology) using a
DeepSeek **thinking-mode** model with **streaming** + tool calls, mimicking a human
SRE. Validated against the **rca100** blind benchmark (103 cases).

## Architecture

```
                 ┌──────────────────────────────────────────────┐
   RCA request → │ rca-server (FastAPI, SSE)  →  mysql (state)   │
                 │   └ rca-agent core (ReAct loop)                │
                 │       ├ llm client (DeepSeek thinking/stream)  │
                 │       ├ context manager (msg + compress)       │
                 │       ├ tools ──┐                              │
                 │       └ memory ┤                              │
                 └─────────────────┼──────────────────────────────┘
                                   │
        ┌──────────────────────────┴────────────────────────────┐
        │  data-provider (parquet | clickhouse)                  │
        │    metrics / logs / traces / events / alerts / topology│
        └──────────────────────────┬────────────────────────────┘
                                   │
   OpenTelemetry ─ otel-collector ─┬─ Tempo (traces) ─┐
                                   └─ ClickHouse (metrics/logs) ─┤── Grafana (dashboards)
   frontend (React+Vite)  ←─ SSE ── rca-server ─────────────────┘
```

| Component | Module |
|---|---|
| LLM RCA core | `rca_agent/agent` + `rca_agent/context` + `rca_agent/tools` + `rca_agent/memory` |
| Data provider | `rca_agent/providers` (parquet, clickhouse, loader) |
| Memory | `rca_agent/memory` + `memory/seed/` |
| Context manager | `rca_agent/context` |
| RCA server | `rca_agent/server` + `rca_agent/store` (MySQL) |
| Monitoring & eval | `rca_agent/observability` (OTel) + `rca_agent/eval` + Grafana |
| Frontend | `frontend/` |
| Infra | `docker-compose.yml` (ClickHouse, MySQL, OTel-collector, Tempo, Grafana) |

The frozen integration contracts live in [`rca_agent/contracts/`](rca_agent/contracts) — every
module programs against these Pydantic models + Protocols.

## Quick start

```bash
# 1. Infra (ClickHouse, MySQL, OTel-collector, Tempo, Grafana)
make up

# 2. Python env + .env
make dev                      # uv sync; copies .env.example → .env, set RCA_DEEPSEEK_API_KEY

# 3. Run the agent on a benchmark case (parquet backend, live DeepSeek)
make run CASE=t001

# 4. Serve + watch the trace stream
make serve
curl -N http://localhost:8000/rca/t001/stream

# 5. Frontend
make frontend                 # http://localhost:5173
```

Access details for every service (host/port/user/password) are in
[`docs/INFRA_ACCESS.md`](docs/INFRA_ACCESS.md). Architecture/deploy/API docs in
[`docs/`](docs).

## Tests

```bash
make test            # unit tests (excludes live)
make test-contracts  # contract conformance gate
make test-live       # incl. real DeepSeek + DB tests (needs key + infra)
```
