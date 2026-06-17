# Hardening & Operational Posture

Production-readiness posture for the rca-agent deployment. For bring-up see
`docs/DEPLOY.md`; for infra credentials see `docs/INFRA_ACCESS.md`.

## Container hardening

- **Server** (`deploy/Dockerfile`) runs as a non-root `app` system user and
  binds port `8000` (>1024), so root is not required for networking.
- **Frontend** (`deploy/frontend.Dockerfile`) uses `nginxinc/nginx-unprivileged`
  (non-root UID 101) and listens on `8080`; `docker-compose.prod.yml` maps the
  host port `8080 → container 8080`.
- Both app services declare **resource limits** (`deploy.resources.limits`:
  memory + cpus) and **json-file log rotation** (`max-size: 10m`, `max-file: 3`)
  so a runaway container or a verbose log stream cannot exhaust the host.

## Edge security headers (nginx)

`deploy/frontend.nginx.conf` emits these on every response, including errors:

- `Content-Security-Policy` — `default-src 'self'`; SPA + API/SSE calls are
  same-origin; `frame-ancestors 'none'` (clickjacking); `base-uri 'self'`.
- `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy`.

## Application resilience (configurable via env)

- **LLM retries**: transient DeepSeek errors (timeouts, 429, 5xx) are retried
  with capped exponential backoff — see `rca_agent/llm/deepseek_client.py`.
- **Bounded caches**: parquet table cache (`RCA_PARQUET_CACHE_MAX`) and memory
  store per-bucket cap (`RCA_MEMORY_MAX_PER_BUCKET`, default 0 = unbounded).
- **ClickHouse query timeout**: `RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC` caps slow
  production queries.
- **SSE**: the frontend closes a stalled event stream after an inactivity
  timeout; the server always terminates a stream with a terminal event.

## Secrets

- `RCA_DEEPSEEK_API_KEY` is the only external secret. Load it via `.env`
  (gitignored; never baked into an image — see `.dockerignore`). Rotate by
  editing `.env` and restarting `rca-server`.
- Benchmark infra credentials (ClickHouse/MySQL `rca`/`rca123`) are dev-only
  defaults; override via environment in production.

## Observability

OTel traces → Tempo, metrics/logs → ClickHouse, dashboards in Grafana (`infra/`).
The `@pytest.mark.live` suite gates real-API/DB tests behind the presence of
`RCA_DEEPSEEK_API_KEY` and running infra, so CI stays offline and unbilled.

## Live validation (passed)

Both the parquet and ClickHouse backends return the correct t001 root cause
(payment `Charge` `app.loyalty.level=gold` defect, `fault_type=app.exception`).
The prod stack (`--profile app`) serves `/health`, `/cases`, and the frontend.
