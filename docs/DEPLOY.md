# Deployment

How to run the RCA agent in production. The stack is split into two profiles:

- **Infra** — the external dependencies, defined in the existing
  [`docker-compose.yml`](../docker-compose.yml): ClickHouse, MySQL,
  otel-collector, Tempo, Grafana. These are generic services brought up the same
  way in every environment.
- **App** — the rca-agent itself (FastAPI server + agent core + frontend),
  intended for a future `docker-compose.prod.yml` overlay. The app talks to the
  infra services over the compose network.

This doc covers both.

---

## 1. Infra stack (existing `docker-compose.yml`)

Bring up all dependencies:

```bash
docker compose up -d clickhouse mysql otel-collector tempo grafana
docker compose ps
```

Service map (host ports + purpose; full creds in [`INFRA_ACCESS.md`](INFRA_ACCESS.md)):

| Service | Host port | Container | Purpose |
|---|---|---|---|
| ClickHouse HTTP | `8123` | `rca-clickhouse` | data-provider backend + OTel metrics/logs |
| ClickHouse native | `9000` | `rca-clickhouse` | clickhouse-client / loader |
| MySQL | `3306` | `rca-mysql` | rca-server persistence (cases/runs/reports) |
| OTel OTLP gRPC | `4317` | `rca-otelcol` | app exports traces/metrics/logs |
| OTel OTLP HTTP | `4318` | `rca-otelcol` | app exports (HTTP) |
| OTel metrics | `8888` | `rca-otelcol` | collector default telemetry endpoint (Prometheus-format, scrape target) |
| Tempo UI | `3200` | `rca-tempo` | trace backend + UI |
| Grafana | `3000` | `rca-grafana` | dashboards (admin/admin) |

### Healthchecks

`docker-compose.yml` defines healthchecks for ClickHouse, MySQL, and Tempo.
Dependent services wait on them via `depends_on: condition: service_healthy`:

- **clickhouse**: `wget --spider http://localhost:8123/ping` every 5s, 20 retries.
- **mysql**: `mysqladmin ping` every 5s, 30 retries.
- **tempo**: `wget --spider http://localhost:3200/ready` every 5s, 30 retries.
- **otel-collector** waits for healthy ClickHouse + Tempo before starting.
- **grafana** depends on Tempo, ClickHouse, MySQL (start-order only).

`otel-collector` and `grafana` have `restart: unless-stopped` and are
considered healthy once they accept traffic; add explicit healthchecks in
prod if you want `service_healthy` gating.

### Persistence / volumes

Named volumes keep data across restarts:

| Volume | Service | Contents |
|---|---|---|
| `clickhouse-data` | clickhouse | `rca` + `rca_otel` databases |
| `mysql-data` | mysql | `rca` database (reports/runs) |
| `tempo-data` | tempo | trace blocks + WAL |
| `grafana-data` | grafana | users, dashboard edits, datasource state |

Back these with durable storage in prod (a managed volume / EBS / network FS).
To **wipe** all data (destructive): `docker compose down -v`.

Grafana dashboards and datasources are **bind-mounted read-only** from
`infra/grafana/`, so dashboard JSONs under
[`infra/grafana/dashboards/`](../infra/grafana/dashboards) are auto-provisioned
into the `RCA Agent` folder on every boot — no manual import needed.

---

## 2. App profile (future `docker-compose.prod.yml`)

The application (rca-server + agent + frontend) is deployed as a separate
overlay. The target layout:

```yaml
# docker-compose.prod.yml  (app profile — not yet committed)
services:
  rca-server:
    build: .
    image: rca-agent/server:latest
    env_file: .env
    environment:
      RCA_DATA_BACKEND: clickhouse        # prod reads from ClickHouse, not parquet
      RCA_CLICKHOUSE_HOST: clickhouse     # compose service DNS
      RCA_MYSQL_URL: mysql+pymysql://rca:${RCA_MYSQL_PASSWORD}@mysql:3306/rca
      RCA_OTEL_ENDPOINT: http://otel-collector:4317
      RCA_MEMORY_BACKEND: inmemory        # or a vector backend in prod
    ports:
      - "8000:8000"
    depends_on:
      clickhouse: { condition: service_healthy }
      mysql:      { condition: service_healthy }
      otel-collector: { condition: service_started }
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 6
    restart: unless-stopped

  rca-frontend:
    build: ./frontend
    image: rca-agent/frontend:latest
    environment:
      VITE_API_BASE: http://rca-server:8000
    ports:
      - "5173:80"     # serve built static assets
    depends_on:
      rca-server: { condition: service_healthy }
    restart: unless-stopped
```

Bring up infra + app together:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

In prod set `RCA_DATA_BACKEND=clickhouse` so the agent queries imported data
instead of reading parquet files from disk (`RCA_CASES_DIR` is then unused).

---

## 3. Environment configuration

All knobs are `RCA_*` env vars (see [`config.py`](../rca_agent/config.py),
[`.env.example`](../.env.example)). Production-critical ones:

| Var | Default | Prod note |
|---|---|---|
| `RCA_DEEPSEEK_API_KEY` | `""` | **required** for live runs; gate features on `settings.has_llm_key` |
| `RCA_DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | point at a DeepSeek-compatible gateway if needed |
| `RCA_DEEPSEEK_MODEL` | `deepseek-reasoner` | thinking-mode model |
| `RCA_REASONING_EFFORT` | `high` | thinking effort |
| `RCA_LLM_MAX_STEPS` | `25` | cap an investigation's ReAct loop |
| `RCA_LLM_MAX_TOKENS` | `8192` | per-request token cap |
| `RCA_DATA_BACKEND` | `parquet` | set `clickhouse` in prod |
| `RCA_CASES_DIR` | machine-local path | parquet backend only; **always set on a new machine** (default is developer-local and won't exist elsewhere); unused in clickhouse mode |
| `RCA_CLICKHOUSE_HOST` / `_PORT` | `localhost` / `8123` | point at the compose `clickhouse` service |
| `RCA_CLICKHOUSE_USER` / `_PASSWORD` | `rca` / `rca123` | dev creds; override in prod |
| `RCA_CLICKHOUSE_DATABASE` | `rca` | benchmark db; `rca_otel` is the OTel sink (fixed by collector) |
| `RCA_MYSQL_URL` | `mysql+pymysql://rca:rca123@localhost:3306/rca` | point at the compose `mysql` service |
| `RCA_MEMORY_BACKEND` | `inmemory` | use a persistent/vector backend in prod |
| `RCA_OTEL_ENDPOINT` | `http://localhost:4317` | OTLP gRPC to the collector |
| `RCA_OTEL_SERVICE_NAME` | `rca-agent` | service name tag on exported signals |
| `RCA_OTEL_ENABLED` | `true` | export traces/metrics/logs |
| `RCA_SERVER_HOST` / `RCA_SERVER_PORT` | `0.0.0.0` / `8000` | bind |

### Secrets handling

- **Never commit real secrets.** `.env` is gitignored; `.env.example` ships
  placeholder values only.
- In prod, inject secrets via the orchestrator: Docker Compose `env_file`
  (a file mode `0600`, not committed), Swarm/Kubernetes secrets, or a secrets
  manager. The ClickHouse/MySQL passwords in `docker-compose.yml` and
  [`INFRA_ACCESS.md`](INFRA_ACCESS.md) are **dev-only defaults** (`rca123` /
  `root123` / `admin`); override them in prod via env (note ClickHouse/MySQL
  users are created at first boot from compose env, so changing them requires
  `docker compose down -v` and a clean re-init).
- DeepSeek API keys are the only true external secret. Store them in your
  secrets manager and mount as `RCA_DEEPSEEK_API_KEY`.

---

## 4. Observability in prod

The app exports OTLP to the collector, which fans out:

- **traces** → Tempo (queryable in Grafana via the `Tempo` datasource).
- **metrics + logs** → ClickHouse `rca_otel` (queryable via the `ClickHouse`
  datasource, plugin `grafana-clickhouse-datasource`).
- **collector self-metrics** on `:8888` — scrape with Prometheus and add a
  `Prometheus` datasource if you want collector dashboards.

Grafana dashboards under [`infra/grafana/dashboards/`](../infra/grafana/dashboards)
auto-load and visualize `rca_runs_total`, `rca_steps_total`,
`rca_tool_calls_total`, `rca_llm_tokens` (Prometheus/ClickHouse), and
MySQL-backed report stats.

---

## 5. Scaling notes

- **Stateless app tier.** `rca-server` keeps no in-process state between
  requests (runs are persisted to MySQL and streamed via SSE). Scale it
  horizontally behind a load balancer; ensure SSE connections are sticky or use
  a shared pub/sub for the stream if you scale beyond one replica per active run.
- **ClickHouse** is the throughput-critical store for observability data. Give
  it dedicated CPU/RAM and separate the `rca` (benchmark) and `rca_otel`
  (observability) workloads onto different disks if volume grows.
- **MySQL** holds reports/runs only — low volume, but back it up regularly.
- **DeepSeek API** is the latency and cost bottleneck. Cap
  `RCA_LLM_MAX_STEPS` / `RCA_LLM_MAX_TOKENS` per environment; monitor
  `rca_llm_tokens` to catch runaway investigations.
- **Memory backend.** `inmemory` is fine for single-replica dev; in prod use a
  shared/persistent memory backend so seed knowledge and learned facts survive
  restarts and are visible to all replicas.

---

## 6. Reset / re-init

```bash
# Wipe ALL data and re-run init scripts:
docker compose down -v
docker compose up -d clickhouse mysql otel-collector tempo grafana

# Force-recreate a service after an infra config change
# (bind-mounted config doesn't change the compose definition):
docker compose up -d --force-recreate otel-collector
```
