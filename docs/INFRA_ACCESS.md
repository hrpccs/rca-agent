# Infrastructure Access Manifest

> Persisted record of every external component this project brings up: how to
> reach it, how to authenticate, and how to reset. **Update this file whenever
> creds/ports change.** All services run via `docker compose` (see
> `docker-compose.yml`).

Start / stop everything:

```bash
docker compose up -d clickhouse mysql otel-collector tempo grafana   # start
docker compose ps                                                     # status
docker compose down                                                   # stop (keeps volumes)
docker compose down -v                                                # stop + WIPE data
```

## Connection summary

| Service | Host:Port (host) | Container | User | Password | Purpose |
|---|---|---|---|---|---|
| ClickHouse HTTP | `localhost:8123` | `rca-clickhouse` | `rca` | `rca123` | data-provider backend + OTel metrics/logs |
| ClickHouse native | `localhost:9000` | `rca-clickhouse` | `rca` | `rca123` | clickhouse-client / loader |
| MySQL | `localhost:3306` | `rca-mysql` | `rca` / `root` | `rca123` / `root123` | rca-server persistence |
| OTel OTLP gRPC | `localhost:4317` | `rca-otelcol` | — | — | app exports traces/metrics/logs |
| OTel OTLP HTTP | `localhost:4318` | `rca-otelcol` | — | — | app exports (HTTP) |
| OTel metrics | `localhost:8888` | `rca-otelcol` | — | — | collector self-metrics |
| Tempo UI | `localhost:3200` | `rca-tempo` | — | — | trace backend + UI |
| Grafana | `localhost:3000` | `rca-grafana` | `admin` | `admin` | dashboards |

Databases:
- ClickHouse: `rca` (benchmark data), `rca_otel` (OTel metrics/logs)
- MySQL: `rca`

## ClickHouse

```bash
# native client via docker (no local client needed):
docker exec -it rca-clickhouse clickhouse-client --user rca --password rca123 --database rca -q "SHOW TABLES"
# HTTP:
curl 'http://localhost:8123/?user=rca&password=rca123&database=rca' --data-binary 'SHOW DATABASES'
```
- **Reset:** `docker compose down -v && docker compose up -d clickhouse` (re-runs `infra/clickhouse/init.sql`).

## MySQL

```bash
docker exec -it rca-mysql mysql -urca -prca123 -Drca -e "SHOW TABLES;"
# or as root:
docker exec -it rca-mysql mysql -uroot -proot123 -e "SHOW DATABASES;"
```
- **Reset:** `docker compose down -v && docker compose up -d mysql` (re-runs `infra/mysql/init.sql`).

## Tempo / Grafana

- Tempo: <http://localhost:3200> (UI + `/ready`). Traces pushed by otel-collector over the compose net.
- Grafana: <http://localhost:3000>, `admin` / `admin`. Provisioned datasources: **Tempo** (traces),
  **ClickHouse** (metrics/logs, plugin `grafana-clickhouse-datasource`), **MySQL** (rca-server state).
  Dashboards auto-load from `infra/grafana/dashboards/`.

## OTel pipeline

App (`rca_agent.observability`) → OTLP gRPC `localhost:4317` → otel-collector →
**traces**: Tempo (`http://tempo:4318`); **metrics + logs**: ClickHouse `rca_otel`.
Collector config: `infra/otelcol/config.yaml`.

## How to reset all infra to a clean state

```bash
docker compose down -v        # remove containers + named volumes (WIPES ALL DATA)
docker compose up -d clickhouse mysql otel-collector tempo grafana
```

---

<!-- The section below is filled in by the coordinator after the first `docker compose up`
     and verified by healthchecks. -->

## Verified at bring-up

_(to be filled: timestamps, healthcheck results, sample query outputs)_
