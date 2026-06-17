# RCA Traces — model, API, SSE resilience, CLI

This document describes how an RCA investigation is represented as a **trace**,
how that trace is produced (streamed over SSE) and persisted (MySQL), and how to
inspect it (REST API + CLI). It is the ops / user-facing companion to
[`docs/API.md`](API.md) (wire format) and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
(component layout).

> 简要 (TL;DR): 一个 **run** = 一次 agent 调用；run 由一条 SSE 流实时产出，结束时落
> 库为一份 `RcaReport`（含完整有序 `RcaStep` 列表）。CLI 的 `runs` / `trace` 直接读取
> 这些已持久化的报告，无需前端即可复盘每个 case 的完整执行轨迹。

---

## 1. The trace model

A **run** is one RCA agent invocation for a case. Conceptually it has:

- a **status** ∈ `running | completed | truncated | error`
- lifecycle timestamps (`started_at` / `finished_at`)
- the `model` used (e.g. `deepseek-reasoner`) and `token_usage`
- an **ordered list of steps** (`RcaStep`) and a terminal `RcaReport`

The trace's atomic unit is the **step** (`RcaStep`, see
[`rca_agent/contracts/rca.py`](../rca_agent/contracts/rca.py)):

| field | meaning |
|---|---|
| `step_id` | stable id for the step |
| `case_id` | the case under investigation |
| `step_kind` | see table below |
| `thought` | reasoning_content excerpt (display only) |
| `tool_name`, `tool_args`, `tool_result`, `tool_result_text` | tool interaction detail |
| `hypothesis`, `confidence` | set on the `conclude` step |
| `entities` | entity refs touched by the step |
| `ts` | UTC timestamp |

### Step kinds and module interaction

`step_kind` ∈ `observe | hypothesize | investigate | tool_call | tool_result | reasoning | conclude | error`
(`StepKind` in [`contracts/rca.py`](../rca_agent/contracts/rca.py)):

| `step_kind` | what module interaction it represents |
|---|---|
| `reasoning` | a DeepSeek **thinking-mode** reasoning turn (carries a `thought` excerpt) |
| `tool_call` | the agent invoking a data-provider / tool (`tool_name`, `tool_args`) |
| `tool_result` | the tool/provider response (`tool_result_text`) |
| `observe` / `hypothesize` / `investigate` | structured ReAct phases |
| `conclude` | the terminal root-cause step (`hypothesis` + `confidence`) |
| `error` | a non-fatal failure during a step |

So a run is a sequence of *reasoning → tool_call → tool_result → … → conclude*
events; the full sequence is the case's complete execution trace.

---

## 2. How a trace is produced: SSE streaming

The server runs the agent and emits each step as it happens over a single SSE
stream ([`rca_agent/server/app.py`](../rca_agent/server/app.py)):

```
POST /rca/{case_id}            -> { case_id, backend, stream_url }
GET   /rca/{case_id}/stream    -> text/event-stream  (step | delta | report | error | done | ping)
```

The wire format is specified in [`docs/API.md`](API.md) and the frozen
[`contracts/streaming.py`](../rca_agent/contracts/streaming.py) (`SSEEvent`).
`seq` increases monotonically per event.

### SSE resilience — why streams don't drop

Two mechanisms keep long streams alive end-to-end:

1. **15s server heartbeat.** While the agent is silent (e.g. a long DeepSeek
   reasoning turn), the server emits an unnamed `ping` SSE message every
   `RCA_SSE_HEARTBEAT_SEC` seconds (default **15**). This re-arms the client's
   idle watchdog (the frontend closes a stream idle for ~60s) without
   cancelling an in-flight agent step. Tunable via env:
   ```bash
   export RCA_SSE_HEARTBEAT_SEC=15
   ```

2. **nginx SSE hardening.** The frontend nginx proxy
   ([`deploy/frontend.nginx.conf`](../deploy/frontend.nginx.conf)) configures the
   `/rca/` location so the proxy itself never buffers, caches, or times out the
   stream:
   ```nginx
   location /rca/ {
       proxy_pass http://rca-server:8000;
       proxy_http_version 1.1;
       proxy_set_header Connection "";
       proxy_set_header X-Accel-Buffering no;   # tell any intermediary NOT to buffer
       proxy_buffering off;                     # crucial for SSE streaming
       proxy_cache off;
       proxy_read_timeout 1h;                   # survive long reasoning turns
       proxy_send_timeout 1h;
       chunked_transfer_encoding off;           # emit a raw event stream
   }
   ```

### Durable partial traces (recovery)

When a run finishes, the server best-effort persists the final `RcaReport` (which
contains the full `steps` list) to MySQL (see §3). On a dropped/aborted
connection, the client can re-fetch the persisted report by `case_id`
(`GET /reports/{case_id}`) to recover the trace without re-running the agent.

> **Note (incremental step persistence).** Full per-step *incremental* persistence
> (each `RcaStep` written to a dedicated `rca_steps` table as it streams, so a
> dropped stream leaves a durable partial trace) is part of the planned Wave-4 / T1
> `TraceStore` work and is **not yet merged to `main`**. Today the durable record
> is the terminal `RcaReport` row (see §3 for the current model and the planned
> surface).

---

## 3. Persistence — MySQL

The storage backend is **MySQL** (db `rca`; see
[`rca_agent/store/schema.sql`](../rca_agent/store/schema.sql) and
[`docs/INFRA_ACCESS.md`](INFRA_ACCESS.md) for connection details).

### Current schema (on `main`)

| table | holds |
|---|---|
| `rca_runs` | run lifecycle rows (`run_id`, `case_id`, `status`, `model`, `started_at`/`finished_at`, `token_usage`) |
| `rca_reports` | one persisted `RcaReport` per finished run — `root_cause_json` + `steps_json` (the full ordered `RcaStep` list) + `confidence` |
| `cases` | task + topology metadata per case |
| `config` | simple key/value application config |

The full ordered step trace for a finished run lives in the **`steps_json`**
column of its `rca_reports` row. See [`rca_agent/store/mysql_store.py`](../rca_agent/store/mysql_store.py)
for the store API (`save_report`, `get_report`, `list_reports`, `start_run`,
`finish_run`).

### REST API (current, on `main`)

| method + path | returns |
|---|---|
| `GET  /health` | `{ "status": "ok" }` |
| `GET  /cases` | `{ "cases": ["t001", ...] }` |
| `POST /rca/{case_id}?backend=parquet` | `{ case_id, backend, stream_url }` |
| `GET  /rca/{case_id}/stream` | SSE stream (see §2) |
| `GET  /reports/{case_id}` | the most recent persisted `RcaReport` for a case |

Example:

```bash
# Start a run and stream it
curl -X POST "http://localhost:8000/rca/t001?backend=parquet"
# -> {"case_id":"t001","backend":"parquet","stream_url":"/rca/t001/stream?backend=parquet"}
curl -N "http://localhost:8000/rca/t001/stream?backend=parquet"

# Fetch the persisted report after the run finishes
curl "http://localhost:8000/reports/t001" | jq '.root_cause.summary, (.steps | length)'
```

### Planned surface (Wave-4 / T1 — not yet on `main`)

The intended trace-inspection API (per the Wave-4 design, pending the T1
`TraceStore` merge) is:

| method + path | returns |
|---|---|
| `GET /runs?case_id=&limit=` | list of run summaries (incl. `step_count`) |
| `GET /runs/{run_id}` | `{ run, steps }` — full run + its steps |
| `GET /runs/{run_id}/steps` | just the ordered steps |
| `GET /cases/{case_id}/runs` | runs for a case |

When T1 lands, `POST /rca/{case_id}` will additionally return a `run_id`, the
`rca_steps` table will hold per-step rows (persisted incrementally as they
stream), and the `runs` / `trace` CLI subcommands will read directly from these
endpoints. Until then they read the current `rca_reports` persistence (see §4).

---

## 4. CLI — inspect traces without the frontend

Two subcommands surface the persisted trace from the shell
([`rca_agent/cli.py`](../rca_agent/cli.py)). They read MySQL via `MysqlStore`
(today: `list_reports` / `get_report`) and never touch the network or DeepSeek.

> **Heads-up (current limitation on `main`).** Today a persisted run is stored
> as one `rca_reports` row (its `steps_json` carries the full ordered step
> list). The `rca_reports` table does **not** yet carry `model` / `status` /
> `token_usage` columns, and the server does not yet log the `report_id` it
> mints on persist. As a result `runs` shows `model=-` and the persisted-run
> `status`/`token_usage` may not reflect the live run, and a `report_id` to pass
> to `trace` is currently obtained from a `run -o report.json` file (or from the
> DB directly) rather than from `runs` output. The planned Wave-4 / T1
> `TraceStore` + `/runs` REST surface (§3) removes this limitation.

### `rca-agent runs` — list recent runs

```bash
rca-agent runs                       # last 50 persisted reports
rca-agent runs --case t001           # filter by case
rca-agent runs --limit 10            # cap the page (env RCA_RUNS_LIMIT, default 50)
```

Example output:

```
3 run(s):
  case=t001  status=completed  model=-  steps=18
  case=t002  status=completed  model=-  steps=24
  case=t003  status=completed  model=-  steps=7

(tip: `rca-agent trace <report_id>` prints a run's full step trace; get a
report_id from a `run -o report.json` file or the `rca_reports` table.)
```

- Empty result → `no runs found.` and exit 0.
- Store error (MySQL unreachable) → a clear `error: ...` on stderr and exit 1
  (never a traceback).

### `rca-agent trace <report_id>` — print a run's full step trace

```bash
rca-agent trace 0192f8c1a4b748e29a8f1c2d3b4e5f60
```

Prints the run header, then each `RcaStep` in order
(`#idx  step_kind  tool=…  :: <thought / result / tool_args>  [ts]`), then the
root cause:

```
=== trace 0192f8c1… :: case=t001 ===
status=completed  model=-  steps=18  confidence=0.82
alert: checkout 错误次数告警
#1   reasoning    :: checkout error-rate spike coincides with cart latency rise…  [2026-…]
#2   tool_call    tool=query_metrics :: {"entity_names": ["cart"], …}  [2026-…]
#3   tool_result  tool=query_metrics :: cart p99 latency 2.4s (baseline 180ms)  [2026-…]
…
#18  conclude     :: conf=0.82  slow inventory DB queries saturated cart→inventory HTTP  [2026-…]
======================================================================
ROOT CAUSE: cart service latency regression caused checkout PlaceOrder errors…
FAULT TYPE: db.slow_query
TOKENS: {'prompt_tokens': 18203, 'completion_tokens': 2417, 'reasoning_tokens': 9050}
```

(`TOKENS:` and an accurate `model`/`status` appear when the report carries them;
for rows persisted by the current `rca_reports` schema they may be absent — see
the heads-up above.)

- `report_id` must be a 32-char hex uuid (e.g. from a `run -o report.json` file
  or the `rca_reports` table); otherwise a usage error and exit 2.
- Unknown id → `error: no such run: <id>` on stderr, exit 1.
- Store error → clear stderr message, exit 1.

See also `rca-agent run -o report.json` which writes the same `RcaReport` JSON
to disk for an ad-hoc (non-server) run — this is currently the most reliable way
to obtain a `report_id` for `trace`.

