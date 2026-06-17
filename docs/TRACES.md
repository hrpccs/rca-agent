# RCA Traces — model, API, SSE resilience, CLI

How an RCA investigation is represented as a **trace**, how it is produced
(streamed over SSE) and **persisted incrementally** (MySQL), and how to inspect
it (REST API + CLI). Ops / user-facing companion to [`docs/API.md`](API.md)
(wire format) and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) (component layout).

> 简要 (TL;DR): 一个 **run** = 一次 agent 调用。run 由一条 SSE 流实时产出，**每个
> step 边产生边落库**（`rca_steps`），所以即便流中途断开，已产出的部分轨迹也会被
> 持久化。CLI 的 `runs` / `trace` 与 `/runs` REST API 直接读取这些轨迹，无需前端即可
> 复盘每个 case 的完整执行轨迹（含失败 / 截断的 run）。

---

## 1. The trace model

A **run** is one RCA agent invocation for a case:

- a **status** ∈ `running | completed | truncated | error`
- lifecycle timestamps (`started_at` / `finished_at`)
- the `model` used (e.g. `deepseek-reasoner`) and `token_usage`
- an **ordered list of steps** (`RcaStep`); the terminal step is a `conclude`
  carrying the root cause. A successful run also has an `RcaReport` row.

The atomic unit is the **step** (`RcaStep`, see
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

`step_kind` ∈ `observe | hypothesize | investigate | tool_call | tool_result | reasoning | conclude | error`:

| `step_kind` | what module interaction it represents |
|---|---|
| `reasoning` | a DeepSeek **thinking-mode** reasoning turn (carries a `thought` excerpt); also used to surface **memory** prior-knowledge retrieval |
| `tool_call` | the agent invoking a data-provider / tool (`tool_name`, `tool_args`) |
| `tool_result` | the tool/provider response (`tool_result_text`) |
| `observe` / `hypothesize` / `investigate` | structured ReAct phases |
| `conclude` | the terminal root-cause step (`hypothesis` + `confidence`) |
| `error` | a non-fatal failure during a step |

A run is therefore a sequence of *reasoning → tool_call → tool_result → … →
conclude* events — the case's complete execution trace, capturing every module
interaction (LLM thinking, tool/provider calls, memory retrieval).

---

## 2. How a trace is produced: SSE streaming

The server runs the agent and emits each step as it happens over a single SSE
stream ([`rca_agent/server/app.py`](../rca_agent/server/app.py)):

```
POST /rca/{case_id}?backend=parquet  -> { case_id, backend, run_id, stream_url }
GET   /rca/{case_id}/stream?backend=parquet&run_id=…
                                      -> text/event-stream (step | delta | report | error | done | ping)
```

`POST /rca/{case_id}` mints a `run_id` (a `rca_runs` row in `running` state) and
returns it; the stream persists every step under that `run_id` (see §3). The wire
format is in [`docs/API.md`](API.md) and the frozen
[`contracts/streaming.py`](../rca_agent/contracts/streaming.py); `seq` increases
monotonically per event.

### SSE resilience — why long streams don't drop

1. **15s server heartbeat.** While the agent is silent (e.g. a long DeepSeek
   reasoning turn), the server emits an unnamed `ping` SSE message every
   `RCA_SSE_HEARTBEAT_SEC` seconds (default **15**). This re-arms the client's
   idle watchdog (the frontend closes a stream idle for ~60s) **without**
   cancelling the in-flight agent step. Tunable via env:
   ```bash
   export RCA_SSE_HEARTBEAT_SEC=15
   ```

2. **nginx SSE hardening.** The frontend nginx proxy
   ([`deploy/frontend.nginx.conf`](../deploy/frontend.nginx.conf)) configures the
   `/rca/` location so the proxy never buffers, caches, or times out the stream:
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

Steps are persisted **incrementally** — each `RcaStep` is written to the
`rca_steps` table as it streams (see §3). So if the connection drops mid-run,
the partial trace is already durable. The frontend detects a dropped connection
and best-effort re-fetches the persisted trace by `run_id` (`GET /runs/{run_id}`)
and shows it, instead of erroring out. Runs that end without a report (client
disconnect, or the producer ending cleanly without a final answer) are closed
with `status="truncated"`.

---

## 3. Persistence — MySQL

Storage is **MySQL** (db `rca`; see
[`rca_agent/store/schema.sql`](../rca_agent/store/schema.sql) and
[`docs/INFRA_ACCESS.md`](INFRA_ACCESS.md) for connection details).

| table | holds |
|---|---|
| `rca_runs` | one row per agent invocation — `run_id`, `case_id`, `status`, `model`, `started_at`/`finished_at`, `token_usage` |
| `rca_steps` | **one row per step**, persisted incrementally as it streams — `step_id`, `run_id`, `case_id`, `seq`, `step_kind`, `payload` (full `RcaStep` JSON) |
| `rca_reports` | one persisted `RcaReport` per *finished* run — `root_cause_json` + `steps_json` + `confidence`, linked to its run via `run_id` |
| `cases` | task + topology metadata per case |
| `config` | simple key/value application config |

The authoritative per-step trace is `rca_steps` (ordered by `seq`); `rca_reports`
is the terminal document for completed runs. See
[`rca_agent/store/mysql_store.py`](../rca_agent/store/mysql_store.py) for the store
API — reports (`save_report`/`get_report`/`list_reports`), runs
(`start_run`/`finish_run`), and the trace API
(`append_step`/`list_steps`/`list_runs`/`get_run`); the
[`TraceStore`](../rca_agent/store/trace_store.py) Protocol is the structural
contract (with an `InMemoryTraceStore` for tests/dev).

### REST API

| method + path | returns |
|---|---|
| `GET  /health` | `{ "status": "ok" }` |
| `GET  /cases` | `{ "cases": ["t001", ...] }` |
| `POST /rca/{case_id}?backend=parquet` | `{ case_id, backend, run_id, stream_url }` |
| `GET  /rca/{case_id}/stream?backend=&run_id=` | SSE stream (see §2) |
| `GET  /runs?case_id=&limit=` | list of run summaries (incl. `step_count`, `status`, timing) |
| `GET  /runs/{run_id}` | `{ run, steps }` — full run + its ordered steps |
| `GET  /runs/{run_id}/steps` | just the ordered steps |
| `GET  /cases/{case_id}/runs` | runs for a case |
| `GET  /reports/{case_id}` | the most recent `RcaReport` for a case |

Example:

```bash
# Start a run (mints run_id) and stream it
curl -sX POST "http://localhost:8000/rca/t001?backend=parquet"
# -> {"case_id":"t001","backend":"parquet","run_id":"0192…","stream_url":"/rca/t001/stream?backend=parquet&run_id=0192…"}
curl -N "http://localhost:8000/rca/t001/stream?backend=parquet&run_id=0192f8c1a4b748e29a8f1c2d3b4e5f60"

# Inspect persisted runs + a full step trace (no re-run needed)
curl -s "http://localhost:8000/runs?case_id=t001" | jq '.runs[] | {run_id: .run_id[0:8], status, step_count}'
curl -s "http://localhost:8000/runs/0192f8c1a4b748e29a8f1c2d3b4e5f60" | jq '.run.status, (.steps | length)'
```

All `/runs*` endpoints return **503** if the store is unavailable and **404** for
an unknown `run_id`; persistence is best-effort and never breaks the live stream.

---

## 4. CLI — inspect traces without the frontend

Two subcommands surface the persisted trace from the shell
([`rca_agent/cli.py`](../rca_agent/cli.py)). They read the trace store
(`list_runs`/`get_run`/`list_steps`) via `MysqlStore` and never touch the network
or DeepSeek.

### `rca-agent runs` — list recent runs

```bash
rca-agent runs                       # last 50 runs (all statuses, incl. errored/truncated)
rca-agent runs --case t001           # filter by case_id
rca-agent runs --limit 10            # cap the page (env RCA_RUNS_LIMIT, default 50)
```

Example output:

```
3 run(s):
  run=0192f8c1a4b7  case=t001  status=completed  model=deepseek-reasoner  steps=18  started=2026-06-17T16:40:…
  run=aa51c09e7721  case=t004  status=error      model=deepseek-reasoner  steps=7   started=2026-06-17T16:51:…
  run=77d3b8e1c4a2  case=t002  status=truncated  model=deepseek-reasoner  steps=24  started=2026-06-17T16:55:…

(tip: `rca-agent trace <run_id>` prints a run's full step trace.)
```

- Empty result → `no runs found.` and exit 0.
- Store error (MySQL unreachable) → a clear `error: ...` on stderr and exit 1
  (never a traceback).

### `rca-agent trace <run_id>` — print a run's full step trace

```bash
rca-agent trace 0192f8c1a4b748e29a8f1c2d3b4e5f60
```

The id may be a **`run_id`** (trace store: `rca_steps`) or a **`report_id`**
(`rca_reports`, e.g. from `run -o` / eval); both are 32-char hex uuids. Prints the
run header, then each `RcaStep` in order, then the root cause (from the terminal
`conclude` step):

```
=== trace 0192f8c1… :: case=t001 ===
status=completed  model=deepseek-reasoner  steps=18  started=2026-…  finished=2026-…
TOKENS: {'prompt_tokens': 18203, 'completion_tokens': 2417, 'reasoning_tokens': 9050}
#1   reasoning    :: memory: retrieved 3 prior(s) for 'checkout 错误次数告警'  [2026-…]
#2   reasoning    :: checkout error-rate spike coincides with cart latency rise…  [2026-…]
#3   tool_call    tool=query_metrics :: {"entity_names": ["cart"], …}  [2026-…]
#4   tool_result  tool=query_metrics :: cart p99 latency 2.4s (baseline 180ms)  [2026-…]
…
#18  conclude     :: conf=0.82  slow inventory DB queries saturated cart→inventory HTTP  [2026-…]
======================================================================
ROOT CAUSE: cart service latency regression caused checkout PlaceOrder errors…
CONFIDENCE: 0.82
```

- `run_id`/`report_id` must be a 32-char hex uuid; otherwise a usage error and
  exit 2.
- Unknown id → `error: no such run: <id>` on stderr, exit 1.
- Store error → clear stderr message, exit 1.

See also `rca-agent run -o report.json`, which writes the same `RcaReport` JSON
to disk for an ad-hoc (non-server) run — its `report_id` can also be passed to
`trace`.
