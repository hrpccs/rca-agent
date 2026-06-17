# API

The `rca-server` is a FastAPI app exposing a small REST + Server-Sent Events
(SSE) API. It starts an RCA investigation, streams the steps/report as they
happen, and persists the final report to MySQL.

Base URL (default): `http://localhost:8000` (`RCA_SERVER_HOST` / `RCA_SERVER_PORT`).

All structured payloads are JSON and match the frozen models in
[`rca_agent/contracts/`](../rca_agent/contracts):

- `RcaStep` / `RcaReport` / `RootCause` — [`contracts/rca.py`](../rca_agent/contracts/rca.py)
- `SSEEvent` / `SSEEventKind` / `SSEDelta` — [`contracts/streaming.py`](../rca_agent/contracts/streaming.py)

---

## Endpoints

### `GET /health`

Liveness probe.

**200 OK**
```json
{ "status": "ok" }
```

---

### `GET /cases`

List known benchmark case ids discoverable under `RCA_CASES_DIR`.

**200 OK**
```json
{ "cases": ["t001", "t002", "t003"] }
```

---

### `POST /rca/{case_id}`

Start an RCA investigation for `case_id`. Returns a handle that can be polled
or streamed. The server loads the `Case`, builds a `DataProvider`
(parquet or clickhouse per `RCA_DATA_BACKEND`), seeds memory, and kicks off the
agent loop asynchronously.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `case_id` | string | benchmark case id, e.g. `t001` |

**Optional request body**
```json
{ "model": "deepseek-reasoner", "reasoning_effort": "high", "max_steps": 25 }
```
Any subset of these may be omitted; server-side defaults come from `Settings`.

**200 OK**
```json
{
  "case_id": "t001",
  "run_id": "run-0192f8c1-…",
  "stream": "/rca/t001/stream"
}
```

**404 Not Found** — unknown `case_id`.

---

### `GET /rca/{case_id}/stream`

Server-Sent Events stream of the in-progress (or most recent) investigation for
`case_id`. This is the primary consumer-facing endpoint; the frontend reads it.

**Response headers**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

Each event is serialized by `sse_format` to the wire format:

```
event: <kind>\ndata: <json>\n\n
```

`<kind>` ∈ `step | delta | report | error | done | ping` (see `SSEEventKind`).
`<json>` is the full `SSEEvent` (`{event, case_id, data, seq}`) serialized with
`model_dump(mode="json")`. `seq` increases monotonically.

#### Event types and `data` payloads

| `event` | `data` | Meaning |
|---|---|---|
| `step` | `RcaStep` | one investigation step |
| `delta` | `SSEDelta` | fine-grained streaming token (optional; agent may emit only `step`s) |
| `report` | `RcaReport` | the completed report |
| `error` | `dict` (`{"error": "..."}`) | the run failed |
| `done` | `dict` (often `{}`) | terminal; stream ends after this |
| `ping` | `dict` (often `{}`) | keepalive |

#### `RcaStep` (`event: step`)

```json
{
  "event": "step",
  "case_id": "t001",
  "data": {
    "step_id": "step-3-a1b2c3",
    "case_id": "t001",
    "step_kind": "tool_call",
    "thought": "Error count on checkout rose; check its downstream cart dependency.",
    "tool_name": "query_metrics",
    "tool_args": { "entity_names": ["cart"], "metrics": ["error_rate"], "window": { "start": "...", "end": "..." } },
    "tool_result": null,
    "tool_result_text": null,
    "hypothesis": null,
    "confidence": null,
    "entities": ["cart"],
    "ts": "2026-04-25T05:20:01.234567+00:00"
  },
  "seq": 5
}
```

`step_kind` ∈ `observe | hypothesize | investigate | tool_call | tool_result | reasoning | conclude | error`.

A reasoning step (carrying a `reasoning_content` excerpt for display):
```json
{
  "event": "step",
  "case_id": "t001",
  "data": {
    "step_id": "step-2-d4e5f6",
    "case_id": "t001",
    "step_kind": "reasoning",
    "thought": "The checkout error-rate spike coincides with a cart latency rise; cart calls inventory over HTTP, so an inventory slowdown would propagate upward.",
    "ts": "2026-04-25T05:19:40+00:00"
  },
  "seq": 3
}
```

#### `SSEDelta` (`event: delta`) — optional fine-grained tokens

```json
{
  "event": "delta",
  "case_id": "t001",
  "data": { "kind": "reasoning", "text": "Cart latency is rising…", "step_id": "step-2-d4e5f6" },
  "seq": 4
}
```

`data.kind` ∈ `text | reasoning | tool_call`. The agent may omit `delta` events
entirely and emit only `step`s; clients must tolerate both.

#### `RcaReport` (`event: report`) — terminal success

```json
{
  "event": "report",
  "case_id": "t001",
  "data": {
    "case_id": "t001",
    "task_id": "t001",
    "alert_title": "checkout 错误次数告警",
    "root_cause": {
      "summary": "cart service latency regression caused checkout PlaceOrder errors; root cause is inventory DB (MySQL RDS) slow queries saturating cart→inventory HTTP calls.",
      "entity_refs": [
        { "entity_id": "…", "entity_name": "cart", "entity_type": "apm.service" },
        { "entity_id": "…", "entity_name": "inventory", "entity_type": "apm.service" }
      ],
      "fault_type": "db.slow_query",
      "evidence": [
        "step-7: cart p99 latency 2.4s (baseline 180ms)",
        "step-9: inventory DB connections saturated; slow query log shows full-table scan"
      ],
      "confidence": 0.82,
      "contributing_factors": [
        "checkout retries amplified cart load",
        "no circuit breaker between cart and inventory"
      ],
      "recommended_actions": [
        "Add index on inventory.orders(product_id) to eliminate the scan",
        "Add a cart→inventory circuit breaker / bulkhead"
      ]
    },
    "steps": [ /* …RcaStep… */ ],
    "started_at": "2026-04-25T05:18:12+00:00",
    "finished_at": "2026-04-25T05:24:55+00:00",
    "model": "deepseek-reasoner",
    "token_usage": { "prompt_tokens": 18203, "completion_tokens": 2417, "reasoning_tokens": 9050 },
    "status": "completed"
  },
  "seq": 42
}
```

`status` ∈ `completed | error | truncated`. `confidence` is `0..1`.

#### `error` and `done`

```
event: error
data: {"event":"error","case_id":"t001","data":{"error":"DeepSeek API 400: missing reasoning_content"},"seq":6}

event: done
data: {"event":"done","case_id":"t001","data":{},"seq":43}

```

After `done` (or `error`), the server closes the stream.

---

### `GET /reports/{id}`

Fetch a previously persisted `RcaReport` by its report/run id.

**Path parameters**

| Name | Type | Description |
|---|---|---|
| `id` | string | report id returned by `POST /rca/{case_id}` or seen in a `report` event |

**200 OK** — an `RcaReport` (same shape as the `report` event `data`).

**404 Not Found** — no such report, or the run has not finished.

---

## Consuming the stream

A minimal SSE client (the frontend does this with `EventSource`):

```bash
curl -N http://localhost:8000/rca/t001/stream
```

```js
const es = new EventSource("/rca/t001/stream");
es.addEventListener("step",   (e) => renderStep(JSON.parse(e.data)));
es.addEventListener("delta",  (e) => renderDelta(JSON.parse(e.data)));
es.addEventListener("report", (e) => { renderReport(JSON.parse(e.data)); });
es.addEventListener("error",  (e) => showError(JSON.parse(e.data)));
es.addEventListener("done",   ()   => es.close());
```

Note the SSE convention: the browser's `EventSource` delivers the `data:` line
to the typed listener; the `event:` line selects the listener. The `data` JSON
is the full `SSEEvent` envelope (`{event, case_id, data, seq}`), and the
investigation payload lives at `.data` inside it.

---

> See [`TRACES.md`](TRACES.md) for the trace *model* (runs / steps / step kinds),
> SSE resilience (heartbeat + nginx hardening), and the `runs` / `trace` CLI
> subcommands for inspecting persisted traces from the shell.
