# Architecture

This document describes the production-grade LLM Root Cause Analysis (RCA) agent:
its nine components, how data flows between them, the DeepSeek thinking-mode +
`reasoning_content` echo invariant, the contract-based module boundaries, and the
SSE streaming contract that binds the server to the frontend.

It is the canonical reference for every worker unit. When code and this document
disagree, the frozen contracts in [`rca_agent/contracts/`](../rca_agent/contracts)
are the source of truth; update this doc to match.

---

## 1. System overview

The agent investigates observability data (alerts, events, metrics, logs, traces,
topology) the way a human SRE would: it observes the alert, forms hypotheses,
calls tools to fetch evidence, reasons about the results, and converges on a root
cause with confidence, evidence, and recommended actions. It is powered by a
DeepSeek **thinking-mode** model that streams both `reasoning_content` (the
private chain-of-thought) and final `content`, and supports OpenAI-compatible
tool calls.

```
                 ┌─────────────────────────────────────────────────────────┐
   RCA request → │ rca-server (FastAPI, SSE)  →  mysql (state/reports)      │
                 │   └ rca-agent core (ReAct loop)                           │
                 │       ├ llm client (DeepSeek thinking/stream)             │
                 │       ├ context manager (msg assembly + compress)         │
                 │       ├ tools ──┐  (query metrics/logs/traces/...)        │
                 │       └ memory ┤  (runbooks / SOPs / domain facts)        │
                 └─────────────────┼─────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┴────────────────────────────────────────┐
        │  data-provider (parquet | clickhouse)                              │
        │    metrics / logs / traces / events / alerts / topology            │
        │    (one Protocol, two backends, identical result models)           │
        └──────────────────────────┬────────────────────────────────────────┘
                                   │
   OpenTelemetry ─ otel-collector ─┬─ Tempo (traces) ─────────────────────┐
                                   └─ ClickHouse rca_otel (metrics/logs) ─┤── Grafana (dashboards)
   frontend (React+Vite)  ←─ SSE ── rca-server ────────────────────────────┘
```

---

## 2. The nine components

### 2.1 LLM RCA core (`rca_agent/agent`)

The brain: a ReAct-style loop that drives a single RCA investigation for one
`case_id`. Per turn it:

1. Asks the `ContextManager` to assemble the messages array to send.
2. Streams the LLM via the `LLMClient`, consuming typed `LLMStreamDelta` events.
3. If the model emitted tool calls, validates them via `validate_tool_call` and
   dispatches each to its `ToolHandler`, collecting `ToolResult`s.
4. Records the assistant turn (content + `reasoning_content` + tool_calls) and
   the tool results back into the `ContextState`.
5. Emits an `RcaStep` (`StepKind` = `observe` / `hypothesize` / `investigate`
   / `tool_call` / `tool_result` / `reasoning` / `conclude` / `error`) downstream.

The loop terminates when the model produces a final root-cause answer
(`StepKind.CONCLUDE`) or hits `RCA_LLM_MAX_STEPS`, at which point it assembles an
`RcaReport` containing a `RootCause` (summary, `entity_refs`, `fault_type`,
`evidence`, `confidence`, `contributing_factors`, `recommended_actions`).

### 2.2 Data provider (`rca_agent/providers`)

The single most important integration contract. `DataProvider` is a
`@runtime_checkable` Protocol with one query method per modality:

```python
query_metrics(f: MetricFilter)   -> list[MetricSeries]
query_logs(f: LogFilter)         -> list[LogLine]
query_traces(f: TraceFilter)     -> list[Trace]
query_events(f: EventFilter)     -> list[K8sEvent]
query_alerts(f: AlertFilter)     -> list[CloudEvent]
query_topology(f: TopologyFilter)-> TopologySubgraph
modalities()                     -> list[Modality]
```

Two backends implement it and return **identical** result models:

- **parquet** — reads the benchmark `.parquet`/`.json` files directly (dev,
  eval, CI; no infra required).
- **clickhouse** — queries the same data after import (production; also serves
  as the OTel sink).

Because the agent tools call the Protocol and consume the result models, they
**never branch on the backend**. Switching backends is a config knob
(`RCA_DATA_BACKEND`), not a code change.

### 2.3 Memory (`rca_agent/memory` + `memory/seed/`)

`MemoryStore` is a `@runtime_checkable` Protocol storing the knowledge a human
SRE consults during triage: app-specific docs, runbooks/SOPs, and
business-agnostic domain facts. Its key property is **efficient, on-demand
retrieval** — not stuffing everything into the LLM context. Storage
(in-process dict, vector DB, external service) and retrieval method (keyword,
TF-IDF, embedding) are both pluggable.

Items carry a `kind` (`runbook` | `sop` | `domain_fact` | `metric_obs` |
`log_obs` | `hypothesis` | `evidence`) and a `case_id` of `"__global__"` for
cross-case/domain knowledge. Seed knowledge lives as markdown under
[`memory/seed/`](../memory/seed) and is indexed at startup.

### 2.4 Context manager (`rca_agent/context`)

Owns the chat-message window and the DeepSeek `reasoning_content` echo
invariant (see section 4). It is the **only** module permitted to manipulate
`reasoning_content`. It exposes `init`, `append_assistant`, `append_tool_result`,
`assemble_turn`, and `compress` (summarize oldest turns to fit a token budget
while preserving the reasoning echo for any retained tool-bearing turn).

### 2.5 RCA server (`rca_agent/server` + `rca_agent/store`)

The target design is a FastAPI app exposing the REST/SSE API (see
[`API.md`](API.md)); the endpoints and the MySQL persistence layer are owned by a
dedicated server/store unit and may not yet be implemented in the current tree
(`rca_agent/server` currently ships only its package skeleton). The intended API:

- `POST /rca/{case_id}` — start an RCA run (returns a run/report id).
- `GET  /rca/{case_id}/stream` — Server-Sent Events stream of `SSEEvent`s.
- `GET  /reports/{id}` — fetch a completed `RcaReport`.
- `GET  /cases` — list known cases.
- `GET  /health` — liveness.

Persistence (cases, configs, runs, reports) is intended to live in MySQL via
`rca_agent/store`. The server bridges the agent core's async step stream to the
SSE wire format using `rca_agent.contracts.sse_format`.

- `POST /rca/{case_id}` — start an RCA run (returns a run/report id).
- `GET  /rca/{case_id}/stream` — Server-Sent Events stream of `SSEEvent`s.
- `GET  /reports/{id}` — fetch a completed `RcaReport`.
- `GET  /cases` — list known cases.
- `GET  /health` — liveness.

Persistence (cases, configs, runs, reports) lives in MySQL via
`rca_agent/store`. The server bridges the agent core's async step stream to the
SSE wire format using `rca_agent.contracts.sse_format`.

### 2.6 Monitoring & eval (`rca_agent/observability` + `rca_agent/eval` + Grafana)

`rca_agent.observability` exports OpenTelemetry traces/metrics/logs over OTLP
gRPC to the collector. `rca_agent.eval` scores agent output against the
benchmark ground truth. Grafana dashboards under
[`infra/grafana/dashboards/`](../infra/grafana/dashboards) visualize agent
metrics (`rca_runs_total`, `rca_steps_total`, `rca_tool_calls_total`,
`rca_llm_tokens`) and MySQL-backed report stats.

### 2.7 Frontend (`frontend/`)

The target design is a React + Vite single-page app (owned by a dedicated
frontend unit; `frontend/` may not yet exist in the current tree) that consumes
the SSE stream and renders the investigation as it unfolds: reasoning steps,
tool calls and results, and the final report. Its TypeScript types are intended
to be generated from the same JSON schema as the Python
`SSEEvent`/`RcaStep`/`RcaReport` models, so the two sides never drift.

### 2.8 ClickHouse

Two databases:

- **`rca`** — benchmark data imported by the loader (the `clickhouse` data
  provider backend queries this).
- **`rca_otel`** — observability signals exported by the otel-collector (app
  metrics + logs; tables auto-created by the clickhouse exporter).

Exposed on HTTP `:8123` and native TCP `:9000`.

### 2.9 MySQL

The rca-server's persistence layer: cases, configs, runs, and reports. Schema is
bootstrapped by [`infra/mysql/init.sql`](../infra/mysql/init.sql) (creates the
`rca` database and grants). The Grafana `MySQL` datasource (uid `mysql`) reads
report/run stats for dashboards.

---

## 3. Data flow

**Investigation flow (per `POST /rca/{case_id}`):**

1. The server loads the `Case` (task + topology) via `rca_agent.cases.load_case`.
2. It constructs a `DataProvider` (parquet or clickhouse, per `RCA_DATA_BACKEND`)
   scoped to that case, and binds a `MemoryStore` (seeded from `memory/seed/`).
3. The agent core initializes a `ContextState` with a system prompt and enters
   the ReAct loop.
4. Each turn: `ContextManager.assemble_turn` → `LLMClient.stream` →
   tool dispatch → `ContextManager.append_assistant` / `append_tool_result`.
5. Each cycle emits one or more `RcaStep`s, which the server serializes to
   `SSEEvent`s and writes to the SSE stream (`event: step` / `event: delta`).
6. On conclusion the core builds an `RcaReport`; the server emits
   `event: report`, persists the report to MySQL, and emits `event: done`.

**Observability flow (always on when `RCA_OTEL_ENABLED=true`):**

```
app (rca_agent.observability) ──OTLP gRPC :4317──► otel-collector
                                                        ├─ traces  → Tempo        (:4318 internal)
                                                        ├─ metrics → ClickHouse rca_otel
                                                        └─ logs    → ClickHouse rca_otel
Grafana (:3000) reads Tempo (traces), ClickHouse (metrics/logs), MySQL (reports)
```

---

## 4. The DeepSeek thinking-mode + `reasoning_content` echo invariant

DeepSeek's `deepseek-reasoner` model emits a private chain-of-thought in a
separate `reasoning_content` field alongside the normal `content`. Thinking mode
is enabled per-request via an `extra_body` knob that lives **only** in the
`rca_agent.llm` implementation — never in the contract. Two DeepSeek-specific
constraints shape the design:

1. **Echo invariant.** For any assistant turn that produced `tool_calls`, the
   matching `reasoning_content` **MUST** be present in the messages sent on
   every subsequent request, or the API returns HTTP 400. Conversely, for turns
   with no tool calls, `reasoning_content` is ignored by the API and is stored
   only for display.
2. **No sampling params.** `temperature` / `top_p` are **not** set when thinking
   is enabled (a DeepSeek constraint); `LLMRequest` therefore omits them.

This invariant is enforced **entirely** inside the `ContextManager`
(`assemble_turn` and `compress`):

- `append_assistant` stores the raw turn (content, `reasoning_content`,
  `tool_calls`) in `ContextState.turns`.
- When assembling the messages array, every assistant message that carries
  `tool_calls` is emitted with its `reasoning_content`; the system prompt is
  always first.
- `compress` summarizes the oldest turns to fit `max_tokens` but **must**
  preserve the `reasoning_content` echo for any retained tool-bearing turn — it
  never drops reasoning from a turn whose `tool_calls` survive.

The agent loop never touches `reasoning_content` directly. This is the single
most fragile integration point in the system and is intentionally centralized.

---

## 5. Contract-based module boundaries

Every module programs against the frozen contracts in
[`rca_agent/contracts/`](../rca_agent/contracts) — Pydantic models and
`@runtime_checkable` Protocols that depend only on stdlib + pydantic. **No
contract imports an implementation module**, so implementations can be swapped
or tested in isolation.

| Contract (Protocol/model) | Owns | Consumers |
|---|---|---|
| `DataProvider` | observable-data query interface | tools, agent |
| `MemoryStore` | knowledge retrieval | tools, agent, server |
| `ContextManager` / `ContextState` | message window + reasoning echo | agent |
| `LLMClient` / `LLMRequest` / `LLMStreamDelta` | streaming chat | agent |
| `ToolSpec` / `ToolHandler` / `RegisteredTool` | tool registry + validation | tools, agent |
| `RcaStep` / `RcaReport` / `RootCause` / `RcaTrace` | investigation + report schema | agent, server, eval, frontend |
| `SSEEvent` / `SSEEventKind` / `SSEDelta` | SSE wire format | server, frontend |
| `Task` / `Topology` / `Case` | benchmark case shape | cases, providers, agent |

The OpenAI `tools=[...]` JSON schema and the runtime argument validation are
both derived from the **same** `args_model` (a Pydantic model) via
`build_openai_tools` / `validate_tool_call`, so the tool description the model
sees and the validation the runtime enforces can never drift.

Contracts are frozen: worker units propose changes to the coordinator rather
than editing `contracts/*` directly.

---

## 6. The SSE streaming contract

The server emits a stream of `SSEEvent` objects over `GET /rca/{case_id}/stream`.
The wire format (defined by `sse_format`) is:

```
event: <kind>\ndata: <json>\n\n
```

where `<kind>` ∈ `step | delta | report | error | done | ping` and `<json>` is
the full `SSEEvent` (`{event, case_id, data, seq}`) serialized with
`model_dump(mode="json")`.

| Event | `data` payload | Meaning |
|---|---|---|
| `step` | `RcaStep` | one investigation step (observe/hypothesize/investigate/tool_call/tool_result/reasoning/conclude/error) |
| `delta` | `SSEDelta` | fine-grained streaming token (`kind` = text\|reasoning\|tool_call); optional — the agent may emit only `step` events |
| `report` | `RcaReport` | the completed report (final root cause, evidence, confidence, actions) |
| `error` | `dict` (e.g. `{"error": "..."}`) | the run failed |
| `done` | `dict` / `{}` | terminal; no more events follow |
| `ping` | `dict` / `{}` | keepalive |

Each event carries a monotonically increasing `seq`. The frontend consumes the
identical shape (TS types generated from the JSON schema), guaranteeing the two
sides never drift. See [`API.md`](API.md) for concrete examples.

---

## 7. Module → file table

| Component | Module(s) | Key contract(s) |
|---|---|---|
| LLM RCA core | `rca_agent/agent` | `RcaStep`, `RcaReport`, `RootCause`, `StepKind` |
| Data provider | `rca_agent/providers` (parquet, clickhouse, loader) | `DataProvider`, filter + result models |
| Memory | `rca_agent/memory` + `memory/seed/` | `MemoryStore`, `MemoryItem`, `MemoryQuery` |
| Context manager | `rca_agent/context` | `ContextManager`, `ContextState`, `TurnRecord` |
| Tools | `rca_agent/tools` | `ToolSpec`, `ToolHandler`, `build_openai_tools`, `validate_tool_call` |
| LLM client | `rca_agent/llm` | `LLMClient`, `LLMRequest`, `LLMStreamDelta` |
| RCA server | `rca_agent/server` + `rca_agent/store` (MySQL) | `SSEEvent`, `sse_format`, `RcaReport` |
| Monitoring & eval | `rca_agent/observability` (OTel) + `rca_agent/eval` + Grafana | `RcaReport` (scoring) |
| Frontend | `frontend/` | `SSEEvent`, `RcaStep`, `RcaReport` (generated TS types) |
| Case loading (shared) | `rca_agent/cases` | `Case`, `Task`, `Topology` |
| Config | `rca_agent/config` | — (env-driven `Settings`) |
| Infra | `docker-compose.yml`, `infra/*` | — (ClickHouse, MySQL, OTel, Tempo, Grafana) |
