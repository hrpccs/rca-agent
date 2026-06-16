# Development

How to set up, run, test, and extend the RCA agent. The system is built as a
set of worker units around frozen contracts; this doc explains the worker/unit
layout and the conventions that keep the units composable.

---

## 1. Prerequisites

- **Python 3.11**, managed via [`uv`](https://docs.astral.sh/uv/) (the only
  Python toolchain used). Install uv first if you don't have it.
- **Docker** + Docker Compose (for the infra stack: ClickHouse, MySQL,
  OTel-collector, Tempo, Grafana).
- **Node.js** (only if you work on `frontend/`).
- A **DeepSeek API key** for live runs (optional for unit tests; get one at
  <https://platform.deepseek.com>).

## 2. Setup

```bash
make dev        # runs `uv sync` and copies .env.example → .env if missing
```

Then edit `.env` and set at minimum:

```
RCA_DEEPSEEK_API_KEY=sk-...        # real key (starts sk-); leave as sk-x... to run offline/mocked
RCA_CASES_DIR=/path/to/rca100/cases  # benchmark case root (parquet backend reads this)
```

> **`RCA_CASES_DIR` has a machine-specific default.** `rca_agent/config.py`
> ships a developer-local default path that will not exist on other machines;
> the parquet backend silently finds zero cases if the path is absent. Always
> set `RCA_CASES_DIR` explicitly on a new machine (or use
> `RCA_DATA_BACKEND=clickhouse` in prod, where it is unused).

All other knobs have sensible defaults (see [`rca_agent/config.py`](../rca_agent/config.py)
and [`.env.example`](../.env.example)). Every setting is namespaced `RCA_*`.

Bring up the infra stack when you need real databases:

```bash
make up          # docker compose up -d clickhouse mysql otel-collector tempo grafana
make ps          # status
make down        # stop (keeps volumes); docker compose down -v to wipe data
```

Connection details (host/port/user/password) for every service are in
[`docs/INFRA_ACCESS.md`](INFRA_ACCESS.md).

## 3. Running

```bash
make run CASE=t001     # run RCA on one benchmark case (CLI, parquet backend)
make serve             # start the FastAPI server (reload) on :8000
make frontend          # frontend dev server on :5173
curl -N http://localhost:8000/rca/t001/stream   # watch the SSE stream
```

> **Implementation status.** `make run`, `make serve`, and `make frontend`
> drive modules owned by sibling units (`rca_agent.cli`, `rca_agent.server.app`,
> `frontend/`). In the current tree some of these may not yet exist; the targets
> succeed once those units land. The contracts and infra are independent of them
> and can be exercised via `make test` / `make test-contracts` today.

## 4. Tests

```bash
make test              # full suite, excludes live (default for CI/units)
make test-contracts    # contract conformance gate only (tests/contracts)
make test-live         # incl. real (billed) DeepSeek calls + live DBs (needs key + infra)
```

- **`make test`** = `uv run pytest -q -m "not live"`. Use this for normal
  unit work; it requires no API key and no infra.
- **`make test-contracts`** = `uv run pytest -q tests/contracts`. Validates the
  frozen contracts themselves (Protocol runtime-checkability, filter defaults,
  tool schema/validate round-trip, SSE wire-format stability). Must pass on the
  foundation before any worker branch.
- **`make test-live`** runs the full suite including `@pytest.mark.live` tests.
  These are auto-skipped unless `RCA_DEEPSEEK_API_KEY` is set to a real key (the
  conftest skips anything starting with `sk-x`). They make real, billed API
  calls and may hit live databases.

Lint/format:

```bash
make lint              # ruff check
make format            # ruff format
```

## 5. Worker / unit layout

The codebase is decomposed into worker units, each owning a slice of
functionality behind the frozen contracts in
[`rca_agent/contracts/`](../rca_agent/contracts). Units are assigned to
isolated git worktrees and integrated by a coordinator. The current unit map:

| Unit | Owns | Boundary contract(s) |
|---|---|---|
| LLM client | `rca_agent/llm` | `LLMClient`, `LLMRequest`, `LLMStreamDelta` |
| Data provider | `rca_agent/providers` (parquet, clickhouse, loader) | `DataProvider` + filter/result models |
| Memory | `rca_agent/memory` + `memory/seed/` | `MemoryStore`, `MemoryItem` |
| Context manager | `rca_agent/context` | `ContextManager`, `ContextState` |
| Tools | `rca_agent/tools` | `ToolSpec`, `ToolHandler`, registry |
| RCA core | `rca_agent/agent` | `RcaStep`, `RcaReport`, `StepKind` |
| RCA server | `rca_agent/server` + `rca_agent/store` | `SSEEvent`, `sse_format` |
| Observability + eval | `rca_agent/observability`, `rca_agent/eval` | `RcaReport` |
| Frontend | `frontend/` | `SSEEvent`, `RcaStep`, `RcaReport` |
| Docs + memory seed + dashboards | `docs/`, `memory/seed/`, `infra/grafana/dashboards/` | (this unit) |

Each unit programs **only** against the contracts; it must not import another
unit's internals.

## 6. Conventions

- **Program against contracts.** Depend on the Protocols/models in
  `rca_agent.contracts`, never on a concrete implementation. This keeps units
  independently testable and swappable.
- **Do not edit the frozen seams.** `rca_agent/contracts/*`, any `__init__.py`,
  `pyproject.toml`, `uv.lock`, and `docker-compose.yml` are off-limits in worker
  PRs. Propose contract changes to the coordinator. You MAY add new files only.
- **DeepSeek specifics stay in `rca_agent/llm`.** The `extra_body` thinking knob,
  `reasoning_effort`, base URL, and the `reasoning_content` echo logic all live
  in the implementation, never in a contract. The `ContextManager` owns the echo
  invariant; no other module touches `reasoning_content`.
- **No new dependencies beyond what's in `pyproject.toml`.** The stdlib +
  existing deps suffice. (This doc/dashboards unit uses none beyond markdown/json.)
- **Env-driven config.** All runtime knobs are `RCA_*` env vars read via
  `rca_agent.config.settings`. Live features gate on `settings.has_llm_key`
  (a real key, i.e. not empty and not `sk-x...`).

## 7. Adding a new data-provider backend

The `DataProvider` Protocol is the single integration seam for observable data.
To add a backend (e.g. Prometheus, Datadog, a new object store):

1. Implement the seven methods (`query_metrics`, `query_logs`, `query_traces`,
   `query_events`, `query_alerts`, `query_topology`, `modalities`). Each takes
   the matching filter model (`MetricFilter`, `LogFilter`, …) and returns the
   matching result models (`MetricSeries`, `LogLine`, …) — **never raw rows**.
   Your backend must produce objects byte-for-byte compatible with the parquet
   backend so the tools don't branch.
2. Construct the provider per-case (it carries `case_id` and knows its own
   scope — case dir, database, or remote workspace).
3. Wire it into the provider factory keyed by a new `RCA_DATA_BACKEND` value.
4. Test it with the shared filter fixtures in `tests/conftest.py`; assert its
   outputs round-trip through the result models. The tools and agent core need
   no changes.

## 8. Adding a new memory backend

`MemoryStore` is a `@runtime_checkable` Protocol with `index`, `retrieve`,
`retrieve_for_context`, and `clear`. To add a backend (e.g. a vector DB):

1. Implement the four methods. Retrieval may be keyword, TF-IDF, or embedding
   based — the contract is method signatures + return types, not algorithm.
2. Preserve the `case_id == "__global__"` convention for cross-case/domain
   knowledge, and honor `MemoryQuery.kind` / `entities` / `top_k` filters.
3. Index the markdown seed files under [`memory/seed/`](../memory/seed) at
   startup so runbooks/SOPs/domain facts are retrievable out of the box. (The
   default `inmemory` backend is owned by the memory unit; seed indexing happens
   once that unit is implemented — until then `retrieve()` returns nothing.)
4. Wire it into the memory factory keyed by a new `RCA_MEMORY_BACKEND` value.

Seed files are markdown with optional front-matter `kind:` (`runbook` | `sop` |
`domain_fact`; defaults to `domain_fact`). `case_id` is global for all seed
knowledge.
