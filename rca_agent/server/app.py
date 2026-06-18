"""FastAPI server that runs the RCA agent and streams its trace over SSE.

Endpoints:
  GET  /health                         liveness
  GET  /cases                          list benchmark cases
  POST /rca/{case_id}                  start a run -> {stream_url, run_id}
  GET  /rca/{case_id}/stream           SSE stream of the agent trace
  GET  /reports/{case_id}              most recent stored report for a case
  GET  /runs                           list runs (optional case_id filter)
  GET  /runs/{run_id}                  run summary + persisted steps
  GET  /runs/{run_id}/steps            persisted steps for a run
  GET  /cases/{case_id}/runs           runs for a case
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from ..agent import build_agent_for_case
from ..cases import list_cases
from ..contracts import Case, RcaReport, RcaStep, SSEEventKind

logger = logging.getLogger("rca_agent.server")

# The dev machine's shell exports a SOCKS proxy that breaks the openai/httpx
# client; clear it so the server can build the DeepSeek client. No-op in prod.
for _v in (
    "all_proxy",
    "ALL_PROXY",
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "socks_proxy",
    "SOCKS_PROXY",
):
    os.environ.pop(_v, None)

# case_id must be a safe single-segment identifier: an alphanumeric lead char
# followed by letters/digits/underscore/dash/dot. The char class excludes both
# path separators (``/`` and ``\``), so a valid id can never escape its parent
# directory when joined as ``root / case_id`` — this is what guards every
# endpoint that fans ``case_id`` out to the filesystem / agent / report store
# against path traversal and injection.
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

# run_id is a 32-char hex (uuid4().hex as minted by MysqlStore.start_run).
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _validate_case_id(case_id: str) -> None:
    """Reject case_ids that are not safe single-segment identifiers.

    Raises ``HTTPException(400)`` so bad ids never reach the filesystem, the
    agent factory, or the report store. The regex alone is sufficient: it
    requires a non-empty alnum lead char (so empty / leading-dot / bare ``..``
    are rejected) and its char class excludes ``/`` and ``\\`` (so path
    separators — the actual traversal vector — cannot appear).
    """
    if not _CASE_ID_RE.match(case_id):
        raise HTTPException(status_code=400, detail="invalid case_id")


def _validate_run_id(run_id: str) -> None:
    """Reject run_ids that are not 32-char lowercase hex.

    Raises ``HTTPException(400)`` so a malformed run_id never reaches the trace
    store. run_id is minted server-side as ``uuid.uuid4().hex``; only the
    ``^[0-9a-f]{32}$`` shape is accepted.
    """
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="invalid run_id")


class ReportStore(Protocol):
    """Structural type for the persistence backend used by the server.

    ``MysqlStore`` satisfies this protocol; tests may substitute any object
    exposing the same two methods.
    """

    def save_report(self, report: RcaReport, run_id: str | None = None) -> str: ...

    def list_reports(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[RcaReport]: ...


class TraceStore(Protocol):
    """Structural type for the incremental trace-persistence backend.

    Declares the methods the server calls to record a run, persist each step as
    it streams, close the run with a terminal status, and read runs/steps back
    for the REST endpoints. ``MysqlStore`` structurally satisfies this Protocol
    once its ``append_step``/``list_steps``/``list_runs``/``get_run`` methods
    land (sibling unit T1); until then the default factory constructs
    ``MysqlStore`` lazily and tests inject a fake. The Protocol keeps this unit
    independently mergeable without a hard import dependency on T1.
    """

    def start_run(self, case_id: str, model: str) -> str: ...

    def finish_run(
        self,
        run_id: str,
        status: str,
        token_usage: dict[str, Any] | None = None,
    ) -> None: ...

    def append_step(
        self, run_id: str, case_id: str, seq: int, step: RcaStep
    ) -> None: ...

    def list_runs(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]: ...

    def list_steps(self, run_id: str, limit: int = 20000) -> list[RcaStep]: ...

    def get_run(self, run_id: str) -> dict[str, Any] | None: ...


def _default_report_store() -> ReportStore:
    """Construct the production report store (MySQL, env-driven)."""
    from ..store.mysql_store import MysqlStore

    return MysqlStore()


def _default_trace_store() -> TraceStore:
    """Construct the production trace store (MySQL, env-driven).

    Imported lazily — mirroring ``_default_report_store`` — so the server module
    imports cleanly even when MySQL/SQLAlchemy are not configured (e.g. unit
    tests with an injected fake).
    """
    from ..store.mysql_store import MysqlStore

    return MysqlStore()


# --------------------------------------------------------------------------- #
# Injectable seams — prod defaults are wired in below; tests swap these.
# --------------------------------------------------------------------------- #
# A case discovery callable: () -> list[str]. Defaults to the on-disk benchmark
# scanner (env-configured via RCA_CASES_DIR).
list_cases_for_server: Callable[[], list[str]] = list_cases

# Agent factory: (case_id, backend) -> (Case, agent). Defaults to the real
# production builder; tests inject a fake that needs no LLM/DB/data. The agent
# is typed as ``Any`` so tests can substitute a duck-typed object whose ``run``
# is an async generator yielding RcaStep/RcaReport.
_agent_factory: Callable[..., tuple[Case, Any]] = build_agent_for_case

# Report store factory: () -> ReportStore. Defaults to the MySQL store; tests
# inject an in-memory stand-in.
_report_store_factory: Callable[[], ReportStore] = _default_report_store

# Trace store factory: () -> TraceStore. Defaults to the MySQL store; tests
# inject an in-memory stand-in.
_trace_store_factory: Callable[[], TraceStore] = _default_trace_store


def set_agent_factory(fn: Callable[..., tuple[Case, Any]] | None) -> None:
    """Swap the agent factory (tests only). ``None`` restores the default."""
    global _agent_factory
    _agent_factory = fn or build_agent_for_case


def set_report_store_factory(fn: Callable[[], ReportStore] | None) -> None:
    """Swap the report store factory (tests only). ``None`` restores the default."""
    global _report_store_factory
    _report_store_factory = fn or _default_report_store


def set_trace_store_factory(fn: Callable[[], TraceStore] | None) -> None:
    """Swap the trace store factory (tests only). ``None`` restores the default."""
    global _trace_store_factory
    _trace_store_factory = fn or _default_trace_store


def set_case_lister(fn: Callable[[], list[str]] | None) -> None:
    """Swap the case-discovery callable (tests only). ``None`` restores default."""
    global list_cases_for_server
    list_cases_for_server = fn or list_cases


def _try_setup_otel() -> None:
    try:
        from ..config import get_settings
        from ..observability.tracing import setup_otel

        s = get_settings()
        if s.otel_enabled:
            setup_otel(endpoint=s.otel_endpoint, service_name=s.otel_service_name)
    except Exception:
        pass


def _parse_started_at(value: Any) -> datetime | None:
    """Coerce a run row's ``started_at`` to an aware UTC ``datetime``.

    ``MysqlStore`` and the in-memory fake may emit either a timezone-aware
    ``datetime`` or an ISO-8601 string (possibly with a trailing ``Z``). We
    accept both, treat naive datetimes as UTC, and return ``None`` for anything
    we cannot interpret so the reaper can skip the row defensively rather than
    crash the app.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # ``fromisoformat`` in 3.11 accepts most ISO shapes, but not a bare
        # trailing ``Z``; normalize it to ``+00:00`` so an ``...Z`` row is
        # parsed as UTC rather than rejected.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        # A naive datetime is ambiguous; assume the server's storage tz is UTC
        # (the schema stores UTC). Stamping UTC is safer than leaving naive,
        # which would crash arithmetic against ``datetime.now(UTC)``.
        dt = dt.replace(tzinfo=UTC)
    return dt


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to ``default`` on missing/bad values.

    A misconfigured value (e.g. ``RCA_RUN_REAP_MIN=10min``) must NOT kill the
    reaper loop — parse once per use, warn on bad input, and keep running with
    the default so the backstop stays armed.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "env %s=%r is not a float; using default %s", name, raw, default,
            extra={"env": name, "value": raw},
        )
        return default
    return val if val > 0 else default


def _reap_orphan_runs(store: TraceStore) -> int:
    """Close runs stuck in ``running`` past the reap age (backstop).

    This is a safety net, NOT the primary cleanup path. The stream's ``finally``
    block is supposed to close every run when the client disconnects (now using
    an explicit ``request.is_disconnected()`` check). But sse-starlette does not
    guarantee generator cancellation on disconnect, and the process can die
    mid-run — so a row can be left ``running`` with steps already streamed and
    then dropped. This reaper walks recent runs and marks the stale ``running``
    ones as ``interrupted`` (NOT ``truncated``: ``truncated`` is reserved for the
    agent's own step-cap force-conclude, so ops can distinguish "the agent
    bailed at its step limit" from "the stream was abandoned").

    Uses ONLY existing TraceStore read/close methods (no schema or store change).
    Best-effort: each row is wrapped so a single bad row or a transient store
    error never aborts the sweep or crashes the app. Returns the count closed.
    """
    max_age_min = _env_float("RCA_RUN_REAP_MIN", 10.0)
    cutoff = datetime.now(UTC).timestamp() - max_age_min * 60
    closed = 0
    try:
        # Newest-first store ordering means a very old orphan beyond this window
        # could be missed on a deployment with many historical runs; the limit is
        # intentionally generous. The per-request ``finally`` is the primary
        # cleanup; this is just the backstop.
        runs = store.list_runs(limit=2000)
    except Exception as exc:
        logger.warning(
            "rca_reap_list_failed: %s: %s",
            type(exc).__name__,
            exc,
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return 0
    for row in runs:
        try:
            if not isinstance(row, dict):
                continue
            if row.get("status") != "running":
                continue
            run_id = row.get("run_id")
            if not run_id:
                continue
            started = _parse_started_at(row.get("started_at"))
            # ``started_at`` may be absent on a partial/legacy schema; if we
            # can't tell the age, do NOT reap (avoids closing a fresh run that
            # just hasn't had its timestamp filled in yet).
            if started is None:
                continue
            if started.timestamp() > cutoff:
                continue  # fresh — leave it alone
            store.finish_run(run_id, "interrupted")
            closed += 1
            logger.info(
                "rca_reap_closed run_id=%s: abandoned run older than %s min marked "
                "'interrupted'",
                run_id,
                max_age_min,
                extra={"run_id": run_id, "status": "interrupted"},
            )
        except Exception as exc:
            # One bad row must not stop the sweep. Log and continue.
            logger.warning(
                "rca_reap_row_failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            continue
    if closed:
        logger.info("rca_reap_sweep closed %d abandoned run(s)", closed)
    return closed


async def _reaper_loop(store: TraceStore) -> None:
    """Periodically sweep orphaned ``running`` runs.

    Loops forever until cancelled by the lifespan shutdown. Each iteration is
    fully wrapped so a transient failure in one sweep never kills the loop
    (which would silently disable the backstop for the rest of the process
    lifetime).
    """
    interval = _env_float("RCA_RUN_REAP_INTERVAL_SEC", 300.0)
    while True:
        try:
            _reap_orphan_runs(store)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "rca_reaper_iteration_failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_setup_otel()
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass

    # Orphan-run reaper: on startup, sweep once to clean up any runs left
    # ``running`` by a previous (crashed/restarted) process, then schedule a
    # periodic background sweep as a backstop for future stream-disconnect
    # leaks that slip past the per-request ``finally`` cleanup. Wrapped so a
    # store misconfiguration (e.g. no DB in dev) never prevents startup.
    reaper_task: asyncio.Task[None] | None = None
    try:
        reaper_store = _trace_store_factory()
        # Synchronous startup sweep — clears the slate before serving traffic.
        _reap_orphan_runs(reaper_store)
        reaper_task = asyncio.create_task(_reaper_loop(reaper_store))
    except Exception as exc:
        logger.warning(
            "rca_reaper_init_failed (non-fatal): %s: %s",
            type(exc).__name__,
            exc,
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )

    try:
        yield
    finally:
        # Cancel the periodic reaper so shutdown is prompt and the task doesn't
        # outlive the app. ``suppress(BaseException)`` swallows CancelledError.
        if reaper_task is not None and not reaper_task.done():
            reaper_task.cancel()
            with suppress(BaseException):
                await reaper_task


app = FastAPI(title="RCA Agent", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/cases")
def get_cases() -> dict:
    return {"cases": list_cases_for_server()}


@app.post("/rca/{case_id}")
def start_rca(case_id: str, backend: str = Query(default="parquet")) -> dict:
    _validate_case_id(case_id)
    cases = list_cases_for_server()
    if case_id not in cases:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}")

    # Best-effort: mint a run row so a dropped stream leaves a durable partial
    # trace. Storage failure is non-fatal — the stream can still mint its own
    # run_id on demand, or proceed without one.
    run_id: str | None = None
    try:
        from ..config import get_settings

        model = get_settings().deepseek_model
        trace = _trace_store_factory()
        run_id = trace.start_run(case_id, model)
    except Exception as exc:
        logger.warning(
            "rca_start_run_failed case_id=%s: %s: %s",
            case_id,
            type(exc).__name__,
            exc,
            extra={
                "case_id": case_id,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        run_id = None

    stream_url = f"/rca/{case_id}/stream?backend={backend}"
    if run_id is not None:
        stream_url += f"&run_id={run_id}"
    return {
        "case_id": case_id,
        "backend": backend,
        "run_id": run_id,
        "stream_url": stream_url,
    }


def _sse(event: str, payload: dict | str, seq: int) -> dict:
    """Build an sse-starlette yield dict carrying a full SSEEvent-shaped payload."""
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False, default=str)
    return {"event": event, "data": payload, "retry": 0}


@app.get("/rca/{case_id}/stream")
async def stream_rca(
    request: Request,
    case_id: str,
    backend: str = Query(default="parquet"),
    run_id: str | None = Query(default=None),
) -> EventSourceResponse:
    _validate_case_id(case_id)
    if run_id is not None:
        _validate_run_id(run_id)
    cases = list_cases_for_server()
    if case_id not in cases:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}")

    case, agent = _agent_factory(case_id, backend=backend)

    # Keepalive: the client closes a stream that receives nothing for its idle
    # timeout. A long DeepSeek reasoning turn can exceed that window between
    # steps, so we emit a heartbeat every few seconds of silence to prove the
    # connection is alive.
    #
    # The heartbeat MUST be a NAMED ``event: ping`` (not a data-only message).
    # The frontend listens via ``eventSource.addEventListener("ping", ...)`` and
    # does NOT register an ``onmessage`` handler, so a data-only ping would be
    # silently dropped by EventSource and never re-arm the client's idle
    # watchdog — which is exactly the live bug: during a >60s DeepSeek turn the
    # browser aborted the stream ("断链") because the unnamed pings never
    # arrived at the listener. sse-starlette emits a named event only when the
    # yield dict carries an ``"event"`` key, so we set it here. Tunable via env.
    heartbeat_interval = float(os.environ.get("RCA_SSE_HEARTBEAT_SEC", "15"))

    async def event_gen() -> AsyncIterator[dict]:
        seq = 0
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel: Any = object()
        # Track whether the run has already been closed by a terminal branch
        # (REPORT -> finish_run(status) or ERROR -> finish_run("error")) so the
        # finally block can close an ABANDONED run (clean producer end with no
        # report, or client disconnect) exactly once and not double-close.
        run_closed = False

        # Resolve the trace store and the effective run_id once, up front. If
        # no run_id was passed (e.g. the client opened the stream directly),
        # best-effort mint one so each step can still be persisted
        # incrementally. Storage failure here is non-fatal: we proceed with
        # run_id=None and simply skip persistence calls below.
        #
        # ``effective_run_id`` shadows the closure-captured ``run_id`` query
        # parameter; we copy it into a local so the ``mint-if-absent`` branch
        # below can rebind it without tripping the closure-local scoping rule.
        effective_run_id: str | None = run_id
        trace: TraceStore | None = None
        try:
            trace = _trace_store_factory()
        except Exception as exc:
            logger.warning(
                "rca_trace_store_unavailable case_id=%s: %s: %s",
                case_id,
                type(exc).__name__,
                exc,
                extra={
                    "case_id": case_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            trace = None
        if trace is not None and effective_run_id is None:
            try:
                from ..config import get_settings

                model = get_settings().deepseek_model
                effective_run_id = trace.start_run(case_id, model)
            except Exception as exc:
                logger.warning(
                    "rca_start_run_in_stream_failed case_id=%s: %s: %s",
                    case_id,
                    type(exc).__name__,
                    exc,
                    extra={
                        "case_id": case_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                effective_run_id = None

        async def produce() -> None:
            """Drain the agent into the queue so the consumer can interleave
            heartbeats without cancelling an in-flight agent step (which
            asyncio.wait_for on agent.__anext__ would do).

            Note on cancellation: this task is cancelled on client disconnect
            (see the ``finally`` below). ``async for`` raises CancelledError
            into the agent; CancelledError is a BaseException, so the
            ``except Exception`` here does NOT swallow it — it propagates to the
            task and is consumed by ``with suppress(BaseException): await task``
            in the finally. That is the desired behavior: the in-flight agent
            run is actually torn down and does not outlive the request.
            """
            try:
                async for ev in agent.run(case):
                    await queue.put(ev)
            except Exception as e:  # surface to the consumer as an error event
                await queue.put(e)
            finally:
                await queue.put(sentinel)

        task = asyncio.create_task(produce())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=heartbeat_interval
                    )
                except TimeoutError:
                    # Heartbeat fired: the producer has been silent for
                    # ``heartbeat_interval``. Before re-arming the watchdog
                    # with a ping, check whether the client has actually gone
                    # away. sse-starlette does not reliably cancel this
                    # generator on disconnect, so without this explicit check
                    # the stream (and the produce() task, and the run row)
                    # would leak until the producer finished naturally — which
                    # for a stuck DeepSeek turn may be never. Breaking here
                    # forces the ``finally`` to run and tear down
                    # deterministically. ``is_disconnected()`` is best-effort;
                    # the finally is the backstop either way.
                    if await request.is_disconnected():
                        logger.info(
                            "rca_stream_client_disconnect case_id=%s run_id=%s: "
                            "client gone during heartbeat, closing stream",
                            case_id,
                            effective_run_id,
                            extra={
                                "case_id": case_id,
                                "run_id": effective_run_id,
                            },
                        )
                        break
                    seq += 1
                    # NAMED ping: ``"event": "ping"`` MUST be present so the
                    # frontend's ``addEventListener("ping")`` fires. The data
                    # payload is an SSEEvent-shaped envelope (event/case_id/
                    # data/seq) so a generic SSEEvent listener also works.
                    yield {
                        "event": SSEEventKind.PING.value,
                        "data": json.dumps(
                            {
                                "event": SSEEventKind.PING.value,
                                "case_id": case_id,
                                "data": {},
                                "seq": seq,
                            },
                            ensure_ascii=False,
                        ),
                        "retry": 0,
                    }
                    continue
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    seq += 1
                    logger.error(
                        "rca_stream_error case_id=%s: %s: %s",
                        case_id,
                        type(item).__name__,
                        item,
                        extra={
                            "case_id": case_id,
                            "error": f"{type(item).__name__}: {item}",
                        },
                    )
                    # Best-effort: close the run as errored. Never let storage
                    # break the stream. ``run_closed`` is set regardless of
                    # whether finish_run succeeded: a failed close attempt must
                    # NOT trigger a retry with a different status, which would
                    # overwrite the real terminal condition.
                    if effective_run_id is not None and trace is not None:
                        run_closed = True
                        try:
                            trace.finish_run(effective_run_id, "error")
                            logger.info(
                                "rca_run_closed run_id=%s status=error "
                                "(producer exception)",
                                effective_run_id,
                                extra={
                                    "run_id": effective_run_id,
                                    "status": "error",
                                },
                            )
                        except Exception as exc:
                            logger.warning(
                                "rca_finish_run_error_failed case_id=%s "
                                "run_id=%s: %s: %s",
                                case_id,
                                effective_run_id,
                                type(exc).__name__,
                                exc,
                                extra={
                                    "case_id": case_id,
                                    "run_id": effective_run_id,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                            )
                    yield _sse(
                        SSEEventKind.ERROR.value,
                        {
                            "event": SSEEventKind.ERROR.value,
                            "case_id": case_id,
                            "data": {"error": f"{type(item).__name__}: {item}"},
                            "seq": seq,
                        },
                        seq,
                    )
                    break
                ev = item
                seq += 1
                if isinstance(ev, RcaReport):
                    # Persist (best-effort); never let storage break the stream.
                    # Thread the run_id so the report row is linked to the
                    # incremental trace run (save_report accepts run_id=None).
                    try:
                        store = _report_store_factory()
                        store.save_report(ev, effective_run_id)
                    except Exception as exc:
                        # Log with the detail in the MESSAGE (not only in
                        # ``extra``) so it is visible under the default stdlib
                        # formatter, which discards unreferenced extra fields.
                        logger.error(
                            "rca_report_persist_failed case_id=%s: %s: %s",
                            case_id,
                            type(exc).__name__,
                            exc,
                            extra={
                                "case_id": case_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                    # Best-effort: close the run with the report's terminal
                    # status (completed | error | truncated) and token usage.
                    # ``run_closed`` is set regardless of whether finish_run
                    # succeeded (see the error branch above for rationale).
                    if effective_run_id is not None and trace is not None:
                        run_closed = True
                        try:
                            trace.finish_run(
                                effective_run_id, ev.status, ev.token_usage
                            )
                            logger.info(
                                "rca_run_closed run_id=%s status=%s (report)",
                                effective_run_id,
                                ev.status,
                                extra={
                                    "run_id": effective_run_id,
                                    "status": ev.status,
                                },
                            )
                        except Exception as exc:
                            logger.warning(
                                "rca_finish_run_failed case_id=%s run_id=%s "
                                "status=%s: %s: %s",
                                case_id,
                                effective_run_id,
                                ev.status,
                                type(exc).__name__,
                                exc,
                                extra={
                                    "case_id": case_id,
                                    "run_id": effective_run_id,
                                    "status": ev.status,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                            )
                    yield _sse(
                        SSEEventKind.REPORT.value,
                        {
                            "event": SSEEventKind.REPORT.value,
                            "case_id": case_id,
                            "data": ev.model_dump(mode="json"),
                            "seq": seq,
                        },
                        seq,
                    )
                    seq += 1
                    yield _sse(
                        SSEEventKind.DONE.value,
                        {
                            "event": SSEEventKind.DONE.value,
                            "case_id": case_id,
                            "data": {"status": ev.status},
                            "seq": seq,
                        },
                        seq,
                    )
                    # A report is terminal: stop consuming the agent so a
                    # misbehaving/injected agent that keeps yielding after its
                    # report cannot produce duplicate REPORT+DONE events or
                    # hold the SSE connection open.
                    break
                elif isinstance(ev, RcaStep):
                    # Incremental persistence: record this step so a dropped
                    # stream leaves a durable partial trace. Best-effort —
                    # never let storage break the stream.
                    if effective_run_id is not None and trace is not None:
                        try:
                            trace.append_step(
                                effective_run_id, case_id, seq, ev
                            )
                        except Exception as exc:
                            logger.warning(
                                "rca_append_step_failed case_id=%s run_id=%s "
                                "seq=%s: %s: %s",
                                case_id,
                                effective_run_id,
                                seq,
                                type(exc).__name__,
                                exc,
                                extra={
                                    "case_id": case_id,
                                    "run_id": effective_run_id,
                                    "seq": seq,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                            )
                    yield _sse(
                        SSEEventKind.STEP.value,
                        {
                            "event": SSEEventKind.STEP.value,
                            "case_id": case_id,
                            "data": ev.model_dump(mode="json"),
                            "seq": seq,
                        },
                        seq,
                    )
        finally:
            if not task.done():
                task.cancel()
            # Await the (possibly just-cancelled) producer task to consume its
            # CancelledError/exception so it never propagates out of the SSE
            # generator or surfaces as an unawaited-coroutine warning. This
            # also ensures the in-flight agent run does not outlive the request
            # — ``produce()`` observes the cancellation (CancelledError is a
            # BaseException, so its ``except Exception`` does not swallow it)
            # and its ``async for`` tears the agent down.
            with suppress(BaseException):
                await task
            # Close an ABANDONED run: if the stream ended without a terminal
            # event (client disconnect / GeneratorExit, or a producer that
            # stopped cleanly without yielding an RcaReport), the run row would
            # otherwise linger in ``running`` forever. Mark it ``interrupted``
            # (NOT ``truncated``): ``truncated`` is reserved for the agent's
            # own step-cap force-conclude, so ops can distinguish "the agent
            # bailed at its step limit" from "the stream was abandoned". The
            # ``run_closed`` guard ensures we never double-close a run that
            # already received its real status (completed | error). Best-effort;
            # storage failure here is non-fatal (the stream is already ending).
            if (
                not run_closed
                and effective_run_id is not None
                and trace is not None
            ):
                try:
                    trace.finish_run(effective_run_id, "interrupted")
                    logger.info(
                        "rca_run_closed run_id=%s status=interrupted "
                        "(stream abandoned / client disconnect)",
                        effective_run_id,
                        extra={
                            "run_id": effective_run_id,
                            "status": "interrupted",
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "rca_finish_run_interrupted_failed case_id=%s "
                        "run_id=%s: %s: %s",
                        case_id,
                        effective_run_id,
                        type(exc).__name__,
                        exc,
                        extra={
                            "case_id": case_id,
                            "run_id": effective_run_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

    return EventSourceResponse(event_gen())


@app.get("/reports/{case_id}")
def get_report(case_id: str) -> dict:
    _validate_case_id(case_id)
    try:
        store = _report_store_factory()
        reports = store.list_reports(case_id=case_id, limit=1)
    except Exception as e:
        # Surface as 503 so the client can distinguish storage-unavailable from
        # genuinely-missing; log the underlying failure for ops.
        logger.error(
            "rca_report_list_failed case_id=%s: %s: %s",
            case_id,
            type(e).__name__,
            e,
            extra={"case_id": case_id, "error": f"{type(e).__name__}: {e}"},
        )
        raise HTTPException(
            status_code=503, detail=f"storage unavailable: {e}"
        ) from e
    if not reports:
        raise HTTPException(status_code=404, detail="no report for case")
    return reports[0].model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Run + trace REST endpoints
# --------------------------------------------------------------------------- #
# These read the incremental trace persisted by the stream. Every store call is
# wrapped so a storage failure surfaces as a clean 503 (never a 500 with a
# stack trace) — matching the ``/reports/{case_id}`` contract.


def _trace_store_or_503() -> TraceStore:
    """Resolve the trace store or raise HTTPException(503).

    Centralizes the try/except so each run/trace handler stays readable.
    """
    try:
        return _trace_store_factory()
    except Exception as e:
        logger.error(
            "rca_trace_store_unavailable: %s: %s",
            type(e).__name__,
            e,
            extra={"error": f"{type(e).__name__}: {e}"},
        )
        raise HTTPException(
            status_code=503, detail=f"storage unavailable: {e}"
        ) from e


def _run_store_error_503(ctx: str, e: Exception) -> HTTPException:
    """Build a 503 HTTPException for a trace-store read failure and log it.

    ``ctx`` is an opaque identifier for the failing request scope (a run_id or
    case_id, depending on the endpoint) used only for log correlation.
    """
    logger.error(
        "rca_run_store_failed ctx=%s: %s: %s",
        ctx,
        type(e).__name__,
        e,
        extra={
            "ctx": ctx,
            "error": f"{type(e).__name__}: {e}",
        },
    )
    return HTTPException(status_code=503, detail=f"storage unavailable: {e}")


@app.get("/runs")
def list_runs(
    case_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=1000),
) -> dict:
    if case_id is not None:
        _validate_case_id(case_id)
    store = _trace_store_or_503()
    try:
        runs = store.list_runs(case_id=case_id, limit=limit)
    except Exception as e:
        raise _run_store_error_503(case_id or "all", e) from e
    return {"runs": runs}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    _validate_run_id(run_id)
    store = _trace_store_or_503()
    try:
        summary = store.get_run(run_id)
        steps = store.list_steps(run_id)
    except Exception as e:
        raise _run_store_error_503(run_id, e) from e
    if summary is None:
        raise HTTPException(status_code=404, detail="unknown run")
    return {"run": summary, "steps": steps}


@app.get("/runs/{run_id}/steps")
def list_run_steps(run_id: str) -> dict:
    _validate_run_id(run_id)
    store = _trace_store_or_503()
    try:
        steps = store.list_steps(run_id)
    except Exception as e:
        raise _run_store_error_503(run_id, e) from e
    return {"steps": steps}


@app.get("/cases/{case_id}/runs")
def list_case_runs(
    case_id: str, limit: int = Query(default=50, ge=1, le=1000)
) -> dict:
    _validate_case_id(case_id)
    store = _trace_store_or_503()
    try:
        runs = store.list_runs(case_id=case_id, limit=limit)
    except Exception as e:
        raise _run_store_error_503(case_id, e) from e
    return {"runs": runs}
