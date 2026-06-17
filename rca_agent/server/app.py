"""FastAPI server that runs the RCA agent and streams its trace over SSE.

Endpoints:
  GET  /health                         liveness
  GET  /cases                          list benchmark cases
  POST /rca/{case_id}                  start a run -> {stream_url}
  GET  /rca/{case_id}/stream           SSE stream of the agent trace
  GET  /reports/{case_id}              most recent stored report for a case
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query
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


class ReportStore(Protocol):
    """Structural type for the persistence backend used by the server.

    ``MysqlStore`` satisfies this protocol; tests may substitute any object
    exposing the same two methods.
    """

    def save_report(self, report: RcaReport, run_id: str | None = None) -> str: ...

    def list_reports(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[RcaReport]: ...


def _default_report_store() -> ReportStore:
    """Construct the production report store (MySQL, env-driven)."""
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


def set_agent_factory(fn: Callable[..., tuple[Case, Any]] | None) -> None:
    """Swap the agent factory (tests only). ``None`` restores the default."""
    global _agent_factory
    _agent_factory = fn or build_agent_for_case


def set_report_store_factory(fn: Callable[[], ReportStore] | None) -> None:
    """Swap the report store factory (tests only). ``None`` restores the default."""
    global _report_store_factory
    _report_store_factory = fn or _default_report_store


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_setup_otel()
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass
    yield


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
    return {
        "case_id": case_id,
        "backend": backend,
        "stream_url": f"/rca/{case_id}/stream?backend={backend}",
    }


def _sse(event: str, payload: dict | str, seq: int) -> dict:
    """Build an sse-starlette yield dict carrying a full SSEEvent-shaped payload."""
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False, default=str)
    return {"event": event, "data": payload, "retry": 0}


@app.get("/rca/{case_id}/stream")
async def stream_rca(
    case_id: str, backend: str = Query(default="parquet")
) -> EventSourceResponse:
    _validate_case_id(case_id)
    cases = list_cases_for_server()
    if case_id not in cases:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}")

    case, agent = _agent_factory(case_id, backend=backend)

    async def event_gen() -> AsyncIterator[dict]:
        seq = 0
        try:
            async for ev in agent.run(case):
                seq += 1
                if isinstance(ev, RcaReport):
                    # Persist (best-effort); never let storage break the stream.
                    try:
                        store = _report_store_factory()
                        store.save_report(ev)
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
                    return
                elif isinstance(ev, RcaStep):
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
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surface errors to the client, then end
            seq += 1
            logger.error(
                "rca_stream_error case_id=%s: %s: %s",
                case_id,
                type(e).__name__,
                e,
                extra={"case_id": case_id, "error": f"{type(e).__name__}: {e}"},
            )
            yield _sse(
                SSEEventKind.ERROR.value,
                {
                    "event": SSEEventKind.ERROR.value,
                    "case_id": case_id,
                    "data": {"error": f"{type(e).__name__}: {e}"},
                    "seq": seq,
                },
                seq,
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
