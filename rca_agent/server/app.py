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
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from ..agent import build_agent_for_case
from ..cases import list_cases
from ..contracts import RcaReport, RcaStep, SSEEventKind

# The dev machine's shell exports a SOCKS proxy that breaks the openai/httpx
# client; clear it so the server can build the DeepSeek client. No-op in prod.
for _v in ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
           "https_proxy", "HTTPS_PROXY", "socks_proxy", "SOCKS_PROXY"):
    os.environ.pop(_v, None)


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
    return {"cases": list_cases()}


@app.post("/rca/{case_id}")
def start_rca(case_id: str, backend: str = Query(default="parquet")) -> dict:
    cases = list_cases()
    if case_id not in cases:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}")
    return {"case_id": case_id, "backend": backend,
            "stream_url": f"/rca/{case_id}/stream?backend={backend}"}


def _sse(event: str, payload: dict | str, seq: int) -> dict:
    """Build an sse-starlette yield dict carrying a full SSEEvent-shaped payload."""
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False, default=str)
    return {"event": event, "data": payload, "retry": 0}


@app.get("/rca/{case_id}/stream")
async def stream_rca(case_id: str, backend: str = Query(default="parquet")) -> EventSourceResponse:
    cases = list_cases()
    if case_id not in cases:
        raise HTTPException(status_code=404, detail=f"unknown case: {case_id}")

    case, agent = build_agent_for_case(case_id, backend=backend)

    async def event_gen() -> AsyncIterator[dict]:
        seq = 0
        try:
            async for ev in agent.run(case):
                seq += 1
                if isinstance(ev, RcaReport):
                    # Persist (best-effort); never let storage break the stream.
                    try:
                        from ..store.mysql_store import MysqlStore

                        MysqlStore().save_report(ev)
                    except Exception:
                        pass
                    yield _sse(SSEEventKind.REPORT.value,
                               {"event": SSEEventKind.REPORT.value, "case_id": case_id,
                                "data": json.loads(ev.model_dump_json()), "seq": seq}, seq)
                    seq += 1
                    yield _sse(SSEEventKind.DONE.value,
                               {"event": SSEEventKind.DONE.value, "case_id": case_id,
                                "data": {"status": ev.status}, "seq": seq}, seq)
                elif isinstance(ev, RcaStep):
                    yield _sse(SSEEventKind.STEP.value,
                               {"event": SSEEventKind.STEP.value, "case_id": case_id,
                                "data": json.loads(ev.model_dump_json()), "seq": seq}, seq)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # surface errors to the client, then end
            seq += 1
            yield _sse(SSEEventKind.ERROR.value,
                       {"event": SSEEventKind.ERROR.value, "case_id": case_id,
                        "data": {"error": f"{type(e).__name__}: {e}"}, "seq": seq}, seq)

    return EventSourceResponse(event_gen())


@app.get("/reports/{case_id}")
def get_report(case_id: str) -> dict:
    try:
        from ..store.mysql_store import MysqlStore

        store = MysqlStore()
        reports = store.list_reports(case_id=case_id, limit=1)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"storage unavailable: {e}")
    if not reports:
        raise HTTPException(status_code=404, detail="no report for case")
    return json.loads(reports[0].model_dump_json())
