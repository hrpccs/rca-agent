"""Offline tests for the FastAPI SSE server (``rca_agent.server.app``).

No real agent, LLM, DB, or data provider is constructed. The agent factory and
report store are injected via the module-level swappable seams, and case
discovery is pointed at an in-memory list. Every assertion is on the HTTP/SSE
contract the frontend and clients depend on.

Note: ``rca_agent.server.__init__`` re-exports the FastAPI instance as ``app``,
shadowing the submodule of the same name, so the helpers + app instance are
imported by explicit dotted path (``from rca_agent.server.app import ...``).
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rca_agent.contracts import (
    Case,
    Modality,
    RcaReport,
    RcaStep,
    RootCause,
    StepKind,
    Task,
    TimeWindow,
    Topology,
)
from rca_agent.server.app import (
    app as fastapi_app,
)
from rca_agent.server.app import (
    set_agent_factory,
    set_case_lister,
    set_report_store_factory,
)

KNOWN_CASES = ["t001", "t002"]


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeReportStore:
    """In-memory stand-in for MysqlStore exposing save_report + list_reports."""

    def __init__(self) -> None:
        self.saved: list[RcaReport] = []

    def save_report(self, report: RcaReport, run_id: str | None = None) -> str:
        self.saved.append(report)
        return f"rid-{len(self.saved)}"

    def list_reports(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[RcaReport]:
        rows = [r for r in self.saved if case_id is None or r.case_id == case_id]
        return list(reversed(rows[-limit:]))


class FakeAgent:
    """Agent that yields a scripted trace without touching the LLM loop."""

    def __init__(self, trace: list[Any]) -> None:
        self._trace = trace

    async def run(self, case: Case) -> AsyncIterator[Any]:
        for ev in self._trace:
            yield ev


def _fake_case(case_id: str) -> Case:
    """Build a minimal but valid Case for the fake agent factory."""
    from datetime import UTC, datetime

    win = TimeWindow(
        start=datetime(2026, 4, 25, 5, 18, 12, tzinfo=UTC),
        end=datetime(2026, 4, 25, 5, 28, 12, tzinfo=UTC),
    )
    return Case(
        task=Task(
            task_id=case_id,
            alert_title="checkout 错误次数告警",
            alert_window=win,
            prompt_text="帮我分析下根因。",
            available_modalities=[Modality.LOGS, Modality.METRICS],
        ),
        topology=Topology(case_id=case_id, window=win),
        case_dir=f"/tmp/fake-{case_id}",
        modalities=[Modality.LOGS, Modality.METRICS],
    )


def _scripted_trace(case_id: str) -> list[Any]:
    return [
        RcaStep(
            step_id=f"{case_id}-s1",
            case_id=case_id,
            step_kind=StepKind.REASONING,
            thought="I should look at the logs.",
        ),
        RcaStep(
            step_id=f"{case_id}-s2",
            case_id=case_id,
            step_kind=StepKind.TOOL_CALL,
            tool_name="query_logs",
            tool_args={"pod": "checkout-0"},
        ),
        RcaReport(
            case_id=case_id,
            task_id=case_id,
            alert_title="checkout 错误次数告警",
            root_cause=RootCause(
                summary="checkout pod OOMKilled",
                confidence=0.9,
                fault_type="k8s.pod_crashloop",
            ),
            status="completed",
        ),
    ]


def _make_factory(trace_map: dict[str, list[Any]] | None = None):
    """Build an agent factory that returns scripted traces per case_id."""
    trace_map = trace_map or {}

    def factory(case_id: str, backend: str | None = None, **kw: Any):
        return _fake_case(case_id), FakeAgent(
            trace_map.get(case_id, _scripted_trace(case_id))
        )

    return factory


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    """Parse an SSE byte stream into a list of {event, data} dicts.

    The wire format is ``event: <kind>\\ndata: <json>\\n\\n`` repeated.
    """
    events: list[dict[str, Any]] = []
    cur_event: str | None = None
    for line in body.splitlines():
        if not line:
            cur_event = None
            continue
        if line.startswith("event:"):
            cur_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            raw = line[len("data:") :].strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"_raw": raw}
            events.append({"event": cur_event, "data": data})
    return events


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_store() -> FakeReportStore:
    return FakeReportStore()


@pytest.fixture
def client(fake_store: FakeReportStore) -> TestClient:
    """A TestClient wired to synthetic cases + injected store + fake agent."""
    set_case_lister(lambda: list(KNOWN_CASES))
    set_report_store_factory(lambda: fake_store)
    set_agent_factory(_make_factory())
    # Raise any server-side exceptions in the test process (TestClient default
    # already does this for sync handlers; we keep it explicit for clarity).
    yield TestClient(fastapi_app)
    # Restore production defaults so other test modules are unaffected.
    set_case_lister(None)
    set_report_store_factory(None)
    set_agent_factory(None)


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #
def test_health_returns_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


# --------------------------------------------------------------------------- #
# GET /cases
# --------------------------------------------------------------------------- #
def test_cases_list_reflects_injected_cases(client: TestClient):
    r = client.get("/cases")
    assert r.status_code == 200
    body = r.json()
    assert body["cases"] == KNOWN_CASES


def test_cases_endpoint_works_without_real_filesystem(tmp_path, monkeypatch):
    """Cases can be discovered from an env-pointed tmp dir (no monkeypatch of list)."""
    # Point RCA_CASES_DIR at a tmp dir with a synthetic case.
    (tmp_path / "t999" / "task.json").parent.mkdir(parents=True)
    (tmp_path / "t999" / "task.json").write_text("{}")
    monkeypatch.setenv("RCA_CASES_DIR", str(tmp_path))
    # list_cases reads settings.cases_dir; the lru_cache on get_settings means
    # we must clear it so the new env var takes effect.
    from rca_agent.config import get_settings

    get_settings.cache_clear()
    set_case_lister(None)  # ensure default list_cases is in use
    try:
        client = TestClient(fastapi_app)
        r = client.get("/cases")
        assert r.status_code == 200
        assert "t999" in r.json()["cases"]
    finally:
        get_settings.cache_clear()
        set_case_lister(None)


# --------------------------------------------------------------------------- #
# GET /reports/{case_id}
# --------------------------------------------------------------------------- #
def test_report_404_when_store_empty(client: TestClient, fake_store: FakeReportStore):
    assert fake_store.saved == []
    r = client.get("/reports/t001")
    assert r.status_code == 404


def test_report_200_when_store_has_it(client: TestClient, fake_store: FakeReportStore):
    # Seed the injected store directly.
    fake_store.saved.append(
        RcaReport(
            case_id="t001",
            task_id="t001",
            alert_title="boom",
            root_cause=RootCause(summary="db down", confidence=0.5),
            status="completed",
        )
    )
    r = client.get("/reports/t001")
    assert r.status_code == 200
    body = r.json()
    assert body["case_id"] == "t001"
    assert body["root_cause"]["summary"] == "db down"


def test_report_503_when_storage_raises(client: TestClient):
    class ExplodingStore:
        def save_report(self, report, run_id=None):
            raise RuntimeError("nope")

        def list_reports(self, case_id=None, limit=50):
            raise RuntimeError("connection refused")

    set_report_store_factory(lambda: ExplodingStore())
    try:
        r = client.get("/reports/t001")
        assert r.status_code == 503
        assert "storage unavailable" in r.json()["detail"]
    finally:
        set_report_store_factory(None)


# --------------------------------------------------------------------------- #
# POST /rca/{case_id}
# --------------------------------------------------------------------------- #
def test_start_rca_accepted_for_known_case(client: TestClient):
    r = client.post("/rca/t001")
    assert r.status_code == 200
    body = r.json()
    assert body["case_id"] == "t001"
    assert body["stream_url"] == "/rca/t001/stream?backend=parquet"


def test_start_rca_404_for_unknown_case(client: TestClient):
    r = client.post("/rca/does-not-exist")
    assert r.status_code == 404


def test_start_rca_respects_backend_query(client: TestClient):
    r = client.post("/rca/t001?backend=clickhouse")
    assert r.status_code == 200
    assert r.json()["backend"] == "clickhouse"
    assert "backend=clickhouse" in r.json()["stream_url"]


# --------------------------------------------------------------------------- #
# GET /rca/{case_id}/stream  (SSE)
# --------------------------------------------------------------------------- #
def test_stream_emits_steps_then_report_then_done(client: TestClient):
    with client.stream("GET", "/rca/t001/stream") as resp:
        assert resp.status_code == 200
        body = resp.read().decode()

    events = _parse_sse_events(body)
    assert events, "no SSE events emitted"

    kinds = [e["event"] for e in events]
    # Two scripted steps (reasoning + tool_call), then report, then done.
    assert kinds.count("step") == 2
    assert "report" in kinds
    assert kinds[-1] == "done", "stream must terminate with a done event"

    # The terminal report must precede done and carry the report payload.
    report_ev = next(e for e in events if e["event"] == "report")
    payload = report_ev["data"]
    assert payload["event"] == "report"
    assert payload["case_id"] == "t001"
    assert "data" in payload and "seq" in payload
    assert payload["data"]["root_cause"]["summary"] == "checkout pod OOMKilled"

    # seq must be monotonically increasing across the whole stream.
    seqs = [e["data"]["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs), "seq values must be unique"

    # Each step event's data envelope has the {event, case_id, data, seq} shape.
    for ev in events:
        assert ev["data"]["event"] == ev["event"]
        assert ev["data"]["case_id"] == "t001"
        assert isinstance(ev["data"]["seq"], int)

    # Serialization uses model_dump(mode="json"): datetimes must come back as
    # ISO strings (not raw datetime objects), proving the JSON-compatible dump.
    step_payload = next(e for e in events if e["event"] == "step")["data"]["data"]
    assert isinstance(step_payload["ts"], str)


def test_stream_terminates_after_first_report_even_if_agent_keeps_yielding(
    client: TestClient,
):
    """A report is terminal: the server must stop consuming the agent after the
    first RcaReport, so a misbehaving agent that yields steps or a second report
    afterwards cannot produce duplicate terminal events or hold the connection.
    """

    class DoubleReportingAgent:
        async def run(self, case):
            yield _scripted_trace("t001")[0]  # one step
            yield _scripted_trace("t001")[2]  # first report -> should end stream
            yield _scripted_trace("t001")[1]  # extra step after report (must be dropped)
            yield _scripted_trace("t001")[2]  # second report (must be dropped)

    def factory(case_id, backend=None, **kw):
        return _fake_case(case_id), DoubleReportingAgent()

    set_agent_factory(factory)
    try:
        with client.stream("GET", "/rca/t001/stream") as resp:
            assert resp.status_code == 200
            body = resp.read().decode()
    finally:
        set_agent_factory(None)

    events = _parse_sse_events(body)
    kinds = [e["event"] for e in events]
    # Exactly one step, one report, one done — the post-report yields are dropped.
    assert kinds.count("step") == 1
    assert kinds.count("report") == 1
    assert kinds.count("done") == 1
    assert kinds[-1] == "done"


def test_stream_persists_report_to_injected_store(
    client: TestClient, fake_store: FakeReportStore
):
    with client.stream("GET", "/rca/t001/stream") as resp:
        resp.read()
    assert len(fake_store.saved) == 1
    assert fake_store.saved[0].case_id == "t001"


def test_stream_terminates_with_error_when_agent_raises(client: TestClient):
    class RaisingAgent:
        async def run(self, case):
            yield _scripted_trace("t001")[0]  # one step before the blow-up
            raise RuntimeError("boom-in-agent")
            yield  # pragma: no cover - unreachable

    def factory(case_id, backend=None, **kw):
        return _fake_case(case_id), RaisingAgent()

    set_agent_factory(factory)
    try:
        with client.stream("GET", "/rca/t001/stream") as resp:
            assert resp.status_code == 200
            body = resp.read().decode()
    finally:
        set_agent_factory(None)

    events = _parse_sse_events(body)
    kinds = [e["event"] for e in events]
    assert "step" in kinds  # the step emitted before the raise is preserved
    assert kinds[-1] == "error", "a raising agent must still terminate the stream"
    err = next(e for e in events if e["event"] == "error")
    assert "boom-in-agent" in err["data"]["data"]["error"]


def test_stream_404_for_unknown_case(client: TestClient):
    r = client.get("/rca/unknown-case/stream")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# case_id validation — path traversal / unsafe ids never reach fs/agent/store
# --------------------------------------------------------------------------- #
# Ids whose raw path segment reaches the handler and is rejected by validation
# (these contain no ``/`` so the HTTP client does not normalize them away).
_IDS_REJECTED_BY_VALIDATOR = ["t001;", "t001 space", ".hidden", "-dash", "a\\b"]


@pytest.mark.parametrize("bad_id", _IDS_REJECTED_BY_VALIDATOR)
def test_unsafe_case_ids_return_400(client: TestClient, bad_id: str):
    """Ids that reach the handler but fail the identifier check -> 400."""
    for method, path in [
        ("POST", f"/rca/{bad_id}"),
        ("GET", f"/rca/{bad_id}/stream"),
        ("GET", f"/reports/{bad_id}"),
    ]:
        r = client.request(method, path)
        assert r.status_code == 400, (
            f"{method} {path} should be 400, got {r.status_code}"
        )


@pytest.mark.parametrize(
    "traversal_id",
    ["../etc/passwd", "..%2Fetc%2Fpasswd", "a/b"],
)
def test_path_traversal_ids_never_reach_agent(
    client: TestClient, traversal_id: str
):
    """Traversal sequences that escape the /rca/<seg> route are normalized away
    by the HTTP client / router and therefore never reach the agent factory or
    filesystem as the raw segment. They must NOT return 200 — a 4xx (or routing
    elsewhere) is the safe outcome. The security contract is: the raw traversal
    string is never handed downstream as a case_id."""
    for method, path in [
        ("POST", f"/rca/{traversal_id}"),
        ("GET", f"/rca/{traversal_id}/stream"),
        ("GET", f"/reports/{traversal_id}"),
    ]:
        r = client.request(method, path)
        # Never 200 for the raw traversal path; it must not be treated as valid.
        assert r.status_code != 200, (
            f"{method} {path} returned 200 — traversal id leaked downstream"
        )


def test_traversal_collapses_to_valid_case_does_not_leak_raw_segment(
    client: TestClient,
):
    """``t001/../t002`` is normalized by the client to ``/rca/t002``. The handler
    only ever sees ``t002`` (a known case), never the ``..`` sequence, so the
    filesystem/agent are never handed a traversal string. This documents that
    behavior rather than treating it as a vulnerability."""
    r = client.post("/rca/t001/../t002")
    assert r.status_code == 200
    # The effective case_id reaching the handler is the normalized tail.
    assert r.json()["case_id"] == "t002"


def test_empty_case_id_returns_404(client: TestClient):
    """An empty path segment does not match the route -> 404 (never forwarded)."""
    assert client.post("/rca/").status_code in (400, 404)
    assert client.get("/rca//stream").status_code in (400, 404)


@pytest.mark.parametrize(
    "good_id", ["t001", "t_002", "t-003", "case.4", "T005", "v1..2"]
)
def test_valid_case_ids_pass_validation(client: TestClient, good_id: str):
    # Add the id to the injected case list so downstream 404 checks are about
    # existence, not validation. We only assert validation did NOT 400.
    #
    # ``v1..2`` is included to document the security model: a case_id with
    # consecutive dots but NO path separator is a single, safe directory
    # segment (``root / "v1..2"`` stays inside ``root``) and must be accepted.
    # Only ``/`` and ``\`` (excluded by the regex char class) are traversal
    # vectors; ``..`` alone cannot escape without a separator.
    set_case_lister(lambda: [good_id])
    try:
        r = client.post(f"/rca/{good_id}")
        assert r.status_code == 200, f"valid id {good_id} rejected with {r.status_code}"
    finally:
        set_case_lister(None)
