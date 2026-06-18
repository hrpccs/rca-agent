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
from datetime import UTC, datetime, timedelta
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
    set_trace_store_factory,
)

KNOWN_CASES = ["t001", "t002"]
# A fixed 32-char hex run_id the fake trace store hands out.
FIXED_RUN_ID = "0123456789abcdef0123456789abcdef"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeReportStore:
    """In-memory stand-in for MysqlStore exposing save_report + list_reports."""

    def __init__(self) -> None:
        self.saved: list[RcaReport] = []
        # run_id passed alongside each save_report call (parallel to ``saved``).
        self.saved_run_ids: list[str | None] = []

    def save_report(self, report: RcaReport, run_id: str | None = None) -> str:
        self.saved.append(report)
        self.saved_run_ids.append(run_id)
        return f"rid-{len(self.saved)}"

    def list_reports(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[RcaReport]:
        rows = [r for r in self.saved if case_id is None or r.case_id == case_id]
        return list(reversed(rows[-limit:]))


class FakeTraceStore:
    """In-memory stand-in for MysqlStore's run/trace methods.

    Records the order and arguments of every persistence call so tests can
    assert that steps are persisted in order, runs are closed with the right
    terminal status, and a run_id minted at POST time is honored through the
    stream. Structurally satisfies the server's ``TraceStore`` Protocol.
    """

    def __init__(self, run_id: str = FIXED_RUN_ID) -> None:
        self.run_id = run_id
        self.started_runs: list[tuple[str, str]] = []  # (case_id, model)
        self.finished_runs: list[tuple[str, str, dict | None]] = []
        # (run_id, case_id, seq, step)
        self.appended_steps: list[tuple[str, str, int, RcaStep]] = []
        # run_id -> list of runs-dict rows for list_runs/get_run.
        self.run_rows: dict[str, dict[str, Any]] = {}
        # run_id -> list of steps for list_steps.
        self.step_rows: dict[str, list[RcaStep]] = {}

    def start_run(self, case_id: str, model: str) -> str:
        self.started_runs.append((case_id, model))
        self.run_rows.setdefault(
            self.run_id,
            {
                "run_id": self.run_id,
                "case_id": case_id,
                "status": "running",
                "model": model,
            },
        )
        self.step_rows.setdefault(self.run_id, [])
        return self.run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        token_usage: dict[str, Any] | None = None,
    ) -> None:
        self.finished_runs.append((run_id, status, token_usage))
        if run_id in self.run_rows:
            self.run_rows[run_id]["status"] = status

    def append_step(
        self, run_id: str, case_id: str, seq: int, step: RcaStep
    ) -> None:
        self.appended_steps.append((run_id, case_id, seq, step))
        self.step_rows.setdefault(run_id, []).append(step)

    def list_runs(
        self, case_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        rows = list(self.run_rows.values())
        if case_id is not None:
            rows = [r for r in rows if r.get("case_id") == case_id]
        return rows[:limit]

    def list_steps(self, run_id: str, limit: int = 20000) -> list[RcaStep]:
        return list(self.step_rows.get(run_id, []))[:limit]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.run_rows.get(run_id)


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
def fake_trace() -> FakeTraceStore:
    return FakeTraceStore()


@pytest.fixture
def client(
    fake_store: FakeReportStore, fake_trace: FakeTraceStore
) -> TestClient:
    """A TestClient wired to synthetic cases + injected stores + fake agent."""
    set_case_lister(lambda: list(KNOWN_CASES))
    set_report_store_factory(lambda: fake_store)
    set_trace_store_factory(lambda: fake_trace)
    set_agent_factory(_make_factory())
    # Raise any server-side exceptions in the test process (TestClient default
    # already does this for sync handlers; we keep it explicit for clarity).
    yield TestClient(fastapi_app)
    # Restore production defaults so other test modules are unaffected.
    set_case_lister(None)
    set_report_store_factory(None)
    set_trace_store_factory(None)
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
    # POST mints a run_id via the injected fake trace store and threads it into
    # the stream_url so the frontend can re-fetch the trace after a disconnect.
    assert body["run_id"] == FIXED_RUN_ID
    assert body["stream_url"] == (
        f"/rca/t001/stream?backend=parquet&run_id={FIXED_RUN_ID}"
    )


def test_start_rca_404_for_unknown_case(client: TestClient):
    r = client.post("/rca/does-not-exist")
    assert r.status_code == 404


def test_start_rca_respects_backend_query(client: TestClient):
    r = client.post("/rca/t001?backend=clickhouse")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "clickhouse"
    assert "backend=clickhouse" in body["stream_url"]
    assert f"run_id={FIXED_RUN_ID}" in body["stream_url"]


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


# --------------------------------------------------------------------------- #
# T2: incremental trace persistence + run_id flow
# --------------------------------------------------------------------------- #
def test_stream_persists_steps_in_order_then_closes_run(
    client: TestClient, fake_trace: FakeTraceStore, fake_store: FakeReportStore
):
    """Each emitted RcaStep is persisted in order with the right seq; the run is
    closed with the report's terminal status and token_usage; save_report is
    still called on the report store."""
    with client.stream(
        "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
    ) as resp:
        assert resp.status_code == 200
        resp.read()

    # Two scripted steps persisted, seq values 1 and 2 (seq increments once per
    # emitted item, steps come first), in insertion order.
    assert len(fake_trace.appended_steps) == 2
    seqs = [rec[2] for rec in fake_trace.appended_steps]
    assert seqs == [1, 2]
    step_ids = [rec[3].step_id for rec in fake_trace.appended_steps]
    assert step_ids == ["t001-s1", "t001-s2"]
    # All persisted against the run_id carried by the stream query param.
    assert all(rec[0] == FIXED_RUN_ID for rec in fake_trace.appended_steps)

    # Run closed exactly once with completed status + the report's token_usage.
    assert len(fake_trace.finished_runs) == 1
    run_id, status, usage = fake_trace.finished_runs[0]
    assert run_id == FIXED_RUN_ID
    assert status == "completed"
    # token_usage on the scripted report is None -> passed through as None.
    assert usage is None

    # The report is still saved to the report store (no regression).
    assert len(fake_store.saved) == 1
    assert fake_store.saved[0].case_id == "t001"


def test_stream_mints_run_id_when_none_passed(
    client: TestClient, fake_trace: FakeTraceStore
):
    """A stream opened without ?run_id= mints one via start_run so each step is
    still persisted incrementally."""
    with client.stream("GET", "/rca/t001/stream") as resp:
        assert resp.status_code == 200
        resp.read()
    # start_run was called exactly once in the stream (POST was not used here).
    assert len(fake_trace.started_runs) == 1
    case_id, model = fake_trace.started_runs[0]
    assert case_id == "t001"
    assert model  # deepseek_model default is non-empty
    # Steps were persisted against the minted run_id.
    assert fake_trace.appended_steps
    assert all(rec[0] == FIXED_RUN_ID for rec in fake_trace.appended_steps)


def test_stream_run_id_query_param_is_honored_not_reminted(
    client: TestClient, fake_trace: FakeTraceStore
):
    """When ?run_id= is supplied, the stream must NOT call start_run again."""
    with client.stream(
        "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
    ) as resp:
        resp.read()
    assert fake_trace.started_runs == []  # not re-minted
    assert fake_trace.appended_steps  # still persisted against the passed id


def test_stream_persists_steps_before_exception_and_closes_as_error(
    client: TestClient, fake_trace: FakeTraceStore
):
    """A raising producer still emits ERROR and closes the run as 'error'; steps
    emitted before the raise were already persisted."""

    class RaisingAgent:
        async def run(self, case):
            yield _scripted_trace("t001")[0]  # one step before the blow-up
            raise RuntimeError("boom-in-agent")
            yield  # pragma: no cover - unreachable

    def factory(case_id, backend=None, **kw):
        return _fake_case(case_id), RaisingAgent()

    set_agent_factory(factory)
    try:
        with client.stream(
            "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
        ) as resp:
            assert resp.status_code == 200
            body = resp.read().decode()
    finally:
        set_agent_factory(None)

    events = _parse_sse_events(body)
    kinds = [e["event"] for e in events]
    assert "step" in kinds
    assert kinds[-1] == "error"

    # The pre-exception step was persisted.
    assert len(fake_trace.appended_steps) == 1
    assert fake_trace.appended_steps[0][3].step_id == "t001-s1"
    # Run closed as errored (no token_usage on the error path).
    assert fake_trace.finished_runs == [(FIXED_RUN_ID, "error", None)]


def test_stream_bad_run_id_returns_400(client: TestClient):
    """A run_id that is not 32-char hex is rejected before reaching the store."""
    r = client.get("/rca/t001/stream?run_id=not-a-real-id")
    assert r.status_code == 400


def test_stream_trace_failure_does_not_break_stream(
    fake_store: FakeReportStore,
):
    """If the trace store itself raises on construction, the stream must still
    deliver steps/report/done and call save_report — persistence is best-effort."""
    set_case_lister(lambda: list(KNOWN_CASES))
    set_report_store_factory(lambda: fake_store)
    set_agent_factory(_make_factory())

    def exploding_trace():
        raise RuntimeError("trace store down")

    set_trace_store_factory(exploding_trace)
    try:
        client = TestClient(fastapi_app)
        with client.stream("GET", "/rca/t001/stream") as resp:
            assert resp.status_code == 200
            body = resp.read().decode()
    finally:
        set_case_lister(None)
        set_report_store_factory(None)
        set_trace_store_factory(None)
        set_agent_factory(None)

    events = _parse_sse_events(body)
    kinds = [e["event"] for e in events]
    assert kinds.count("step") == 2
    assert "report" in kinds
    assert kinds[-1] == "done"
    assert len(fake_store.saved) == 1


def test_start_rca_run_id_none_when_trace_store_down(
    fake_store: FakeReportStore,
):
    """POST /rca must still succeed (200) with run_id=None when the trace store
    is unavailable — and the stream_url must omit the run_id param."""
    set_case_lister(lambda: list(KNOWN_CASES))
    set_report_store_factory(lambda: fake_store)

    def exploding_trace():
        raise RuntimeError("trace store down")

    set_trace_store_factory(exploding_trace)
    set_agent_factory(_make_factory())
    try:
        client = TestClient(fastapi_app)
        r = client.post("/rca/t001")
        assert r.status_code == 200
        body = r.json()
        assert body["run_id"] is None
        assert "run_id=" not in body["stream_url"]
        assert body["stream_url"] == "/rca/t001/stream?backend=parquet"
    finally:
        set_case_lister(None)
        set_report_store_factory(None)
        set_trace_store_factory(None)
        set_agent_factory(None)


# --------------------------------------------------------------------------- #
# GET /runs, /runs/{run_id}, /runs/{run_id}/steps, /cases/{case_id}/runs
# --------------------------------------------------------------------------- #
def test_list_runs_returns_fake_store_envelope(
    client: TestClient, fake_trace: FakeTraceStore
):
    # Seed two runs in the fake store.
    fake_trace.run_rows = {
        FIXED_RUN_ID: {
            "run_id": FIXED_RUN_ID,
            "case_id": "t001",
            "status": "completed",
        },
        "fedcba9876543210fedcba9876543210": {
            "run_id": "fedcba9876543210fedcba9876543210",
            "case_id": "t002",
            "status": "running",
        },
    }
    r = client.get("/runs")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["runs"], list)
    assert len(body["runs"]) == 2


def test_list_runs_filters_by_case_id(
    client: TestClient, fake_trace: FakeTraceStore
):
    fake_trace.run_rows = {
        FIXED_RUN_ID: {
            "run_id": FIXED_RUN_ID,
            "case_id": "t001",
            "status": "completed",
        },
        "fedcba9876543210fedcba9876543210": {
            "run_id": "fedcba9876543210fedcba9876543210",
            "case_id": "t002",
            "status": "running",
        },
    }
    r = client.get("/runs?case_id=t001")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["case_id"] == "t001"


def test_list_runs_400_for_bad_case_id(client: TestClient):
    r = client.get("/runs?case_id=bad/id")
    assert r.status_code == 400


def test_list_runs_503_when_store_raises(client: TestClient):
    def exploding():
        raise RuntimeError("connection refused")

    set_trace_store_factory(exploding)
    try:
        r = client.get("/runs")
        assert r.status_code == 503
        assert "storage unavailable" in r.json()["detail"]
    finally:
        set_trace_store_factory(None)


def test_get_run_returns_summary_and_steps(
    client: TestClient, fake_trace: FakeTraceStore
):
    step = RcaStep(
        step_id="t001-s1",
        case_id="t001",
        step_kind=StepKind.REASONING,
        thought="hi",
    )
    fake_trace.run_rows = {
        FIXED_RUN_ID: {
            "run_id": FIXED_RUN_ID,
            "case_id": "t001",
            "status": "completed",
        }
    }
    fake_trace.step_rows = {FIXED_RUN_ID: [step]}

    r = client.get(f"/runs/{FIXED_RUN_ID}")
    assert r.status_code == 200
    body = r.json()
    assert body["run"]["run_id"] == FIXED_RUN_ID
    assert len(body["steps"]) == 1
    assert body["steps"][0]["step_id"] == "t001-s1"


def test_get_run_404_for_unknown_run(
    client: TestClient, fake_trace: FakeTraceStore
):
    # Empty store: get_run returns None.
    r = client.get(f"/runs/{FIXED_RUN_ID}")
    assert r.status_code == 404
    assert r.json()["detail"] == "unknown run"


def test_get_run_400_for_bad_run_id(client: TestClient):
    r = client.get("/runs/not-hex")
    assert r.status_code == 400


def test_get_run_503_when_store_raises(client: TestClient):
    def exploding():
        raise RuntimeError("connection refused")

    set_trace_store_factory(exploding)
    try:
        r = client.get(f"/runs/{FIXED_RUN_ID}")
        assert r.status_code == 503
    finally:
        set_trace_store_factory(None)


def test_list_run_steps_endpoint(
    client: TestClient, fake_trace: FakeTraceStore
):
    step = RcaStep(
        step_id="t001-s1",
        case_id="t001",
        step_kind=StepKind.TOOL_CALL,
        tool_name="query_logs",
    )
    fake_trace.step_rows = {FIXED_RUN_ID: [step]}

    r = client.get(f"/runs/{FIXED_RUN_ID}/steps")
    assert r.status_code == 200
    body = r.json()
    assert len(body["steps"]) == 1
    assert body["steps"][0]["tool_name"] == "query_logs"


def test_list_run_steps_400_for_bad_run_id(client: TestClient):
    r = client.get("/runs/zzz/steps")
    assert r.status_code == 400


def test_list_case_runs_endpoint(
    client: TestClient, fake_trace: FakeTraceStore
):
    fake_trace.run_rows = {
        FIXED_RUN_ID: {
            "run_id": FIXED_RUN_ID,
            "case_id": "t001",
            "status": "completed",
        }
    }
    r = client.get("/cases/t001/runs")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["case_id"] == "t001"


def test_list_case_runs_400_for_bad_case_id(client: TestClient):
    # A path-separator id never reaches the handler (the router collapses it),
    # so it must NOT be 200. A safe (4xx) outcome is the contract.
    r = client.get("/cases/bad/id/runs")
    assert r.status_code != 200
    # A bad id that DOES reach the handler as a single segment is 400.
    r = client.get("/cases/bad;id/runs")
    assert r.status_code == 400


def test_list_case_runs_503_when_store_raises(client: TestClient):
    def exploding():
        raise RuntimeError("connection refused")

    set_trace_store_factory(exploding)
    try:
        r = client.get("/cases/t001/runs")
        assert r.status_code == 503
    finally:
        set_trace_store_factory(None)


# --------------------------------------------------------------------------- #
# Heartbeat regression — NAMED ping still emitted on a slow producer
# --------------------------------------------------------------------------- #
def test_heartbeat_ping_emitted_on_slow_producer(
    fake_store: FakeReportStore, fake_trace: FakeTraceStore, monkeypatch
):
    """A producer that sleeps longer than the heartbeat interval must yield at
    least one ping before the first real event.

    The ping MUST be a NAMED ``event: ping`` (not a data-only message): the
    frontend listens via ``addEventListener("ping")`` and has no ``onmessage``
    handler, so a data-only ping would be silently dropped and never re-arm the
    client's idle watchdog. This is the regression that caused live idle
    disconnects during long DeepSeek turns."""
    import asyncio

    set_case_lister(lambda: list(KNOWN_CASES))
    set_report_store_factory(lambda: fake_store)
    set_trace_store_factory(lambda: fake_trace)
    # Tight heartbeat so the test stays fast. The producer sleeps much longer
    # than the heartbeat (0.5s vs 0.05s — a 10x margin) so the ping reliably
    # fires before the first real event even under CI scheduling jitter.
    monkeypatch.setenv("RCA_SSE_HEARTBEAT_SEC", "0.05")

    class SlowAgent:
        async def run(self, case):
            await asyncio.sleep(0.5)
            yield _scripted_trace("t001")[0]
            yield _scripted_trace("t001")[2]  # report -> terminal

    def factory(case_id, backend=None, **kw):
        return _fake_case(case_id), SlowAgent()

    set_agent_factory(factory)
    try:
        client = TestClient(fastapi_app)
        with client.stream("GET", "/rca/t001/stream") as resp:
            assert resp.status_code == 200
            body = resp.read().decode()
    finally:
        set_case_lister(None)
        set_report_store_factory(None)
        set_trace_store_factory(None)
        set_agent_factory(None)

    events = _parse_sse_events(body)
    # The ping MUST be NAMED: the parsed event name is "ping" (NOT None), and
    # its data carries event=="ping". A data-only ping (event is None) would
    # never fire the frontend's addEventListener("ping") and is the bug.
    pings_named = [
        e for e in events if e["event"] == "ping" and e["data"].get("event") == "ping"
    ]
    assert pings_named, (
        "expected at least one NAMED event:ping before the first step; a "
        "data-only ping would be dropped by the frontend's listener"
    )
    # A data-only ping (the old buggy format) must NOT be present.
    pings_unnamed = [
        e
        for e in events
        if e["event"] is None and e["data"].get("event") == "ping"
    ]
    assert not pings_unnamed, "ping must be NAMED, not a data-only message"
    # The stream still terminates normally after the report.
    kinds = [e["event"] for e in events if e["event"] is not None]
    assert kinds[-1] == "done"


# --------------------------------------------------------------------------- #
# Regression: abandoned run is closed as 'interrupted'; run_id threaded to
# save_report; finish_run not double-called.
# --------------------------------------------------------------------------- #
def test_stream_closes_run_as_interrupted_when_producer_ends_without_report(
    client: TestClient, fake_trace: FakeTraceStore
):
    """A producer that yields steps then ends cleanly (no RcaReport, no
    exception) must still close the run — as 'interrupted' — so it does not
    linger in 'running' forever. Guards against the abandoned-run leak.

    Status is 'interrupted' (NOT 'truncated'): 'truncated' is reserved for the
    agent's own step-cap force-conclude, so ops can distinguish a stream that
    was abandoned from one where the agent bailed at its step limit."""

    class NoReportAgent:
        async def run(self, case):
            yield _scripted_trace("t001")[0]  # one step, then ends

    def factory(case_id, backend=None, **kw):
        return _fake_case(case_id), NoReportAgent()

    set_agent_factory(factory)
    try:
        with client.stream(
            "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
        ) as resp:
            assert resp.status_code == 200
            resp.read()
    finally:
        set_agent_factory(None)

    # The one step was persisted.
    assert len(fake_trace.appended_steps) == 1
    # Exactly one finish_run, with status 'interrupted' (the abandoned-run
    # closer in the finally block), NOT 'completed' and NOT zero calls.
    assert len(fake_trace.finished_runs) == 1
    assert fake_trace.finished_runs[0] == (FIXED_RUN_ID, "interrupted", None)


def test_stream_passes_run_id_to_save_report(
    client: TestClient, fake_store: FakeReportStore
):
    """The report-persistence seam must thread the effective run_id so the
    report row is linked to the incremental trace run."""
    with client.stream(
        "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
    ) as resp:
        resp.read()
    assert fake_store.saved_run_ids == [FIXED_RUN_ID]


def test_stream_finish_run_called_exactly_once_on_normal_completion(
    client: TestClient, fake_trace: FakeTraceStore
):
    """The terminal REPORT branch closes the run; the finally-block abandoned-
    run closer must NOT double-close it (run_closed guard)."""
    with client.stream(
        "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
    ) as resp:
        resp.read()
    assert len(fake_trace.finished_runs) == 1
    assert fake_trace.finished_runs[0][1] == "completed"


def test_stream_finish_run_called_exactly_once_on_error(
    client: TestClient, fake_trace: FakeTraceStore
):
    """The ERROR branch closes the run as 'error'; the finally closer must not
    re-close it (would otherwise flip 'error' -> 'interrupted')."""

    class RaisingAgent:
        async def run(self, case):
            raise RuntimeError("immediate boom")
            yield  # pragma: no cover

    def factory(case_id, backend=None, **kw):
        return _fake_case(case_id), RaisingAgent()

    set_agent_factory(factory)
    try:
        with client.stream(
            "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
        ) as resp:
            resp.read()
    finally:
        set_agent_factory(None)

    assert len(fake_trace.finished_runs) == 1
    assert fake_trace.finished_runs[0] == (FIXED_RUN_ID, "error", None)


# --------------------------------------------------------------------------- #
# F1: SSE resilience — named ping, disconnect cleanup, orphan-run reaper
# --------------------------------------------------------------------------- #
def test_termination_status_report_completes_run(
    client: TestClient, fake_trace: FakeTraceStore
):
    """REPORT branch stamps status='completed' (from the report's own status)."""
    with client.stream(
        "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
    ) as resp:
        resp.read()
    statuses = [s for _, s, _ in fake_trace.finished_runs]
    assert statuses == ["completed"]


def test_termination_status_truncated_when_report_says_truncated(
    client: TestClient, fake_trace: FakeTraceStore
):
    """When the agent's own report carries status='truncated' (the step-cap
    force-conclude path), the run is closed as 'truncated' — distinct from the
    'interrupted' abandonment status. Proves the two paths are distinguishable."""

    class TruncatedReportAgent:
        async def run(self, case):
            yield _scripted_trace("t001")[0]
            rep = _scripted_trace("t001")[2]
            rep.status = "truncated"
            yield rep

    def factory(case_id, backend=None, **kw):
        return _fake_case(case_id), TruncatedReportAgent()

    set_agent_factory(factory)
    try:
        with client.stream(
            "GET", f"/rca/t001/stream?run_id={FIXED_RUN_ID}"
        ) as resp:
            resp.read()
    finally:
        set_agent_factory(None)

    assert len(fake_trace.finished_runs) == 1
    assert fake_trace.finished_runs[0][1] == "truncated"


async def _drive_event_gen_then_aclose(case_id: str = "t001", run_id: str = FIXED_RUN_ID):
    """Build the stream's inner ``event_gen`` via the real ``stream_rca``
    endpoint and close it early (``.aclose()``) to simulate a client disconnect
    mid-stream.

    sse-starlette does not guarantee generator cancellation on disconnect, so
    we cannot reliably reproduce the disconnect via TestClient. Instead we call
    ``stream_rca`` directly (with a stub ``Request`` whose ``is_disconnected``
    reports True) to get the ``EventSourceResponse``, drive its body iterator
    one step, then ``.aclose()`` it — which triggers the ``finally`` exactly as
    a GeneratorExit would in the real transport. This is the unit test that
    would have caught the original 'run leaks as running on disconnect' bug.
    """
    # NOTE: ``rca_agent.server.__init__`` re-exports the FastAPI instance as
    # ``app``, shadowing the submodule. Import the endpoint + setters by their
    # explicit dotted path (see the module docstring at the top of this file).
    from rca_agent.server.app import (
        set_agent_factory,
        set_case_lister,
        set_report_store_factory,
        set_trace_store_factory,
        stream_rca,
    )

    # Scripted slow agent: yields one step then blocks forever (until closed).
    class HangingAgent:
        def __init__(self):
            self.cancelled = False

        async def run(self, case):
            yield _scripted_trace(case_id)[0]
            # Block until the generator is closed (CancelledError) — mimics a
            # long DeepSeek turn in flight when the client drops.
            try:
                import asyncio as _aio

                await _aio.Event().wait()
            except BaseException:
                self.cancelled = True
                raise

    hanging = HangingAgent()

    def factory(cid, backend=None, **kw):
        return _fake_case(cid), hanging

    fake_trace_local = FakeTraceStore(run_id=run_id)
    fake_report_local = FakeReportStore()
    set_case_lister(lambda: list(KNOWN_CASES))
    set_trace_store_factory(lambda: fake_trace_local)
    set_report_store_factory(lambda: fake_report_local)
    set_agent_factory(factory)
    try:
        # Stub Request whose is_disconnected() reports True so the heartbeat
        # path breaks on the first silence — the primary F1 disconnect path.
        class StubRequest:
            async def is_disconnected(self):
                return True

        # stream_rca is an async endpoint returning EventSourceResponse. Call
        # it to get the response, then pull its body iterator (which wraps the
        # inner event_gen). Closing that iterator closes event_gen.
        resp = await stream_rca(
            StubRequest(),  # type: ignore[arg-type]
            case_id=case_id,
            run_id=run_id,
        )
        body_iter = resp.body_iterator
        # Drive one real event (the step), then close mid-stream.
        first = None
        async for chunk in body_iter:
            first = chunk
            break
        assert first is not None, "expected at least one step before close"
        # Close mid-stream — simulates client disconnect.
        if hasattr(body_iter, "aclose"):
            await body_iter.aclose()
    finally:
        set_case_lister(None)
        set_trace_store_factory(None)
        set_report_store_factory(None)
        set_agent_factory(None)
    return fake_trace_local, hanging


@pytest.mark.asyncio
async def test_stream_disconnect_closes_run_as_interrupted_once():
    """Client disconnect mid-stream (simulated via early ``.aclose()`` of the
    response body iterator) must close the run exactly once as 'interrupted',
    not leave it 'running' and not double-close. This is the regression test
    for the live 'runs leak as running on disconnect' bug."""
    fake_trace_local, hanging = await _drive_event_gen_then_aclose()
    # The producer was cancelled (teardown did not outlive the request).
    assert hanging.cancelled, "produce() task should have been cancelled on close"
    # Exactly one finish_run, status 'interrupted' — NOT 'running' leak, NOT
    # double-closed, NOT 'truncated' (that's the step-cap path).
    assert len(fake_trace_local.finished_runs) == 1
    assert fake_trace_local.finished_runs[0][1] == "interrupted"


def test_reap_orphan_runs_closes_only_stale_running(monkeypatch):
    """The orphan-run reaper closes only ``running`` rows older than the reap
    age, with status 'interrupted', and leaves fresh/completed/errored rows
    alone. Uses only the existing TraceStore methods (no mysql_store edit)."""
    from rca_agent.server.app import _reap_orphan_runs

    # Tight reap window so the test is deterministic: anything > 0.01 min
    # (~0.6s) old is stale. Fresh rows are < that.
    monkeypatch.setenv("RCA_RUN_REAP_MIN", "0.01")

    store = FakeTraceStore()
    now = datetime.now(UTC)

    def _row(rid, status, started):
        return {"run_id": rid, "case_id": "t001", "status": status,
                "started_at": started}

    store.run_rows = {
        # stale + running -> reaped
        "a" * 32: _row("a" * 32, "running", now - timedelta(minutes=5)),
        # fresh + running -> left alone
        "b" * 32: _row("b" * 32, "running", now),
        # stale + completed -> left alone (not running)
        "c" * 32: _row("c" * 32, "completed", now - timedelta(minutes=5)),
        # stale + error -> left alone
        "d" * 32: _row("d" * 32, "error", now - timedelta(minutes=5)),
    }
    # FakeTraceStore.list_runs ignores the case_id arg and returns all rows.

    closed = _reap_orphan_runs(store)
    assert closed == 1
    statuses = {rid: r["status"] for rid, r in store.run_rows.items()}
    assert statuses["a" * 32] == "interrupted"
    assert statuses["b" * 32] == "running"
    assert statuses["c" * 32] == "completed"
    assert statuses["d" * 32] == "error"
    # Exactly one finish_run call, against the stale running row.
    assert len(store.finished_runs) == 1
    assert store.finished_runs[0][0] == "a" * 32
    assert store.finished_runs[0][1] == "interrupted"


def test_reap_orphan_runs_robust_to_bad_started_at(monkeypatch):
    """The reaper must never crash on a row with an unparseable/missing
    ``started_at``; it skips that row and continues the sweep."""
    from rca_agent.server.app import _reap_orphan_runs

    monkeypatch.setenv("RCA_RUN_REAP_MIN", "0.01")
    store = FakeTraceStore()
    now = datetime.now(UTC)
    store.run_rows = {
        "a" * 32: {"run_id": "a" * 32, "status": "running",
                   "started_at": "not-a-date"},
        "b" * 32: {"run_id": "b" * 32, "status": "running", "started_at": None},
        "c" * 32: {"run_id": "c" * 32, "status": "running",
                   "started_at": now - timedelta(days=1)},
        # missing started_at key entirely
        "e" * 32: {"run_id": "e" * 32, "status": "running"},
    }
    # Should not raise; only the genuinely-old parseable row is reaped.
    closed = _reap_orphan_runs(store)
    assert closed == 1
    assert store.run_rows["c" * 32]["status"] == "interrupted"
    # The bad rows were left as running (skipped, not crashed).
    assert store.run_rows["a" * 32]["status"] == "running"
    assert store.run_rows["b" * 32]["status"] == "running"
    assert store.run_rows["e" * 32]["status"] == "running"


def test_reap_orphan_runs_accepts_iso_string_and_datetime(monkeypatch):
    """``started_at`` may arrive as either an ISO string (MySQL driver) or a
    datetime (in-memory fake); both must be parsed as UTC and aged correctly.
    A trailing ``Z`` suffix (common from some drivers) must be tolerated."""
    from rca_agent.server.app import _reap_orphan_runs

    monkeypatch.setenv("RCA_RUN_REAP_MIN", "0.01")
    store = FakeTraceStore()
    old_str = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    old_str_z = (
        datetime.now(UTC) - timedelta(minutes=30) - timedelta(days=1)
    ).isoformat().replace("+00:00", "Z")
    old_dt = datetime.now(UTC) - timedelta(minutes=30)
    store.run_rows = {
        "a" * 32: {"run_id": "a" * 32, "status": "running",
                   "started_at": old_str},
        "b" * 32: {"run_id": "b" * 32, "status": "running",
                   "started_at": old_str_z},
        "c" * 32: {"run_id": "c" * 32, "status": "running",
                   "started_at": old_dt},
    }
    closed = _reap_orphan_runs(store)
    assert closed == 3
    for rid in store.run_rows:
        assert store.run_rows[rid]["status"] == "interrupted"


def test_reap_orphan_runs_returns_zero_when_list_raises(monkeypatch):
    """A store whose ``list_runs`` blows up must not propagate; the reaper
    returns 0 and the app stays up."""
    from rca_agent.server.app import _reap_orphan_runs

    class BrokenStore:
        def list_runs(self, case_id=None, limit=50):
            raise RuntimeError("db down")

    assert _reap_orphan_runs(BrokenStore()) == 0  # type: ignore[arg-type]


def test_reap_orphan_runs_skips_rows_whose_finish_run_raises(monkeypatch):
    """A row whose ``finish_run`` blows up is skipped; the sweep continues and
    closes the next valid row."""
    from rca_agent.server.app import _reap_orphan_runs

    monkeypatch.setenv("RCA_RUN_REAP_MIN", "0.01")

    class FlakeyStore:
        def __init__(self):
            self.calls = []

        def list_runs(self, case_id=None, limit=50):
            now = datetime.now(UTC)
            return [
                {"run_id": "x" * 32, "status": "running",
                 "started_at": now - timedelta(minutes=5)},
                {"run_id": "y" * 32, "status": "running",
                 "started_at": now - timedelta(minutes=5)},
            ]

        def finish_run(self, run_id, status, token_usage=None):
            self.calls.append(run_id)
            if run_id == "x" * 32:
                raise RuntimeError("transient")

    store = FlakeyStore()
    closed = _reap_orphan_runs(store)
    # Only the second row closed (first raised); the sweep was not aborted.
    assert closed == 1
    assert store.calls == ["x" * 32, "y" * 32]


def test_lifespan_starts_reaper_and_cleans_up():
    """The FastAPI lifespan must start the periodic reaper on startup and
    cancel it cleanly on shutdown (no leaked task, no crash if the store is
    unavailable). Drives the lifespan context directly."""
    import asyncio

    from rca_agent.server.app import lifespan

    # Inject a fake trace store so the startup sweep + periodic loop run
    # against in-memory data, not a real DB.
    fake = FakeTraceStore()
    now = datetime.now(UTC)
    # A stale running row the startup sweep should close immediately.
    fake.run_rows = {"z" * 32: {"run_id": "z" * 32, "status": "running",
                                "started_at": now - timedelta(minutes=30)}}
    set_trace_store_factory(lambda: fake)
    try:
        async def run():
            # Fast reaper interval so the test doesn't hang on sleep.
            import os

            os.environ["RCA_RUN_REAP_INTERVAL_SEC"] = "0.01"
            os.environ["RCA_RUN_REAP_MIN"] = "0.01"
            async with lifespan(fastapi_app):
                # Startup sweep already ran synchronously inside __aenter__.
                pass
            os.environ.pop("RCA_RUN_REAP_INTERVAL_SEC", None)
            os.environ.pop("RCA_RUN_REAP_MIN", None)

        asyncio.run(run())
    finally:
        set_trace_store_factory(None)
    # The stale row was reaped at startup.
    assert fake.run_rows["z" * 32]["status"] == "interrupted"


def test_lifespan_does_not_crash_when_store_unavailable():
    """If the trace store can't be constructed (dev with no DB), lifespan must
    still start the app — the reaper is best-effort."""
    import asyncio

    from rca_agent.server.app import lifespan

    def exploding():
        raise RuntimeError("no db")

    set_trace_store_factory(exploding)
    try:
        async def run():
            async with lifespan(fastapi_app):
                pass

        asyncio.run(run())  # must not raise
    finally:
        set_trace_store_factory(None)
