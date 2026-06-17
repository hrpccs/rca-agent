"""Tests for rca_agent.providers.clickhouse_provider.

The unit tests below stub the ClickHouse client so they run without the live
DB; the mapping/SQL-building/render logic is what matters here. A separate
``@pytest.mark.live`` test exercises the real DB end-to-end.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from rca_agent.contracts import (
    AlertFilter,
    CloudEvent,
    DataProvider,
    EventFilter,
    LogFilter,
    MetricFilter,
    Modality,
    Span,
    TimeWindow,
    TopologyFilter,
    TraceFilter,
)
from rca_agent.providers import clickhouse_provider as mod
from rca_agent.providers.clickhouse_provider import ClickhouseProvider


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def window() -> TimeWindow:
    return TimeWindow(
        start=datetime(2026, 4, 25, 5, 18, 12, tzinfo=UTC),
        end=datetime(2026, 4, 25, 5, 28, 12, tzinfo=UTC),
        start_us=1777094292716735,
        end_us=1777094892716735,
    )


class _FakeResult:
    def __init__(self, rows: list[tuple]) -> None:
        self.result_rows = rows


class _FakeClient:
    """Records the last query/params and returns canned rows."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, list[Any] | None]] = []
        self.next_rows: list[tuple] = []
        self.command_calls: list[str] = []

    def query(self, sql: str, params: list[Any] | None = None) -> _FakeResult:
        self.queries.append((sql, params))
        return _FakeResult(list(self.next_rows))

    def command(self, stmt: str) -> None:
        self.command_calls.append(stmt)


def _make_provider(rows: list[tuple] | None = None) -> tuple[ClickhouseProvider, _FakeClient]:
    fake = _FakeClient()
    fake.next_rows = rows or []
    p = ClickhouseProvider.__new__(ClickhouseProvider)
    p.case_id = "t001"
    p._client = fake  # type: ignore[attr-defined]
    return p, fake


def _param_ids(params: list[Any] | None) -> list[str]:
    """Entity queries are `WHERE case_id=%s AND id IN (%s,...)`; return the ids
    (i.e. all params except the leading case_id)."""
    if not params:
        return []
    return list(params[1:])


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_safe_json_parses_dict():
    assert mod._safe_json('{"a": 1}') == {"a": 1}


def test_safe_json_empty_and_garbage_returns_empty():
    assert mod._safe_json(None) == {}
    assert mod._safe_json("") == {}
    assert mod._safe_json("not json") == {}
    assert mod._safe_json("[1,2]") == {"value": [1, 2]}


def test_window_us_prefers_us_fields(window):
    assert mod._window_us(window) == (window.start_us, window.end_us)


def test_window_us_falls_back_to_datetime():
    w = TimeWindow(
        start=datetime(2026, 4, 25, 0, 0, 0, tzinfo=UTC),
        end=datetime(2026, 4, 26, 0, 0, 0, tzinfo=UTC),
    )
    s, e = mod._window_us(w)
    assert s == int(w.start.timestamp() * 1_000_000)
    assert e == int(w.end.timestamp() * 1_000_000)


def test_window_ns_is_micros_times_1000(window):
    s, e = mod._window_ns(window)
    assert s == window.start_us * 1000
    assert e == window.end_us * 1000


def test_in_clause_none_returns_none():
    ph, p = mod._in_clause(None)
    assert ph is None and p == []
    ph, p = mod._in_clause([])
    assert ph is None and p == []


def test_in_clause_builds_placeholders():
    ph, p = mod._in_clause(["a", "b"])
    assert ph == "%s, %s"
    assert p == ["a", "b"]


def test_split_sql_statements_skips_comment_only():
    stmts = mod._split_sql_statements(
        "-- header comment\nCREATE TABLE x (a Int8);\n-- mid\n\nSELECT 1;"
    )
    assert len(stmts) == 2
    assert "header comment" not in stmts[0]
    assert stmts[1].startswith("SELECT")


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #
def test_provider_satisfies_protocol(window):
    p, _ = _make_provider()
    assert isinstance(p, DataProvider)
    assert {m.value for m in p.modalities()} == {m.value for m in Modality}


def test_modalities_returns_all_six():
    p, _ = _make_provider()
    assert len(p.modalities()) == 6


# --------------------------------------------------------------------------- #
# query_metrics: grouping + window micros + parameterisation
# --------------------------------------------------------------------------- #
def test_query_metrics_groups_into_series(window):
    rows = [
        ("e1", "svc1", "apm pods", "checkout", "apm", "qps", "ms1", 1000, 1.0),
        ("e1", "svc1", "apm pods", "checkout", "apm", "qps", "ms1", 2000, 2.0),
        ("e2", "svc2", "k8s pod", "db", "k8s", "cpu", "ms2", 1000, 0.5),
    ]
    p, fake = _make_provider(rows)
    out = p.query_metrics(MetricFilter(window=window, metrics=["qps"]))
    assert len(out) == 2
    qps = next(s for s in out if s.metric == "qps")
    assert qps.entity_id == "e1"
    assert qps.points == [(1000, 1.0), (2000, 2.0)]
    # window micros are parameterised, never f-stringed into SQL
    sql, params = fake.queries[0]
    assert "case_id = %s" in sql and "time >= %s" in sql
    assert params[0] == "t001"
    assert params[1] == window.start_us


def test_query_metrics_empty_returns_empty(window):
    p, _ = _make_provider([])
    assert p.query_metrics(MetricFilter(window=window)) == []


def test_query_metrics_in_clause_filters(window):
    p, fake = _make_provider([])
    p.query_metrics(
        MetricFilter(window=window, entity_ids=["a", "b"], domains=["apm"], metrics=["qps"])
    )
    sql, params = fake.queries[0]
    assert "entity_id IN (%s, %s)" in sql
    assert "domain IN (%s)" in sql
    assert "a" in params and "b" in params and "apm" in params


# --------------------------------------------------------------------------- #
# query_logs
# --------------------------------------------------------------------------- #
def test_query_logs_maps_row(window):
    rows = [
        ("err boom", datetime(2026, 4, 25, 5, 20, tzinfo=UTC), "pod1", "ns", "ctr", "h1"),
    ]
    p, fake = _make_provider(rows)
    out = p.query_logs(LogFilter(window=window, contains="boom"))
    assert len(out) == 1
    assert out[0].content == "err boom"
    assert out[0].pod == "pod1" and out[0].host == "h1"
    sql, params = fake.queries[0]
    assert "positionCaseInsensitive(content, %s) > 0" in sql
    assert "boom" in params


def test_query_logs_empty(window):
    p, _ = _make_provider([])
    assert p.query_logs(LogFilter(window=window)) == []


# --------------------------------------------------------------------------- #
# query_traces
# --------------------------------------------------------------------------- #
def test_query_traces_groups_spans(window):
    res = '{"k":"v"}'
    rows = [
        ("t1", "s1", "", "SERVER", "doX", 1, 2, 100, "svc", "OK", "", res, "{}"),
        ("t1", "s2", "s1", "CLIENT", "doY", 1, 2, 50, "svc", "ERROR", "boom", "{}", "{}"),
    ]
    p, _ = _make_provider(rows)
    out = p.query_traces(TraceFilter(window=window))
    assert len(out) == 1
    tr = out[0]
    assert tr.trace_id == "t1"
    assert len(tr.spans) == 2
    err = next(s for s in tr.spans if s.span_id == "s2")
    assert err.status_code == "ERROR"
    ok = next(s for s in tr.spans if s.span_id == "s1")
    assert ok.resources == {"k": "v"}
    assert err.resources == {}  # s2 row had resources "{}"
    assert tr.slowest_span().span_id == "s1"  # 100ns > 50ns


def test_query_traces_empty(window):
    p, _ = _make_provider([])
    assert p.query_traces(TraceFilter(window=window)) == []


def test_query_traces_min_duration(window):
    p, fake = _make_provider([])
    p.query_traces(TraceFilter(window=window, min_duration_ns=5000))
    sql, params = fake.queries[0]
    assert "duration >= %s" in sql
    assert 5000 in params


# --------------------------------------------------------------------------- #
# query_events
# --------------------------------------------------------------------------- #
def test_query_events_parses_event_id_json(window):
    event_id = '{"reason":"BackOff","message":"stop","metadata":{"name":"p.x"}}'
    rows = [(event_id, "h1", "Warning", "p1", "c1", datetime(2026, 4, 25, tzinfo=UTC))]
    p, _ = _make_provider(rows)
    out = p.query_events(EventFilter(window=window, levels=["Warning"]))
    assert len(out) == 1
    ev = out[0]
    assert ev.reason == "BackOff"
    assert ev.message == "stop"
    assert ev.metadata["metadata"]["name"] == "p.x"
    assert ev.level == "Warning"


def test_query_events_empty(window):
    p, _ = _make_provider([])
    assert p.query_events(EventFilter(window=window)) == []


# --------------------------------------------------------------------------- #
# query_alerts
# --------------------------------------------------------------------------- #
def test_query_alerts_parses_json_columns(window):
    rows = [
        (
            "id1", "alert", "sub", "CRITICAL", "Alarm", "checkout",
            datetime(2026, 4, 25, tzinfo=UTC),
            '{"entity":{"id":"x"}}', '{"a":"b"}', '{"ann":1}', '{"detailValue":[1]}',
        )
    ]
    p, _ = _make_provider(rows)
    out = p.query_alerts(AlertFilter(window=window, severities=["CRITICAL"]))
    assert len(out) == 1
    a: CloudEvent = out[0]
    assert a.id == "id1" and a.severity == "CRITICAL"
    assert a.resource == {"entity": {"id": "x"}}
    assert a.labels == {"a": "b"}
    assert a.data == {"detailValue": [1]}


def test_query_alerts_empty(window):
    p, _ = _make_provider([])
    assert p.query_alerts(AlertFilter(window=window)) == []


# --------------------------------------------------------------------------- #
# query_topology
# --------------------------------------------------------------------------- #
def test_query_topology_exact_seeds_hops0(window):
    all_ents = {
        "e1": ("e1", "svc", "checkout", "", "", "{}"),
        "e2": ("e2", "svc", "db", "", "", "{}"),
    }
    edges = [("e1", "svc", "e2", "svc", "calls")]
    fake = _FakeClient()

    def query(sql, params=None):
        fake.queries.append((sql, params))
        # entity query filters by id IN (...); edge query returns all.
        if "topology_entities" in sql:
            ids = _param_ids(params)
            return _FakeResult([all_ents[i] for i in ids if i in all_ents])
        return _FakeResult(list(edges))

    fake.query = query  # type: ignore[assignment]
    p = ClickhouseProvider.__new__(ClickhouseProvider)
    p.case_id = "t001"
    p._client = fake  # type: ignore[attr-defined]
    sub = p.query_topology(TopologyFilter(entity_ids=["e1"], hops=0))
    assert {e["id"] for e in sub.entities} == {"e1"}
    # hops=0 → no neighborhood expansion → e2 not in entities, edge pruned
    assert sub.edges == []


def test_query_topology_neighborhood_expansion(window):
    all_ents = {
        "e1": ("e1", "svc", "checkout", "", "", "{}"),
        "e2": ("e2", "svc", "db", "", "", "{}"),
        "e3": ("e3", "svc", "cache", "", "", "{}"),
    }
    edges = [
        ("e1", "svc", "e2", "svc", "calls"),
        ("e2", "svc", "e3", "svc", "calls"),
    ]
    fake = _FakeClient()

    def query(sql, params=None):
        fake.queries.append((sql, params))
        if "topology_entities" in sql:
            ids = _param_ids(params)
            return _FakeResult([all_ents[i] for i in ids if i in all_ents])
        return _FakeResult(list(edges))

    fake.query = query  # type: ignore[assignment]
    p = ClickhouseProvider.__new__(ClickhouseProvider)
    p.case_id = "t001"
    p._client = fake  # type: ignore[attr-defined]
    sub = p.query_topology(TopologyFilter(entity_ids=["e1"], hops=1))
    ids = {e["id"] for e in sub.entities}
    assert "e1" in ids and "e2" in ids
    assert "e3" not in ids  # only 1 hop
    # edge e1->e2 retained (both in reach), e2->e3 pruned (e3 not in reach)
    assert len(sub.edges) == 1


def test_query_topology_empty(window):
    p, _ = _make_provider([])
    sub = p.query_topology(TopologyFilter())
    assert sub.entities == [] and sub.edges == []


# --------------------------------------------------------------------------- #
# render_* helpers
# --------------------------------------------------------------------------- #
def test_render_empty_helpers():
    p, _ = _make_provider()
    assert "(none)" in p.render_metrics([])
    assert "(none)" in p.render_logs([])
    assert "(none)" in p.render_traces([])
    assert "(none)" in p.render_events([])
    assert "(none)" in p.render_alerts([])
    assert "(none)" in p.render_topology(__import__("rca_agent.contracts", fromlist=["TopologySubgraph"]).TopologySubgraph())


def test_render_metrics_includes_stats():
    from rca_agent.contracts import MetricSeries

    p, _ = _make_provider()
    ms = MetricSeries(entity_id="e1", entity_name="svc1", metric="qps", points=[(1, 2.0), (2, 4.0)])
    txt = p.render_metrics([ms])
    assert "qps" in txt and "svc1" in txt
    assert "min=" in txt and "max=" in txt


def test_render_traces_slowest():
    p, _ = _make_provider()
    tr = __import__("rca_agent.contracts", fromlist=["Trace"]).Trace(
        trace_id="t1",
        spans=[Span(trace_id="t1", span_id="s1", name="x", duration_ns=1_000_000)],
    )
    txt = p.render_traces([tr])
    assert "t1" in txt and "1.0ms" in txt


# --------------------------------------------------------------------------- #
# from_case / __init__
# --------------------------------------------------------------------------- #
def test_from_case_returns_provider_instance(monkeypatch):
    captured = {}

    class _FakeConn:
        def __call__(self, **kw):
            captured.update(kw)
            return _FakeClient()

    monkeypatch.setattr(mod.clickhouse_connect, "get_client", _FakeConn())
    p = ClickhouseProvider.from_case("t042")
    assert isinstance(p, ClickhouseProvider)
    assert p.case_id == "t042"
    assert captured["database"] == "rca"
    # query/socket timeout applied at client creation (default).
    assert captured["send_receive_timeout"] == 30


# --------------------------------------------------------------------------- #
# Query timeout (env RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC)
# --------------------------------------------------------------------------- #
def test_query_timeout_default_is_30():
    assert mod._query_timeout_sec() == 30


def test_query_timeout_env_override(monkeypatch):
    monkeypatch.setenv("RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC", "11")
    assert mod._query_timeout_sec() == 11


def test_query_timeout_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC", "not-a-number")
    assert mod._query_timeout_sec() == 30
    monkeypatch.setenv("RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC", "-5")
    assert mod._query_timeout_sec() == 30


def test_query_timeout_passed_to_client_factory(monkeypatch):
    captured = {}

    def fake_factory(**kw):
        captured.update(kw)
        return _FakeClient()

    monkeypatch.setenv("RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC", "42")
    ClickhouseProvider("t001", client_factory=fake_factory)
    assert captured["send_receive_timeout"] == 42


def test_query_timeout_respects_dsn_override(monkeypatch):
    # An explicit send_receive_timeout in the dsn must NOT be clobbered.
    captured = {}

    def fake_factory(**kw):
        captured.update(kw)
        return _FakeClient()

    monkeypatch.setenv("RCA_CLICKHOUSE_QUERY_TIMEOUT_SEC", "42")
    ClickhouseProvider(
        "t001",
        dsn={"host": "h", "port": 8123, "send_receive_timeout": 7},
        client_factory=fake_factory,
    )
    assert captured["send_receive_timeout"] == 7


# --------------------------------------------------------------------------- #
# Client injection (constructor + factory)
# --------------------------------------------------------------------------- #
def test_constructor_accepts_injected_client(window):
    fake = _FakeClient()
    p = ClickhouseProvider("t001", client=fake)
    assert p._client is fake
    # Injected client bypasses factory entirely — no env/DSN read needed.
    p.query_logs(LogFilter(window=window))
    assert len(fake.queries) == 1


def test_constructor_uses_factory_when_no_client():
    calls = []

    def fake_factory(**kw):
        calls.append(kw)
        return _FakeClient()

    p = ClickhouseProvider("t001", client_factory=fake_factory)
    assert len(calls) == 1
    assert calls[0]["database"] == "rca"
    assert isinstance(p._client, _FakeClient)


# --------------------------------------------------------------------------- #
# Connection-error resilience
# --------------------------------------------------------------------------- #
class _ExplodingClient:
    """A client whose .query always raises (simulates CH down / socket error)."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def query(self, sql: str, params: list[Any] | None = None):
        self.calls += 1
        raise self.exc


def _explode_provider(exc: Exception) -> ClickhouseProvider:
    return ClickhouseProvider("t001", client=_ExplodingClient(exc))


def test_query_error_returns_empty_not_none(window, caplog):
    p = _explode_provider(ConnectionError("refused"))
    with caplog.at_level("ERROR"):
        out = p.query_logs(LogFilter(window=window))
    assert out == []  # empty, not None, not raised
    # structural log emitted
    assert any("clickhouse.query_failed" in r.message or "query_failed" in r.message
               for r in caplog.records)
    rec = next(r for r in caplog.records if "query_failed" in r.message)
    assert rec.case_id == "t001"  # type: ignore[attr-defined]


def test_query_error_all_modalities_survive(window, caplog):
    """Each modality swallows the error and returns an empty result."""
    p = _explode_provider(TimeoutError("read timeout"))
    with caplog.at_level("ERROR"):
        assert p.query_metrics(MetricFilter(window=window)) == []
        assert p.query_logs(LogFilter(window=window)) == []
        assert p.query_traces(TraceFilter(window=window)) == []
        assert p.query_events(EventFilter(window=window)) == []
        assert p.query_alerts(AlertFilter(window=window)) == []
    sub = p.query_topology(TopologyFilter())
    assert sub.entities == [] and sub.edges == []
    # At least one structural failure log per modality (5 list-returning +
    # topology, which issues 1+ queries). Assert presence, not exact count,
    # so the test isn't coupled to topology's internal query branching.
    fails = [r for r in caplog.records if "query_failed" in r.message]
    assert len(fails) >= 6


def test_query_error_logs_error_type_and_message(window, caplog):
    p = _explode_provider(OSError("connection reset"))
    with caplog.at_level("ERROR"):
        p.query_alerts(AlertFilter(window=window))
    rec = next(r for r in caplog.records if "query_failed" in r.message)
    assert rec.error_type == "OSError"  # type: ignore[attr-defined]
    assert "connection reset" in rec.error  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Renderers: full coverage for all 6 modalities with non-empty data
# --------------------------------------------------------------------------- #
def test_render_logs_non_empty():
    from rca_agent.contracts import LogLine

    p, _ = _make_provider()
    txt = p.render_logs([LogLine(
        ts=datetime(2026, 4, 25, 5, 20, tzinfo=UTC),
        pod="api-1", content="boom",
    )])
    assert "## Logs" in txt
    assert "api-1" in txt and "boom" in txt


def test_render_events_non_empty():
    from rca_agent.contracts import K8sEvent

    p, _ = _make_provider()
    txt = p.render_events([K8sEvent(
        ts=datetime(2026, 4, 25, 5, 20, tzinfo=UTC),
        level="Warning", pod="api-1", reason="BackOff", message="stopping",
    )])
    assert "## Events" in txt
    assert "Warning" in txt and "BackOff" in txt and "stopping" in txt


def test_render_alerts_non_empty():
    from rca_agent.contracts import CloudEvent

    p, _ = _make_provider()
    txt = p.render_alerts([CloudEvent(
        id="a1", type="alert", severity="CRITICAL", subtype="sub",
        subject="checkout", status="Alarm", ts=datetime(2026, 4, 25, tzinfo=UTC),
        data={"detail": "x"},
    )])
    assert "## Alerts" in txt
    assert "CRITICAL" in txt and "checkout" in txt and "Alarm" in txt
    assert "detail" in txt  # data JSON rendered


def test_render_topology_non_empty():
    from rca_agent.contracts import TopologySubgraph

    p, _ = _make_provider()
    sub = TopologySubgraph(
        entities=[{"id": "e1", "type": "svc", "name": "checkout"}],
        edges=[{"src": "e1", "src_type": "svc", "dst": "e2", "dst_type": "svc", "relation": "calls"}],
    )
    txt = p.render_topology(sub)
    assert "## Topology" in txt
    assert "Entities:" in txt and "Edges:" in txt
    assert "checkout" in txt and "calls" in txt


# --------------------------------------------------------------------------- #
# Live end-to-end (gated; needs running ClickHouse)
# --------------------------------------------------------------------------- #
@pytest.mark.live
def test_live_provider_roundtrip(window):
    """Hit the real DB; only runs with RCA_DEEPSEEK_API_KEY + infra up."""
    p = ClickhouseProvider("t001")
    p.ensure_schema()
    # Empty queries must not raise and must return [].
    assert p.query_alerts(AlertFilter(window=window)) == []
    assert p.query_logs(LogFilter(window=window)) == []
    assert isinstance(p.query_topology(TopologyFilter()), type(p.query_topology(TopologyFilter())))
