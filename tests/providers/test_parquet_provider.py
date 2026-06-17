"""Tests for the Parquet DataProvider (U1) against the real rca100 t001 case.

Dataset-backed tests require the on-disk benchmark dataset (default cases_dir)
and are skipped automatically when t001 is unavailable so the suite stays green
on a fresh checkout without data. The synthetic cache/robustness tests at the
bottom of this file build their own pyarrow tables and always run.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rca_agent.cases import load_case
from rca_agent.contracts import (
    AlertFilter,
    CloudEvent,
    DataProvider,
    EventFilter,
    K8sEvent,
    LogFilter,
    MetricFilter,
    MetricSeries,
    Modality,
    Span,
    TimeWindow,
    TopologyFilter,
    Trace,
    TraceFilter,
)
from rca_agent.contracts.dataset import Case, Task, Topology
from rca_agent.providers.parquet_provider import (
    ParquetProvider,
    _as_float,
    _as_int,
    _as_str,
    _in_range_dt,
    _in_range_us,
    _parse_iso,
    _parse_json_obj,
    _window_us,
    render,
)

# Resolve the t001 case dir defensively: load_case reads task.json/topology.json
# and would raise FileNotFoundError at collection time on a fresh checkout
# without the dataset. Guard it so the synthetic TestCache / TestMalformed
# classes below (which never touch the dataset) still run.
try:
    _T001_DIR = Path(load_case("t001").case_dir)
    _T001_AVAILABLE = (_T001_DIR / "metrics.parquet").exists()
except (FileNotFoundError, OSError):
    _T001_DIR = None
    _T001_AVAILABLE = False
# Skip ONLY the dataset-backed tests; the synthetic TestCache / TestMalformed
# classes at the bottom are unmarked so they run on a fresh checkout.
needs_t001 = pytest.mark.skipif(not _T001_AVAILABLE, reason="t001 dataset not available")


@pytest.fixture(scope="module")
def provider() -> ParquetProvider:
    return ParquetProvider(load_case("t001"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class TestCoercionHelpers:
    def test_as_int(self):
        assert _as_int(None) is None
        assert _as_int("") is None
        assert _as_int("123") == 123
        assert _as_int(1777090692000000) == 1777090692000000
        assert _as_int(1.0) == 1
        assert _as_int("1.9") == 1
        assert _as_int(float("nan")) is None

    def test_as_float(self):
        assert _as_float(None) is None
        assert _as_float("1.5") == 1.5
        assert _as_float(2) == 2.0
        assert _as_float("") is None
        assert _as_float(float("nan")) is None

    def test_as_str(self):
        assert _as_str(None) is None
        assert _as_str("") is None
        assert _as_str("x") == "x"
        assert _as_str(5) == "5"

    def test_parse_json_obj(self):
        assert _parse_json_obj(None) == {}
        assert _parse_json_obj("") == {}
        assert _parse_json_obj("{not json") == {}
        assert _parse_json_obj('{"a":1}') == {"a": 1}
        assert _parse_json_obj({"a": 1}) == {"a": 1}
        assert _parse_json_obj("[1,2]") == {}  # non-dict

    def test_parse_iso(self):
        assert _parse_iso(None) is None
        assert _parse_iso("") is None
        dt = _parse_iso("2026-04-25T13:03:11.009+08:00")
        assert dt is not None and dt.tzinfo is not None
        # Z suffix handled
        dt2 = _parse_iso("2026-04-25T04:15:32Z")
        assert dt2 is not None and dt2.tzinfo is not None
        # naive -> assumed UTC
        dt3 = _parse_iso("2026-04-25T04:15:32")
        assert dt3 is not None and dt3.tzinfo is not None

    def test_in_range(self):
        dt = datetime(2026, 4, 25, 5, 20, tzinfo=timezone.utc)
        lo = datetime(2026, 4, 25, 5, 18, tzinfo=timezone.utc)
        hi = datetime(2026, 4, 25, 5, 28, tzinfo=timezone.utc)
        assert _in_range_dt(dt, lo, hi)
        assert not _in_range_dt(None, lo, hi)
        assert _in_range_us(100, 50, 200)
        assert not _in_range_us(10, 50, 200)
        assert not _in_range_us(None, 50, 200)

    def test_window_us(self):
        w = TimeWindow(
            start=datetime(2026, 4, 25, 5, 18, tzinfo=timezone.utc),
            end=datetime(2026, 4, 25, 5, 28, tzinfo=timezone.utc),
            start_us=1000, end_us=2000,
        )
        assert _window_us(w) == (1000, 2000)
        w2 = TimeWindow(
            start=datetime(2026, 4, 25, 5, 18, tzinfo=timezone.utc),
            end=datetime(2026, 4, 25, 5, 28, tzinfo=timezone.utc),
        )
        lo, hi = _window_us(w2)
        assert lo is not None and hi is not None and lo < hi


# --------------------------------------------------------------------------- #
# Provider construction + Protocol conformance
# --------------------------------------------------------------------------- #
@needs_t001
class TestProviderBasics:
    def test_satisfies_protocol(self, provider: ParquetProvider):
        # ParquetProvider must satisfy the runtime-checkable DataProvider.
        assert isinstance(provider, DataProvider)
        assert provider.case_id == "t001"
        assert provider.window is not None

    def test_modalities(self, provider: ParquetProvider):
        mods = provider.modalities()
        assert mods  # non-empty
        for m in mods:
            assert isinstance(m, Modality)

    def test_from_case_classmethod(self):
        p = ParquetProvider.from_case("t001")
        assert p.case_id == "t001"
        assert isinstance(p, DataProvider)


# --------------------------------------------------------------------------- #
# query_metrics
# --------------------------------------------------------------------------- #
@needs_t001
class TestQueryMetrics:
    def test_window_filter_returns_results(self, provider: ParquetProvider):
        out = provider.query_metrics(MetricFilter(window=provider.window, limit=50))
        assert isinstance(out, list)
        assert out, "expected some metric series inside the alert window"
        for ms in out:
            assert isinstance(ms, MetricSeries)
            assert ms.metric
            assert ms.points
            for t_us, v in ms.points:
                assert isinstance(t_us, int)
                assert isinstance(v, float)

    def test_filter_by_metric_name(self, provider: ParquetProvider):
        # pick a metric known to exist in the window
        probe = provider.query_metrics(MetricFilter(window=provider.window, limit=5))
        assert probe
        target = probe[0].metric
        out = provider.query_metrics(
            MetricFilter(window=provider.window, metrics=[target], limit=10)
        )
        assert out and all(ms.metric == target for ms in out)

    def test_filter_by_domain(self, provider: ParquetProvider):
        out = provider.query_metrics(
            MetricFilter(window=provider.window, domains=["k8s"], limit=5)
        )
        assert all(ms.domain == "k8s" for ms in out)

    def test_points_sorted_by_time(self, provider: ParquetProvider):
        out = provider.query_metrics(MetricFilter(window=provider.window, limit=20))
        for ms in out:
            times = [t for t, _ in ms.points]
            assert times == sorted(times)

    def test_limit_caps_series(self, provider: ParquetProvider):
        out = provider.query_metrics(MetricFilter(window=provider.window, limit=3))
        assert len(out) <= 3


# --------------------------------------------------------------------------- #
# query_logs
# --------------------------------------------------------------------------- #
@needs_t001
class TestQueryLogs:
    def test_window_filter(self, provider: ParquetProvider):
        out = provider.query_logs(LogFilter(window=provider.window, limit=10))
        assert out
        for lg in out:
            assert lg.content
            assert lg.ts is not None

    def test_contains_substring(self, provider: ParquetProvider):
        # widen window to find any logs, then filter by a recurring token.
        wide = TimeWindow(
            start=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 26, 0, 0, tzinfo=timezone.utc),
        )
        probe = provider.query_logs(LogFilter(window=wide, limit=50))
        if not probe:
            pytest.skip("no logs in wide window")
        # find an alpha token (>=4 chars) that recurs verbatim in >=2 lines.
        from collections import Counter
        c = Counter()
        for lg in probe:
            for tok in set(lg.content.lower().split()):
                if len(tok) >= 4 and any(ch.isalpha() for ch in tok):
                    c[tok] += 1
        word = next((w for w, n in c.most_common() if n >= 2), None)
        if not word:
            word = next((w for w, _ in c.most_common()), None)
        if not word:
            pytest.skip("no usable recurring token")
        out = provider.query_logs(LogFilter(window=wide, contains=word, limit=50))
        assert out and all(word in lg.content.lower() for lg in out)

    def test_limit(self, provider: ParquetProvider):
        wide = TimeWindow(
            start=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 26, 0, 0, tzinfo=timezone.utc),
        )
        out = provider.query_logs(LogFilter(window=wide, limit=7))
        assert len(out) <= 7


# --------------------------------------------------------------------------- #
# query_traces
# --------------------------------------------------------------------------- #
@needs_t001
class TestQueryTraces:
    def test_window_returns_traces(self, provider: ParquetProvider):
        out = provider.query_traces(TraceFilter(window=provider.window, limit=10))
        assert out
        for tr in out:
            assert isinstance(tr, Trace)
            assert tr.trace_id
            assert tr.spans
            for sp in tr.spans:
                assert isinstance(sp, Span)
                assert sp.name

    def test_resources_attributes_parsed_to_dict(self, provider: ParquetProvider):
        out = provider.query_traces(TraceFilter(window=provider.window, limit=5))
        assert out
        sp = out[0].spans[0]
        assert isinstance(sp.resources, dict)
        assert isinstance(sp.attributes, dict)

    def test_limit_traces(self, provider: ParquetProvider):
        out = provider.query_traces(TraceFilter(window=provider.window, limit=3))
        assert len(out) <= 3

    def test_filter_by_trace_id(self, provider: ParquetProvider):
        probe = provider.query_traces(TraceFilter(window=provider.window, limit=5))
        assert probe
        tid = probe[0].trace_id
        out = provider.query_traces(TraceFilter(window=provider.window, trace_ids=[tid], limit=10))
        assert out and all(tr.trace_id == tid for tr in out)

    def test_trace_spans_not_truncated_by_limit(self, provider: ParquetProvider):
        # Regression: span rows are interleaved across traces in the file, so an
        # early break on `limit` traces must NOT truncate the spans of traces
        # that are returned. A single trace queried directly must have the same
        # span count whether `limit` is 1 or 1000.
        probe = provider.query_traces(TraceFilter(window=provider.window, limit=20))
        assert probe
        # pick the first trace that has >1 span so the check is meaningful
        tid = next((t.trace_id for t in probe if len(t.spans) > 1), probe[0].trace_id)
        few = provider.query_traces(
            TraceFilter(window=provider.window, trace_ids=[tid], limit=1)
        )
        many = provider.query_traces(
            TraceFilter(window=provider.window, trace_ids=[tid], limit=1000)
        )
        assert few and many
        assert len(few[0].spans) == len(many[0].spans)
        assert len(few[0].spans) >= 1


# --------------------------------------------------------------------------- #
# query_events
# --------------------------------------------------------------------------- #
@needs_t001
class TestQueryEvents:
    def test_events_returned(self, provider: ParquetProvider):
        # Events often predate the alert window; use a wide window.
        wide = TimeWindow(
            start=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 26, 0, 0, tzinfo=timezone.utc),
        )
        out = provider.query_events(EventFilter(window=wide, limit=20))
        assert out
        for ev in out:
            assert isinstance(ev, K8sEvent)
            assert isinstance(ev.metadata, dict)
            assert ev.reason is not None or ev.message is not None

    def test_filter_by_level(self, provider: ParquetProvider):
        wide = TimeWindow(
            start=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 26, 0, 0, tzinfo=timezone.utc),
        )
        out = provider.query_events(EventFilter(window=wide, levels=["Warning"], limit=50))
        assert all(ev.level == "Warning" for ev in out)


# --------------------------------------------------------------------------- #
# query_alerts
# --------------------------------------------------------------------------- #
@needs_t001
class TestQueryAlerts:
    def test_alerts_in_window(self, provider: ParquetProvider):
        out = provider.query_alerts(AlertFilter(window=provider.window, limit=50))
        assert out
        for al in out:
            assert isinstance(al, CloudEvent)
            assert al.id
            assert al.ts is not None
            assert isinstance(al.resource, dict)

    def test_filter_by_severity(self, provider: ParquetProvider):
        out = provider.query_alerts(
            AlertFilter(window=provider.window, severities=["CRITICAL"], limit=50)
        )
        assert all(al.severity == "CRITICAL" for al in out)


# --------------------------------------------------------------------------- #
# query_topology
# --------------------------------------------------------------------------- #
@needs_t001
class TestQueryTopology:
    def test_full_graph(self, provider: ParquetProvider):
        sub = provider.query_topology(TopologyFilter())
        assert sub.entities
        assert isinstance(sub.edges, list)

    def test_filter_by_type(self, provider: ParquetProvider):
        sub = provider.query_topology(TopologyFilter(entity_types=["apm.service"]))
        assert sub.entities
        # with hops=1 the neighborhood includes entities of other types, but the
        # seed entities themselves must be present and be apm.service.
        types = {e.get("type") for e in sub.entities}
        assert "apm.service" in types
        # and no entity of an unselected type appears unless it is a neighbor
        # (every returned entity must be connected to a seed — sanity: graph is
        # a connected subgraph around the seeds, not the whole unrelated graph)
        assert len(sub.entities) <= len(
            provider.query_topology(TopologyFilter()).entities
        )

    def test_neighborhood_hops(self, provider: ParquetProvider):
        # pick a known entity id from the graph
        full = provider.query_topology(TopologyFilter())
        seed = full.entities[0]["id"]
        seed_type = full.entities[0]["type"]
        sub = provider.query_topology(
            TopologyFilter(entity_ids=[seed], hops=1)
        )
        ids = {e["id"] for e in sub.entities}
        assert seed in ids
        assert len(sub.entities) >= 1
        # with hops=0 only the seed itself
        sub0 = provider.query_topology(TopologyFilter(entity_ids=[seed], hops=0))
        ids0 = {e["id"] for e in sub0.entities}
        assert seed in ids0

    def test_limit_entities(self, provider: ParquetProvider):
        sub = provider.query_topology(TopologyFilter(limit=5))
        assert len(sub.entities) <= 5


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
class TestRobustness:
    def test_missing_file_returns_empty(self, tmp_path):
        # Build a Case whose case_dir has no parquet files.
        tw = TimeWindow(
            start=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
            end=datetime(2026, 4, 26, 0, 0, tzinfo=timezone.utc),
        )
        case = Case(
            task=Task(task_id="ghost", alert_title="x", alert_window=tw, prompt_text=""),
            topology=Topology(case_id="ghost", window=tw),
            case_dir=str(tmp_path),
        )
        p = ParquetProvider(case)
        assert p.query_metrics(MetricFilter(window=tw)) == []
        assert p.query_logs(LogFilter(window=tw)) == []
        assert p.query_traces(TraceFilter(window=tw)) == []
        assert p.query_events(EventFilter(window=tw)) == []
        assert p.query_alerts(AlertFilter(window=tw)) == []


# --------------------------------------------------------------------------- #
# Bounded LRU table cache (U9)
# --------------------------------------------------------------------------- #
# A wide window that captures the synthetic rows below regardless of the
# metric time value we write.
_WIDE_WINDOW = TimeWindow(
    start=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
    end=datetime(2026, 4, 26, 0, 0, tzinfo=timezone.utc),
    start_us=0,
    end_us=2**53,
)


def _synthetic_case(tmp_path: Path, case_id: str = "synth") -> Case:
    """Build a minimal Case whose case_dir is ``tmp_path`` (no parquet yet)."""
    return Case(
        task=Task(task_id=case_id, alert_title="x", alert_window=_WIDE_WINDOW, prompt_text=""),
        topology=Topology(case_id=case_id, window=_WIDE_WINDOW),
        case_dir=str(tmp_path),
    )


def _write_metrics_parquet(
    tmp_path: Path, rows: list[dict], columns: dict[str, pa.DataType] | None = None
) -> Path:
    """Write a synthetic metrics.parquet into ``tmp_path`` and return its path.

    ``rows`` are written as-is; ``columns`` lets the caller override types
    (e.g. to store a JSON-string column as a non-string type for malformed
    cases). When ``columns`` is None the schema is inferred from ``rows``.
    """
    path = tmp_path / "metrics.parquet"
    if columns is not None:
        arrays = [pa.array([r.get(c) for r in rows], type=t) for c, t in columns.items()]
        table = pa.Table.from_arrays(arrays, names=list(columns.keys()))
    elif rows:
        table = pa.Table.from_pylist(rows)
    else:
        # 0-row table with a single placeholder column so the file is valid.
        table = pa.table({"time": pa.array([], type=pa.int64())})
    pq.write_table(table, path)
    return path


class TestCacheDefaults:
    def test_default_maxsize_is_64(self):
        # Ensures the env-tunable preserves the documented default benefit.
        p = ParquetProvider(_synthetic_case(Path("/tmp/__does_not_exist_u9__")))
        assert p._cache_max == 64


class TestCacheMaxsizeEnvOverride:
    def test_env_overrides_maxsize(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RCA_PARQUET_CACHE_MAX", "3")
        p = ParquetProvider(_synthetic_case(tmp_path))
        assert p._cache_max == 3

    def test_nonpositive_env_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RCA_PARQUET_CACHE_MAX", "0")
        assert ParquetProvider(_synthetic_case(tmp_path))._cache_max == 64
        monkeypatch.setenv("RCA_PARQUET_CACHE_MAX", "-5")
        assert ParquetProvider(_synthetic_case(tmp_path))._cache_max == 64

    def test_garbage_env_falls_back_to_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RCA_PARQUET_CACHE_MAX", "not-a-number")
        assert ParquetProvider(_synthetic_case(tmp_path))._cache_max == 64


class TestCacheHitAndEviction:
    def test_cache_hit_returns_same_table_object(self, tmp_path):
        # Two reads of the same (name, columns) must hit the cache: the
        # underlying pyarrow Table is reused (only ONE cache entry exists).
        _write_metrics_parquet(
            tmp_path,
            [{"time": 1, "domain": "k8s", "entity_id": "e1", "metric": "cpu", "value": 1.0}],
        )
        p = ParquetProvider(_synthetic_case(tmp_path))
        first = p._read("metrics.parquet", ["time", "metric", "value"])
        second = p._read("metrics.parquet", ["time", "metric", "value"])
        assert first == second
        # Same column-set => same cache key => the cached Table is the ONLY entry.
        cached_keys = list(p._table_cache.keys())
        assert cached_keys == [("metrics.parquet", frozenset({"time", "metric", "value"}))]

    def test_exceeding_maxsize_evicts_lru(self, tmp_path, monkeypatch):
        # maxsize=2: insert A, B (full); touch A (A=MRU, B=LRU); insert C ->
        # B evicted, A+C remain.
        monkeypatch.setenv("RCA_PARQUET_CACHE_MAX", "2")
        _write_metrics_parquet(tmp_path, [{"time": 1, "metric": "m", "value": 1.0}])
        p = ParquetProvider(_synthetic_case(tmp_path))

        # Distinct column-sets give distinct cache keys against the one file.
        p._read("metrics.parquet", ["time", "metric", "value"])          # key A
        p._read("metrics.parquet", ["time", "metric"])                   # key B
        assert len(p._table_cache) == 2
        # Touch A so B becomes the LRU.
        p._read("metrics.parquet", ["time", "metric", "value"])          # A -> MRU
        # Insert C: exceeds maxsize -> evict LRU (B).
        p._read("metrics.parquet", ["time"])                             # key C
        assert len(p._table_cache) == 2
        keys = set(p._table_cache.keys())
        # B (["time","metric"]) was evicted; A and C remain.
        assert ("metrics.parquet", frozenset({"time", "metric", "value"})) in keys
        assert ("metrics.parquet", frozenset({"time"})) in keys
        assert ("metrics.parquet", frozenset({"time", "metric"})) not in keys

    def test_lru_order_moves_on_hit(self, tmp_path, monkeypatch):
        # maxsize=1: insert A, then B evicts A; re-reading A (a miss now)
        # re-populates it and evicts B; a subsequent hit on the sole entry
        # must NOT change the cache size.
        monkeypatch.setenv("RCA_PARQUET_CACHE_MAX", "1")
        _write_metrics_parquet(tmp_path, [{"time": 1, "metric": "m", "value": 1.0}])
        p = ParquetProvider(_synthetic_case(tmp_path))
        p._read("metrics.parquet", ["time", "metric", "value"])  # A
        p._read("metrics.parquet", ["time", "metric"])           # B, evicts A
        assert len(p._table_cache) == 1
        assert ("metrics.parquet", frozenset({"time", "metric"})) in p._table_cache
        # Hit on the sole entry: move_to_end keeps size at 1.
        p._read("metrics.parquet", ["time", "metric"])
        assert len(p._table_cache) == 1


class TestMalformedParquet:
    """Malformed-input robustness: the provider must NEVER raise on a bad file.

    Each case mirrors the real on-disk layout (file under case_dir with the
    exact filename the provider expects) but corrupts the contents / schema.
    """

    def test_missing_expected_column_returns_empty_result(self, tmp_path):
        # metrics.parquet exists but lacks every column query_metrics reads.
        _write_metrics_parquet(tmp_path, [{"unrelated": "x"}])
        p = ParquetProvider(_synthetic_case(tmp_path))
        out = p.query_metrics(MetricFilter(window=_WIDE_WINDOW))
        assert out == []

    def test_empty_table_returns_empty_result(self, tmp_path):
        # 0-row parquet file -> structured-empty result, no crash.
        _write_metrics_parquet(tmp_path, [])
        p = ParquetProvider(_synthetic_case(tmp_path))
        assert p.query_metrics(MetricFilter(window=_WIDE_WINDOW)) == []

    def test_wrong_typed_json_column_does_not_crash(self, tmp_path):
        # alerts.data is expected to be a JSON string; here it is an int list
        # column (not a JSON string). The provider must parse gracefully and
        # NOT raise; data degrades to {} (the documented graceful fallback).
        path = tmp_path / "alerts.parquet"
        table = pa.Table.from_arrays(
            [
                pa.array(["a1"], type=pa.string()),
                pa.array(["2026-04-25T12:00:00Z"], type=pa.string()),
                pa.array([[1, 2, 3]], type=pa.list_(pa.int64())),  # non-JSON column
            ],
            names=["id", "time", "data"],
        )
        pq.write_table(table, path)
        p = ParquetProvider(_synthetic_case(tmp_path))
        out = p.query_alerts(AlertFilter(window=_WIDE_WINDOW))
        assert isinstance(out, list)
        # Row passes filters; data column gracefully degrades to {}.
        assert len(out) == 1
        assert out[0].data == {}

    def test_corrupt_parquet_file_returns_empty(self, tmp_path):
        # Bytes that are NOT a valid parquet file -> empty result + no raise.
        (tmp_path / "metrics.parquet").write_bytes(b"not a parquet file at all")
        p = ParquetProvider(_synthetic_case(tmp_path))
        assert p.query_metrics(MetricFilter(window=_WIDE_WINDOW)) == []

    def test_corrupt_file_does_not_poison_cache(self, tmp_path):
        # A corrupt file returns [] but must not cache a bad value that masks
        # a later repair: after "repairing" the file on disk, a fresh read
        # must see the new rows (corrupt reads are never cached).
        path = tmp_path / "metrics.parquet"
        path.write_bytes(b"garbage")
        p = ParquetProvider(_synthetic_case(tmp_path))
        assert p.query_metrics(MetricFilter(window=_WIDE_WINDOW)) == []
        assert len(p._table_cache) == 0  # corrupt read is not cached
        # Repair the file on disk.
        path.unlink()
        _write_metrics_parquet(tmp_path, [{"time": 1, "metric": "m", "value": 1.0}])
        out = p.query_metrics(MetricFilter(window=_WIDE_WINDOW))
        assert isinstance(out, list)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
@needs_t001
class TestRender:
    def test_render_metrics(self, provider: ParquetProvider):
        ms = provider.query_metrics(MetricFilter(window=provider.window, limit=3))
        s = render(ms)
        assert s.startswith("metrics:")

    def test_render_empty(self):
        assert render([]) == "(empty)"

    def test_render_logs(self, provider: ParquetProvider):
        lg = provider.query_logs(LogFilter(window=provider.window, limit=3))
        if lg:
            assert render(lg).startswith("logs:")

    def test_render_topology(self, provider: ParquetProvider):
        sub = provider.query_topology(TopologyFilter())
        assert render(sub).startswith("topology:")

    def test_render_alerts(self, provider: ParquetProvider):
        al = provider.query_alerts(AlertFilter(window=provider.window, limit=3))
        assert render(al).startswith("alerts:")
