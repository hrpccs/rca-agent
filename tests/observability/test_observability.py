"""Unit tests for rca_agent.observability (tracing + metrics + eval scaffold).

These tests assert spans/metrics via the SDK InMemory exporters and do NOT
depend on the live OTel collector. A single ``@pytest.mark.live`` smoke test
exports one span to the real collector when live infra is present.
"""
from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from rca_agent.config import get_settings
from rca_agent.observability import metrics as m
from rca_agent.observability import tracing as t


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure get_settings() picks up env overrides and doesn't leak across tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Fixtures: install in-memory SDK providers and reset module globals per test.
# --------------------------------------------------------------------------- #
@pytest.fixture
def inmemory_providers(monkeypatch):
    """Install a TracerProvider + MeterProvider backed by in-memory exporters.

    Also resets the tracing/metrics module-global ``_initialized`` flag and
    lazily-created instruments so each test starts from a clean slate.
    """
    # Reset tracing module state so setup is forced fresh.
    monkeypatch.setattr(t, "_initialized", False)
    monkeypatch.setattr(t, "_tracer_provider", None)
    monkeypatch.setattr(t, "_meter_provider", None)
    monkeypatch.setattr(t, "_logger_provider", None)

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    # Install onto the tracing module globals directly (get_tracer/get_meter
    # prefer these) so we avoid the OTel global-API "no override" guard across
    # tests.
    monkeypatch.setattr(t, "_tracer_provider", tracer_provider)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(
        resource=Resource.create({"service.name": "test"}),
        metric_readers=[metric_reader],
    )
    monkeypatch.setattr(t, "_meter_provider", meter_provider)

    # Reset lazily-created instruments so they bind to the new meter.
    monkeypatch.setattr(m, "_tool_calls_total", None)
    monkeypatch.setattr(m, "_provider_query_duration", None)
    monkeypatch.setattr(m, "_provider_errors_total", None)
    monkeypatch.setattr(m, "_memory_retrieval_hits", None)
    monkeypatch.setattr(m, "_llm_tokens", None)
    monkeypatch.setattr(m, "_steps_total", None)
    monkeypatch.setattr(m, "_runs_total", None)

    # Mark initialized so get_tracer/get_meter skip setup_otel (we set the
    # providers directly on the global API above).
    monkeypatch.setattr(t, "_initialized", True)

    yield span_exporter, metric_reader


def _collected_spans(span_exporter: InMemorySpanExporter):
    # force_flush is a no-op for InMemorySpanExporter (SimpleSpanProcessor is
    # synchronous) but harmless.
    return span_exporter.get_finished_spans()


def _metrics_snapshot(metric_reader: InMemoryMetricReader):
    """Collect + return the current metric records as a list of Metric datums."""
    return metric_reader.get_metrics_data()


def _sum_values(snapshot, instrument_name: str, expected_attrs: dict[str, str]) -> float:
    """Sum data-point values across resource/scope/metric for an instrument
    whose points match expected_attrs."""
    total = 0.0
    if snapshot is None:
        return total
    for rm in snapshot.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != instrument_name:
                    continue
                for dp in metric.data.data_points:
                    if expected_attrs.items() <= dp.attributes.items():
                        total += getattr(
                            dp,
                            "value",
                            getattr(dp, "sum", 0.0),
                        )
    return total


# --------------------------------------------------------------------------- #
# setup_otel
# --------------------------------------------------------------------------- #
def test_setup_otel_is_idempotent(monkeypatch):
    """Repeated setup_otel calls must not raise and must short-circuit."""
    monkeypatch.setattr(t, "_initialized", False)
    monkeypatch.setattr(t, "_tracer_provider", None)
    monkeypatch.setattr(t, "_meter_provider", None)
    monkeypatch.setattr(t, "_logger_provider", None)
    # Point exporter at a non-resolving endpoint so no network happens; we only
    # check idempotency / guard behaviour, not export.
    monkeypatch.setenv("RCA_OTEL_ENABLED", "true")
    monkeypatch.setenv("RCA_OTEL_ENDPOINT", "http://127.0.0.1:4317")
    get_settings.cache_clear()
    calls = {"n": 0}

    # Patch the OTLPSpanExporter/MetricExporter so setup_otel runs fully
    # offline. The metric reader inspects exporter attributes, so we subclass
    # the real exporter class to retain them while no-op'ing export/shutdown.
    import opentelemetry.exporter.otlp.proto.grpc.metric_exporter as me
    import opentelemetry.exporter.otlp.proto.grpc.trace_exporter as te

    class DummySpanExporter:
        def __init__(self, *a, **kw):
            calls["n"] += 1

        def export(self, *a, **kw):
            return None

        def shutdown(self, *a, **kw):
            return None

    class DummyMetricExporter(me.OTLPMetricExporter):
        def __init__(self, *a, **kw):
            calls["n"] += 1
            # Bypass the real gRPC channel construction.
            self._preferred_temporality = {}
            self._preferred_aggregation = {}

        def export(self, *a, **kw):
            return None

        def shutdown(self, *a, **kw):
            return None

    monkeypatch.setattr(te, "OTLPSpanExporter", DummySpanExporter)
    monkeypatch.setattr(me, "OTLPMetricExporter", DummyMetricExporter)

    t.setup_otel()
    first = calls["n"]
    t.setup_otel()  # idempotent -> should not re-create exporters
    assert calls["n"] == first
    assert t._initialized is True


def test_setup_otel_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(t, "_initialized", False)
    monkeypatch.setattr(t, "_tracer_provider", None)
    monkeypatch.setattr(t, "_meter_provider", None)
    monkeypatch.setenv("RCA_OTEL_ENABLED", "false")
    get_settings.cache_clear()
    t.setup_otel()
    assert t._initialized is True
    assert t._tracer_provider is None
    assert t._meter_provider is None


# --------------------------------------------------------------------------- #
# get_tracer / get_meter
# --------------------------------------------------------------------------- #
def test_get_tracer_and_meter(inmemory_providers):
    span_exporter, metric_reader = inmemory_providers
    tracer = t.get_tracer()
    meter = t.get_meter()
    assert tracer is not None
    assert meter is not None
    # Produces a real span.
    with t.span("manual"):
        pass
    spans = _collected_spans(span_exporter)
    assert any(s.name == "manual" for s in spans)


# --------------------------------------------------------------------------- #
# trace_rca_step
# --------------------------------------------------------------------------- #
def test_trace_rca_step_sync_sets_attributes(inmemory_providers):
    span_exporter, _ = inmemory_providers

    @t.trace_rca_step("investigate")
    def investigate(case_id, step_kind="observe"):
        return {"case_id": case_id, "step_kind": "investigate", "tool_name": "query_logs"}

    investigate("case-1", step_kind="investigate")

    spans = _collected_spans(span_exporter)
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "rca.investigate"
    assert s.attributes["case_id"] == "case-1"
    # step_kind from kwargs takes precedence over the return value.
    assert s.attributes["step_kind"] == "investigate"
    assert s.attributes["tool_name"] == "query_logs"
    assert s.status.is_ok


def test_trace_rca_step_records_exception(inmemory_providers):
    span_exporter, _ = inmemory_providers

    @t.trace_rca_step()
    def boom(case_id):
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        boom("case-9")

    spans = _collected_spans(span_exporter)
    assert len(spans) == 1
    s = spans[0]
    assert not s.status.is_ok
    # An exception event was recorded.
    assert any(ev.name == "exception" for ev in s.events)
    assert s.attributes["case_id"] == "case-9"


def test_trace_rca_step_empty_message_status(inmemory_providers):
    """An exception whose str() is empty still yields a non-empty, type-named
    status description (the SDK span __exit__ prefixes the exception type)."""

    class _EmptyMsg(Exception):
        pass

    span_exporter, _ = inmemory_providers

    @t.trace_rca_step()
    def boom():
        raise _EmptyMsg()

    with pytest.raises(_EmptyMsg):
        boom()

    s = _collected_spans(span_exporter)[0]
    assert not s.status.is_ok
    # The SDK's __exit__ sets description to f"{type}: {exc}"; for an empty
    # message that is "_EmptyMsg: " — never blank, and carries the type name.
    assert s.status.description and "_EmptyMsg" in s.status.description


def test_trace_rca_step_does_not_leak_param_defaults(inmemory_providers):
    """Omitted params must not contribute default-value span attributes."""
    span_exporter, _ = inmemory_providers

    @t.trace_rca_step()
    def step(case_id=None, step_kind="observe"):
        return None

    step()  # neither case_id nor step_kind supplied

    s = _collected_spans(span_exporter)[0]
    # Defaults must not leak: no attributes were passed by the caller.
    assert "case_id" not in (s.attributes or {})
    assert "step_kind" not in (s.attributes or {})


def test_trace_rca_step_async(inmemory_providers):
    span_exporter, _ = inmemory_providers

    @t.trace_rca_step("async_step")
    async def async_step(case_id, tool_name="x"):
        return {"case_id": case_id, "tool_name": tool_name}

    asyncio.run(async_step("case-2", tool_name="span_q"))

    spans = _collected_spans(span_exporter)
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "rca.async_step"
    assert s.attributes["case_id"] == "case-2"
    assert s.attributes["tool_name"] == "span_q"


def test_trace_rca_step_custom_step_kind_attr(inmemory_providers):
    span_exporter, _ = inmemory_providers

    @t.trace_rca_step(step_kind_attr="kind")
    def step(case_id, kind="observe"):
        return None

    step("case-3", kind="hypothesize")

    spans = _collected_spans(span_exporter)
    assert spans[0].attributes["case_id"] == "case-3"
    assert spans[0].attributes["step_kind"] == "hypothesize"


# --------------------------------------------------------------------------- #
# span() helper
# --------------------------------------------------------------------------- #
def test_span_helper_context_manager(inmemory_providers):
    span_exporter, _ = inmemory_providers
    with t.span("adhoc"):
        pass
    spans = _collected_spans(span_exporter)
    assert any(s.name == "adhoc" for s in spans)


# --------------------------------------------------------------------------- #
# Metrics recorders
# --------------------------------------------------------------------------- #
def test_record_tool_call(inmemory_providers):
    _, metric_reader = inmemory_providers
    m.record_tool_call("query_logs", "ok", case_id="c1")
    m.record_tool_call("query_logs", "ok")
    snap = _metrics_snapshot(metric_reader)
    assert _sum_values(snap, "rca_tool_calls_total", {"tool": "query_logs", "status": "ok"}) == 2


def test_record_provider_query(inmemory_providers):
    _, metric_reader = inmemory_providers
    m.record_provider_query("metrics", "clickhouse", 0.5, ok=True)
    m.record_provider_query("logs", "parquet", 0.2, ok=False)
    snap = _metrics_snapshot(metric_reader)
    # duration recorded for both
    assert _sum_values(snap, "rca_provider_query_duration_seconds",
                       {"modality": "metrics", "backend": "clickhouse"}) == pytest.approx(0.5)
    # error counter only for the failed one
    assert _sum_values(snap, "rca_provider_errors_total", {"modality": "logs"}) == 1


def test_record_provider_query_zero_duration_not_recorded(inmemory_providers):
    """A clamped (zero/negative/None) duration must not emit a histogram point."""
    _, metric_reader = inmemory_providers
    m.record_provider_query("metrics", "clickhouse", 0.0, ok=True)
    m.record_provider_query("metrics", "clickhouse", -1.0, ok=True)
    snap = _metrics_snapshot(metric_reader)
    assert _sum_values(
        snap, "rca_provider_query_duration_seconds",
        {"modality": "metrics", "backend": "clickhouse"},
    ) == 0.0


def test_record_memory_hits(inmemory_providers):
    _, metric_reader = inmemory_providers
    m.record_memory_hits("c1", 3)
    m.record_memory_hits("c1", 2)
    snap = _metrics_snapshot(metric_reader)
    assert _sum_values(snap, "rca_memory_retrieval_hits", {"case_id": "c1"}) == 5


def test_record_memory_hits_zero_not_recorded(inmemory_providers):
    """Zero/negative hit counts must not emit a point (no per-case zero bucket)."""
    _, metric_reader = inmemory_providers
    m.record_memory_hits("c1", 0)
    m.record_memory_hits("c1", -2)
    snap = _metrics_snapshot(metric_reader)
    found = False
    if snap is not None:
        for rm in snap.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "rca_memory_retrieval_hits":
                        found = True
    assert not found


def test_record_llm_tokens(inmemory_providers):
    _, metric_reader = inmemory_providers
    m.record_llm_tokens("deepseek-reasoner", prompt=100, completion=50)
    snap = _metrics_snapshot(metric_reader)
    assert _sum_values(snap, "rca_llm_tokens",
                       {"kind": "prompt", "model": "deepseek-reasoner"}) == 100
    assert _sum_values(snap, "rca_llm_tokens",
                       {"kind": "completion", "model": "deepseek-reasoner"}) == 50


def test_record_step_and_run(inmemory_providers):
    _, metric_reader = inmemory_providers
    m.record_step("observe")
    m.record_step("observe")
    m.record_run("completed")
    snap = _metrics_snapshot(metric_reader)
    assert _sum_values(snap, "rca_steps_total", {"step_kind": "observe"}) == 2
    assert _sum_values(snap, "rca_runs_total", {"status": "completed"}) == 1


def test_record_llm_tokens_zero_is_noop(inmemory_providers):
    """Zero/negative token counts must not emit points (guard clamping)."""
    _, metric_reader = inmemory_providers
    m.record_llm_tokens("m", prompt=0, completion=0)
    m.record_llm_tokens("m", prompt=-5, completion=-5)
    snap = _metrics_snapshot(metric_reader)
    # No instrument should have emitted any points.
    found = False
    if snap is not None:
        for rm in snap.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "rca_llm_tokens":
                        found = True
    # InMemoryMetricReader omits instruments with no data points, so the
    # rca_llm_tokens instrument should be entirely absent.
    assert not found


# --------------------------------------------------------------------------- #
# Evaluation recorder scaffold
# --------------------------------------------------------------------------- #
def test_evaluation_recorder_default_inmemory():
    rec = m.get_evaluation_recorder()
    assert isinstance(rec, m.InMemoryEvaluationRecorder)
    rec.record("c1", "precision", 0.9)
    assert rec.records == [("c1", "precision", 0.9)]


def test_evaluation_recorder_is_protocol():
    assert isinstance(m.InMemoryEvaluationRecorder(), m.EvaluationRecorder)


def test_set_evaluation_recorder():
    class CustomRecorder:
        def __init__(self):
            self.seen = []

        def record(self, case_id, metric, value):
            self.seen.append((case_id, metric, value))

    custom = CustomRecorder()
    m.set_evaluation_recorder(custom)
    try:
        assert m.get_evaluation_recorder() is custom
        m.get_evaluation_recorder().record("c2", "recall", 0.8)
        assert custom.seen == [("c2", "recall", 0.8)]
    finally:
        m.set_evaluation_recorder(m.InMemoryEvaluationRecorder())


# --------------------------------------------------------------------------- #
# Live collector smoke test (opt-in)
# --------------------------------------------------------------------------- #
@pytest.mark.live
def test_live_smoke_export(monkeypatch):
    """Export one span + one metric to the real OTLP collector on :4317."""
    monkeypatch.setattr(t, "_initialized", False)
    monkeypatch.setenv("RCA_OTEL_ENABLED", "true")
    monkeypatch.setenv("RCA_OTEL_ENDPOINT", "http://localhost:4317")
    t.setup_otel()
    with t.span("live-smoke"):
        m.record_tool_call("query_logs", "ok")
    # Force-flush span processor so the span leaves the process.
    if t._tracer_provider is not None:
        t._tracer_provider.force_flush()
