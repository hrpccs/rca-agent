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


def test_setup_otel_construction_failure_degrades_to_noop(monkeypatch, caplog):
    """If exporter / provider construction raises, setup_otel must degrade to
    no-op providers, log a single structured warning, set _initialized, and
    NEVER raise into the caller."""
    monkeypatch.setattr(t, "_initialized", False)
    monkeypatch.setattr(t, "_tracer_provider", None)
    monkeypatch.setattr(t, "_meter_provider", None)
    monkeypatch.setattr(t, "_logger_provider", None)
    monkeypatch.setenv("RCA_OTEL_ENABLED", "true")
    monkeypatch.setenv("RCA_OTEL_ENDPOINT", "http://127.0.0.1:4317")
    get_settings.cache_clear()

    # Force a concrete failure mid-construction: the span exporter import
    # itself raises a ValueError (malformed endpoint). The except branches in
    # setup_otel must catch it.
    import opentelemetry.exporter.otlp.proto.grpc.trace_exporter as te

    class _BoomExporter:
        def __init__(self, *a, **kw):
            raise ValueError("boom: malformed endpoint")

        def shutdown(self, *a, **kw):
            return None

    monkeypatch.setattr(te, "OTLPSpanExporter", _BoomExporter)

    with caplog.at_level("WARNING", logger="rca_agent.observability.tracing"):
        # Must not raise.
        t.setup_otel()

    # Degraded to no-op providers.
    assert t._initialized is True
    assert t._tracer_provider is None
    assert t._meter_provider is None
    # Exactly one structured warning about the setup failure.
    setup_warnings = [r for r in caplog.records
                      if r.name == "rca_agent.observability.tracing"
                      and "otel setup" in r.getMessage()]
    assert len(setup_warnings) == 1, f"expected one setup warning, got {setup_warnings}"
    rec = setup_warnings[0]
    assert rec.levelname == "WARNING"
    # Structured extra payload present + diagnosable.
    assert getattr(rec, "component", None) == "otel"
    assert getattr(rec, "phase", None) == "setup_otel"
    assert getattr(rec, "error_type", None) == "ValueError"
    assert "boom" in getattr(rec, "error", "")


def test_setup_otel_partial_failure_after_tracer_installed(monkeypatch, caplog):
    """A failure AFTER the tracer provider is globally installed (e.g. the
    metric exporter constructor raising) must still: not raise, set
    _initialized, log a single warning, and leave get_tracer/get_meter
    returning usable no-op tracers/meters (the module globals are cleared so
    callers fall back to the SDK no-op path). This covers the partial-state
    window the pre-install failure test does not exercise."""
    monkeypatch.setattr(t, "_initialized", False)
    monkeypatch.setattr(t, "_tracer_provider", None)
    monkeypatch.setattr(t, "_meter_provider", None)
    monkeypatch.setattr(t, "_logger_provider", None)
    monkeypatch.setenv("RCA_OTEL_ENABLED", "true")
    monkeypatch.setenv("RCA_OTEL_ENDPOINT", "http://127.0.0.1:4317")
    get_settings.cache_clear()

    import opentelemetry.exporter.otlp.proto.grpc.metric_exporter as me
    import opentelemetry.exporter.otlp.proto.grpc.trace_exporter as te

    # Span exporter constructs fine (tracer provider WILL be installed)...
    class _OkSpanExporter:
        def export(self, *a, **kw):
            return None

        def shutdown(self, *a, **kw):
            return None

    # ...but the metric exporter blows up, AFTER set_tracer_provider ran.
    class _BoomMetricExporter(me.OTLPMetricExporter):
        def __init__(self, *a, **kw):
            raise OSError("metric exporter boom")

    monkeypatch.setattr(te, "OTLPSpanExporter", _OkSpanExporter)
    monkeypatch.setattr(me, "OTLPMetricExporter", _BoomMetricExporter)

    with caplog.at_level("WARNING", logger="rca_agent.observability.tracing"):
        t.setup_otel()  # must not raise

    # Degraded: _initialized set, module globals cleared so callers fall to
    # the SDK no-op path (traces via the still-installed-but-discarded global
    # provider, metrics via no-op).
    assert t._initialized is True
    assert t._tracer_provider is None
    assert t._meter_provider is None
    setup_warnings = [r for r in caplog.records
                      if r.name == "rca_agent.observability.tracing"
                      and "otel setup" in r.getMessage()]
    assert len(setup_warnings) == 1
    # get_tracer / get_meter must still hand back usable no-op instruments —
    # never None, never raising — even from the degraded state.
    tracer = t.get_tracer()
    meter = t.get_meter()
    assert tracer is not None
    assert meter is not None
    with tracer.start_as_current_span("post-failure"):
        meter.create_counter("c").add(1, {})


def test_get_tracer_never_raises(monkeypatch, caplog):
    """get_tracer must return a usable tracer and never raise, even if the
    installed provider's get_tracer raises — it falls back to the SDK no-op
    tracer with a single warning."""
    monkeypatch.setattr(t, "_initialized", True)

    class _BrokenProvider:
        def get_tracer(self, *a, **kw):
            raise RuntimeError("provider broken")

    monkeypatch.setattr(t, "_tracer_provider", _BrokenProvider())

    with caplog.at_level("WARNING", logger="rca_agent.observability.tracing"):
        tracer = t.get_tracer()
    assert tracer is not None  # no-op tracer returned, not None
    # A single warning was logged.
    tracer_warnings = [r for r in caplog.records
                       if "get_tracer" in r.getMessage()]
    assert len(tracer_warnings) == 1
    assert tracer_warnings[0].levelname == "WARNING"


def test_get_meter_never_raises(monkeypatch, caplog):
    """get_meter must return a usable meter and never raise, even if the
    installed provider's get_meter raises — it falls back to the SDK no-op
    meter with a single warning."""
    monkeypatch.setattr(t, "_initialized", True)

    class _BrokenProvider:
        def get_meter(self, *a, **kw):
            raise RuntimeError("provider broken")

    monkeypatch.setattr(t, "_meter_provider", _BrokenProvider())

    with caplog.at_level("WARNING", logger="rca_agent.observability.tracing"):
        meter = t.get_meter()
    assert meter is not None  # no-op meter returned, not None
    meter_warnings = [r for r in caplog.records if "get_meter" in r.getMessage()]
    assert len(meter_warnings) == 1
    assert meter_warnings[0].levelname == "WARNING"


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


def test_trace_rca_step_async_records_exception_event(inmemory_providers):
    """An async decorated fn that raises must yield a span with an exception
    event + non-ok status, and re-raise to the caller."""
    span_exporter, _ = inmemory_providers

    @t.trace_rca_step("async_boom")
    async def async_boom(case_id):
        raise ValueError("async failure")

    with pytest.raises(ValueError, match="async failure"):
        asyncio.run(async_boom("case-x"))

    spans = _collected_spans(span_exporter)
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "rca.async_boom"
    assert not s.status.is_ok
    assert any(ev.name == "exception" for ev in s.events)
    # Call attributes are still attached on the exception path.
    assert s.attributes["case_id"] == "case-x"


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


def test_record_provider_query_none_duration_not_recorded(inmemory_providers):
    """A None duration (timer never set) must not emit a histogram point
    and must not raise — mirrors the zero/negative clamp guard."""
    _, metric_reader = inmemory_providers
    # ok=True so no error counter either; nothing should be emitted.
    m.record_provider_query("metrics", "clickhouse", None, ok=True)  # type: ignore[arg-type]
    snap = _metrics_snapshot(metric_reader)
    assert _sum_values(
        snap, "rca_provider_query_duration_seconds",
        {"modality": "metrics", "backend": "clickhouse"},
    ) == 0.0
    # And the failed-query branch isn't triggered (ok=True).
    assert _sum_values(
        snap, "rca_provider_errors_total", {"modality": "metrics"},
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


def test_record_memory_hits_none_not_recorded(inmemory_providers):
    """A None hit count (retrieval returned None) must not emit a point
    and must not raise — mirrors the zero/negative clamp guard."""
    _, metric_reader = inmemory_providers
    m.record_memory_hits("c1", None)  # type: ignore[arg-type]
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


def test_record_llm_tokens_none_is_noop(inmemory_providers):
    """None token counts must not emit points and must not raise — mirrors
    the zero/negative clamp guard."""
    _, metric_reader = inmemory_providers
    m.record_llm_tokens("m", prompt=None, completion=None)  # type: ignore[arg-type]
    snap = _metrics_snapshot(metric_reader)
    found = False
    if snap is not None:
        for rm in snap.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == "rca_llm_tokens":
                        found = True
    assert not found


def test_record_llm_tokens_mixed_clamps_per_kind(inmemory_providers):
    """A valid prompt with a None completion records only the prompt side,
    and vice-versa — the clamp guard is applied per kind, not wholesale."""
    _, metric_reader = inmemory_providers
    m.record_llm_tokens("m", prompt=100, completion=None)  # type: ignore[arg-type]
    m.record_llm_tokens("m", prompt=None, completion=40)  # type: ignore[arg-type]
    snap = _metrics_snapshot(metric_reader)
    assert _sum_values(snap, "rca_llm_tokens",
                       {"kind": "prompt", "model": "m"}) == 100
    assert _sum_values(snap, "rca_llm_tokens",
                       {"kind": "completion", "model": "m"}) == 40


# --------------------------------------------------------------------------- #
# Concurrent recording (asyncio.gather must sum correctly)
# --------------------------------------------------------------------------- #
def test_concurrent_metric_recording_sums(inmemory_providers):
    """Many coroutines recording metrics via asyncio.gather must sum to the
    expected totals — the SDK Counter/Histogram add/record paths are not
    awaited, so concurrent await-points must not drop or double-count."""
    _, metric_reader = inmemory_providers
    N = 50

    async def worker(i: int) -> None:
        # Interleave a couple of await points so the event loop actually
        # schedules tasks concurrently.
        await asyncio.sleep(0)
        m.record_tool_call("query_logs", "ok", case_id=f"c{i}")
        m.record_provider_query("metrics", "clickhouse", 0.1, ok=True)
        m.record_memory_hits(f"c{i}", 2)
        m.record_llm_tokens("m", prompt=10, completion=5)
        m.record_step("observe")
        await asyncio.sleep(0)

    async def run_all() -> None:
        # gather must run inside a coroutine so a loop is active.
        await asyncio.gather(*(worker(i) for i in range(N)))

    asyncio.run(run_all())

    snap = _metrics_snapshot(metric_reader)
    # The same case_id is used once per worker, so total counts == N for the
    # per-case-attribute groupings.
    assert _sum_values(snap, "rca_tool_calls_total",
                       {"tool": "query_logs", "status": "ok"}) == N
    assert _sum_values(snap, "rca_provider_query_duration_seconds",
                       {"modality": "metrics", "backend": "clickhouse"}) == pytest.approx(0.1 * N)
    assert _sum_values(snap, "rca_steps_total", {"step_kind": "observe"}) == N
    # memory_hits: N distinct case_ids, 2 each -> total 2N across all cases.
    assert _sum_values(snap, "rca_memory_retrieval_hits", {}) == 2 * N
    # tokens: prompt 10*N + completion 5*N.
    assert _sum_values(snap, "rca_llm_tokens", {"kind": "prompt", "model": "m"}) == 10 * N
    assert _sum_values(snap, "rca_llm_tokens", {"kind": "completion", "model": "m"}) == 5 * N


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
