"""OpenTelemetry setup and tracing helpers for the RCA agent.

This module is the single place that configures the global TracerProvider /
MeterProvider / LoggerProvider and exports them to OTLP gRPC. Every other
module should call :func:`get_tracer` / :func:`get_meter` and decorate agent
steps / tool handlers with :func:`trace_rca_step`.

Design notes
------------
* :func:`setup_otel` is **idempotent**: a module-level ``_initialized`` flag
  guards the provider construction so repeated calls (e.g. from several FastAPI
  startup hooks) are safe.
* It is guarded by ``settings.otel_enabled`` — when OTel is disabled the global
  providers are the SDK no-ops, so instrumentation code paths never crash.
* Logs are best-effort: the OTel logs API/exporter import paths have shifted
  across versions, so a failure to import them downgrades to traces+metrics
  only (the priority surfaces).
* The :func:`trace_rca_step` decorator is sync/async agnostic and sets span
  attributes derived from the wrapped function's kwargs (``case_id``,
  ``step_kind``, ``tool_name``) and, where available, its return value.
"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from rca_agent.config import get_settings

__all__ = [
    "setup_otel",
    "get_tracer",
    "get_meter",
    "trace_rca_step",
    "span",
]

# Module-global provider handles so repeated setup_otel() calls are no-ops and
# so tests / shutdown can reach them without touching the OTel global API.
_tracer_provider: TracerProvider | None = None
_meter_provider: metrics.MeterProvider | None = None
_logger_provider: Any | None = None
_initialized: bool = False

_TRACER_NAME = "rca_agent"
_METER_NAME = "rca_agent"


def _build_resource(service_name: str) -> Resource:
    return Resource.create({SERVICE_NAME: service_name})


def setup_otel(endpoint: str | None = None, service_name: str | None = None) -> None:
    """Configure Tracer + Meter + Logger providers exporting to OTLP gRPC.

    Reads defaults from :func:`rca_agent.config.get_settings`. Idempotent —
    safe to call many times; only the first call constructs the providers.
    When ``settings.otel_enabled`` is False this is a no-op (no-ops remain
    installed).

    Parameters
    ----------
    endpoint:
        OTLP gRPC endpoint, e.g. ``http://localhost:4317``. Falls back to
        ``settings.otel_endpoint``.
    service_name:
        OTel ``service.name`` resource attribute. Falls back to
        ``settings.otel_service_name``.
    """
    global _tracer_provider, _meter_provider, _logger_provider, _initialized

    if _initialized:
        return

    settings = get_settings()
    if not settings.otel_enabled:
        # Leave the global no-op providers in place; flag still set so the
        # guard short-circuits future calls.
        _initialized = True
        return

    endpoint = endpoint or settings.otel_endpoint
    service_name = service_name or settings.otel_service_name
    resource = _build_resource(service_name)

    # Mark initialized up front (in a try) so a failure partway through — e.g.
    # a metric exporter constructor raising after the tracer provider is
    # already registered — does not leave _initialized False and cause a
    # re-entrant setup_otel to call set_tracer_provider again (OTel rejects
    # overriding an installed global provider).
    try:
        # ---- TracerProvider + OTLP span exporter (BatchSpanProcessor) ----
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=True),
            )
        )
        trace.set_tracer_provider(tracer_provider)
        _tracer_provider = tracer_provider

        # ---- MeterProvider + OTLP metric exporter ----
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=15000
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter_provider = meter_provider

        # ---- LoggerProvider (best-effort; logs API varies by SDK version) ----
        _setup_logger_provider(endpoint, resource)
    finally:
        _initialized = True


def _setup_logger_provider(endpoint: str, resource: Any) -> None:
    """Best-effort OTel logs setup.

    The ``opentelemetry._logs`` / ``opentelemetry.sdk._logs`` paths moved over
    releases and the OTLP log exporter is in a separate instrumentation
    package. Any import/construct error is swallowed: traces and metrics are
    the priority observability surfaces.
    """
    global _logger_provider

    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    except Exception:  # noqa: BLE001 — best-effort, never fatal
        return

    try:
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=True))
        )
        set_logger_provider(logger_provider)
        _logger_provider = logger_provider
    except Exception:  # noqa: BLE001 — best-effort, never fatal
        _logger_provider = None


def get_tracer() -> Tracer:
    """Return the application tracer (lazy-initializing OTel if needed).

    Prefers the module-global provider installed by :func:`setup_otel` so the
    tracer always reflects the most recently configured provider (this also
    lets tests inject an in-memory provider without fighting the OTel global
    API's "no override" guard).
    """
    if not _initialized:
        setup_otel()
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer(_TRACER_NAME)
    return trace.get_tracer(_TRACER_NAME)


def get_meter() -> metrics.Meter:
    """Return the application meter (lazy-initializing OTel if needed)."""
    if not _initialized:
        setup_otel()
    if _meter_provider is not None:
        return _meter_provider.get_meter(_METER_NAME)
    return metrics.get_meter(_METER_NAME)


P = ParamSpec("P")
R = TypeVar("R")


def _bind_kwargs(func: Callable[..., Any], sig: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    """Merge positional + keyword arguments into a name->value mapping.

    Defaults are intentionally NOT applied: span attributes should reflect what
    the caller actually passed, not the function's default values (otherwise
    every span on a function with ``step_kind="observe"`` default would carry a
    meaningless ``step_kind`` even when the caller omitted it).

    Best-effort: if the signature can't be bound (e.g. ``*args`` splat), the
    raw ``kwargs`` are returned unchanged.
    """
    try:
        bound = sig.bind_partial(*args, **kwargs)
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _extract_attrs(
    bound: dict[str, Any], result: Any, step_kind_attr: str = "step_kind"
) -> dict[str, Any]:
    """Build span attributes from the bound call args / return value."""
    attrs: dict[str, Any] = {}
    if step_kind_attr in bound and bound[step_kind_attr] is not None:
        attrs["step_kind"] = str(bound[step_kind_attr])
    for key in ("case_id", "tool_name"):
        if key in bound and bound[key] is not None:
            attrs[key] = str(bound[key])

    # If the wrapped function returns an RcaStep (or a mapping that looks like
    # one), enrich the span with its step_kind / tool_name when missing.
    if result is not None:
        for key in ("step_kind", "tool_name", "case_id"):
            if key not in attrs:
                val = _attr_from(result, key)
                if val is not None:
                    attrs[key] = str(val)
    return attrs


def _attr_from(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def trace_rca_step(
    name: str | None = None,
    *,
    step_kind_attr: str = "step_kind",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate an agent step / tool handler so it runs under an OTel span.

    The span is named ``rca.<name>`` (``name`` defaults to the function name).
    It sets attributes ``case_id``, ``step_kind`` (looked up under the parameter
    named by ``step_kind_attr``) and ``tool_name`` from the wrapped function's
    bound arguments (positional or keyword) and its return value where present.

    On exception the span is left as ERROR: the SDK span's context-manager
    ``__exit__`` automatically records the exception as an event and sets an
    ERROR status with a ``"{type}: {message}"`` description when the exception
    propagates out of the ``with`` block, so we only add call attributes here and
    re-raise (a manual ``set_status``/``record_exception`` would be overwritten
    by ``__exit__``).

    Works for both ``async def`` and ``def`` functions.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = f"rca.{name or func.__name__}"
        tracer = get_tracer()
        # Introspect the signature once at decoration time rather than on every
        # call (inspect.signature is comparatively expensive).
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            sig = inspect.Signature()
        is_async = inspect.iscoroutinefunction(func)

        def _bind(args: tuple, kwargs: dict) -> dict[str, Any]:
            return _bind_kwargs(func, sig, args, kwargs)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                bound = _bind(args, kwargs)
                with tracer.start_as_current_span(span_name) as otel_span:
                    try:
                        result = await func(*args, **kwargs)  # type: ignore[no-any-return]
                        attrs = _extract_attrs(bound, result, step_kind_attr)
                        if attrs:
                            otel_span.set_attributes(attrs)
                        return result
                    except Exception:
                        # Attribute what we can from the call; the SDK's
                        # __exit__ records the exception + ERROR status.
                        attrs = _extract_attrs(bound, None, step_kind_attr)
                        if attrs:
                            otel_span.set_attributes(attrs)
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = _bind(args, kwargs)
            with tracer.start_as_current_span(span_name) as otel_span:
                try:
                    result = func(*args, **kwargs)
                    attrs = _extract_attrs(bound, result, step_kind_attr)
                    if attrs:
                        otel_span.set_attributes(attrs)
                    return result
                except Exception:
                    attrs = _extract_attrs(bound, None, step_kind_attr)
                    if attrs:
                        otel_span.set_attributes(attrs)
                    raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def span(name: str) -> Iterator[trace.Span]:
    """Context manager that starts an ad-hoc span on the app tracer."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        yield s
