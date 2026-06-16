"""Metric instruments + convenience recorders + eval scaffold for the RCA agent.

All instruments are created lazily from :func:`rca_agent.observability.tracing.get_meter`
so importing this module is side-effect-free; the first ``record_*`` call
triggers OTel setup. When OTel is disabled the meter is the SDK no-op, so the
``record_*`` helpers are safe no-ops.

The :class:`EvaluationRecorder` Protocol + :class:`InMemoryEvaluationRecorder`
are a **deliberately minimal scaffold** for the future agent-evaluation unit;
do not build the full eval harness here.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from opentelemetry import metrics

from rca_agent.observability.tracing import get_meter

__all__ = [
    "record_tool_call",
    "record_provider_query",
    "record_memory_hits",
    "record_llm_tokens",
    "record_step",
    "record_run",
    "EvaluationRecorder",
    "InMemoryEvaluationRecorder",
    "get_evaluation_recorder",
    "set_evaluation_recorder",
]

# Lazily-created instruments (module-global so we don't recreate on every call).
_tool_calls_total: metrics.Counter | None = None
_provider_query_duration: metrics.Histogram | None = None
_provider_errors_total: metrics.Counter | None = None
_memory_retrieval_hits: metrics.Counter | None = None
_llm_tokens: metrics.Counter | None = None
_steps_total: metrics.Counter | None = None
_runs_total: metrics.Counter | None = None


def _instrument(name: str) -> Any:
    """Return the named instrument, creating it lazily on first access."""
    global _tool_calls_total, _provider_query_duration, _provider_errors_total
    global _memory_retrieval_hits, _llm_tokens, _steps_total, _runs_total

    if name == "rca_tool_calls_total":
        if _tool_calls_total is None:
            _tool_calls_total = get_meter().create_counter(
                name, unit="1", description="Total RCA tool invocations."
            )
        return _tool_calls_total
    if name == "rca_provider_query_duration_seconds":
        if _provider_query_duration is None:
            _provider_query_duration = get_meter().create_histogram(
                name,
                unit="s",
                description="Duration of data-provider queries by modality/backend.",
            )
        return _provider_query_duration
    if name == "rca_provider_errors_total":
        if _provider_errors_total is None:
            _provider_errors_total = get_meter().create_counter(
                name, unit="1", description="Total data-provider query errors."
            )
        return _provider_errors_total
    if name == "rca_memory_retrieval_hits":
        if _memory_retrieval_hits is None:
            _memory_retrieval_hits = get_meter().create_counter(
                name,
                unit="1",
                description="Number of memory items retrieved per query.",
            )
        return _memory_retrieval_hits
    if name == "rca_llm_tokens":
        if _llm_tokens is None:
            _llm_tokens = get_meter().create_counter(
                name, unit="1", description="LLM tokens consumed (prompt/completion)."
            )
        return _llm_tokens
    if name == "rca_steps_total":
        if _steps_total is None:
            _steps_total = get_meter().create_counter(
                name, unit="1", description="RCA agent steps executed by kind."
            )
        return _steps_total
    if name == "rca_runs_total":
        if _runs_total is None:
            _runs_total = get_meter().create_counter(
                name, unit="1", description="RCA runs by terminal status."
            )
        return _runs_total
    raise KeyError(f"unknown instrument: {name}")


# --------------------------------------------------------------------------- #
# Convenience recorders (no-op safe when OTel disabled)
# --------------------------------------------------------------------------- #
def record_tool_call(tool: str, status: str, case_id: str | None = None) -> None:
    """Increment ``rca_tool_calls_total{tool, status[, case_id]}``."""
    attrs: dict[str, Any] = {"tool": str(tool), "status": str(status)}
    if case_id is not None:
        attrs["case_id"] = str(case_id)
    _instrument("rca_tool_calls_total").add(1, attrs)


def record_provider_query(
    modality: str, backend: str, seconds: float, ok: bool
) -> None:
    """Record a data-provider query: duration + error counter when failed."""
    duration = float(seconds) if seconds and seconds > 0 else 0.0
    # Only record a latency sample when we have a real (positive) duration — a
    # clamped 0.0 would skew histogram percentiles and is indistinguishable
    # from a malformed timer.
    if duration > 0:
        _instrument("rca_provider_query_duration_seconds").record(
            duration, {"modality": str(modality), "backend": str(backend)}
        )
    if not ok:
        _instrument("rca_provider_errors_total").add(1, {"modality": str(modality)})


def record_memory_hits(case_id: str, n: int) -> None:
    """Record the number of memory items retrieved for a case."""
    count = int(n) if n and n > 0 else 0
    # Skip emission entirely for zero hits so we don't create per-case zero
    # buckets / cardinality for cases that had no retrieval.
    if count:
        _instrument("rca_memory_retrieval_hits").add(count, {"case_id": str(case_id)})


def record_llm_tokens(model: str, prompt: int, completion: int) -> None:
    """Increment ``rca_llm_tokens`` for prompt and completion token counts."""
    p = int(prompt) if prompt and prompt > 0 else 0
    c = int(completion) if completion and completion > 0 else 0
    counter = _instrument("rca_llm_tokens")
    if p:
        counter.add(p, {"kind": "prompt", "model": str(model)})
    if c:
        counter.add(c, {"kind": "completion", "model": str(model)})


def record_step(step_kind: str) -> None:
    """Increment ``rca_steps_total{step_kind}``."""
    _instrument("rca_steps_total").add(1, {"step_kind": str(step_kind)})


def record_run(status: str) -> None:
    """Increment ``rca_runs_total{status}``."""
    _instrument("rca_runs_total").add(1, {"status": str(status)})


# --------------------------------------------------------------------------- #
# Evaluation recorder scaffold (minimal — full eval is another unit)
# --------------------------------------------------------------------------- #
@runtime_checkable
class EvaluationRecorder(Protocol):
    """Protocol for recording agent-evaluation metrics for a case.

    This is an intentionally thin scaffold. The future agent-evaluation unit
    will provide concrete backends (e.g. a ClickHouse / MLflow recorder); for
    now the default :class:`InMemoryEvaluationRecorder` lets tests and local
    experimentation capture per-case metrics without any external dependency.
    """

    def record(self, case_id: str, metric: str, value: float) -> None: ...


class InMemoryEvaluationRecorder:
    """Default in-memory evaluation recorder (scaffold)."""

    def __init__(self) -> None:
        self._records: list[tuple[str, str, float]] = []

    def record(self, case_id: str, metric: str, value: float) -> None:
        self._records.append((str(case_id), str(metric), float(value)))

    @property
    def records(self) -> list[tuple[str, str, float]]:
        """List of ``(case_id, metric, value)`` tuples in insertion order."""
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()


_evaluation_recorder: EvaluationRecorder = InMemoryEvaluationRecorder()


def get_evaluation_recorder() -> EvaluationRecorder:
    """Return the currently-installed evaluation recorder."""
    return _evaluation_recorder


def set_evaluation_recorder(recorder: EvaluationRecorder) -> None:
    """Install a custom evaluation recorder (e.g. a real backend)."""
    global _evaluation_recorder
    _evaluation_recorder = recorder
