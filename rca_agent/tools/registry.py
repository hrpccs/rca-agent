"""Default tool registry for the RCA agent.

``build_default_tools()`` returns the builtin SRE investigation tools as a list
of :class:`RegisteredTool`, ready to be turned into an OpenAI ``tools`` array
via :func:`build_openai_tools`.

The handler functions are pure: ``(args, provider, memory) -> ToolResult``.
They are NOT pre-bound to a particular provider/memory here — the runtime
supplies the bound (provider, memory) at dispatch time (see
``ToolHandler.__call__``). Keeping them unbound keeps them trivially unit
testable against fakes.
"""
from __future__ import annotations

from typing import Any

from ..contracts import RegisteredTool, ToolSpec, build_openai_tools, validate_tool_call
from .builtin import (
    GetTopologyArgs,
    InspectEntityArgs,
    QueryAlertsArgs,
    QueryEventsArgs,
    QueryLogsArgs,
    QueryMetricsArgs,
    QueryTracesArgs,
    StoreObservationArgs,
    get_topology,
    inspect_entity,
    query_alerts,
    query_events,
    query_logs,
    query_metrics,
    query_traces,
    store_observation,
)

# Re-export for convenience so callers can import everything from one place.
__all__ = [
    "build_default_tools",
    "build_openai_tools",
    "validate_tool_call",
]


# (spec-name, description, args_model, handler)
_TOOL_DEFS: list[tuple[str, str, type, Any]] = [
    (
        "query_alerts",
        "Fetch alert rows (CNCF CloudEvents) within the case alert window. Use this first to understand what fired.",
        QueryAlertsArgs,
        query_alerts,
    ),
    (
        "query_events",
        "Fetch Kubernetes events (Warning/Normal) within the case window; useful for pod evictions, crashes, scheduling failures.",
        QueryEventsArgs,
        query_events,
    ),
    (
        "query_metrics",
        "Fetch metric series (k8s/apm) within the case window to detect anomalies: CPU, memory, latency, error rates.",
        QueryMetricsArgs,
        query_metrics,
    ),
    (
        "query_logs",
        "Fetch application/container log lines within the case window; filter by pod/namespace/contains/level.",
        QueryLogsArgs,
        query_logs,
    ),
    (
        "query_traces",
        "Fetch distributed traces within the case window; find slow/error spans and the responsible service.",
        QueryTracesArgs,
        query_traces,
    ),
    (
        "get_topology",
        "Fetch the service/pod topology subgraph to understand dependencies and blast radius around an entity.",
        GetTopologyArgs,
        get_topology,
    ),
    (
        "inspect_entity",
        "Look up a single topology entity by id or name and return its properties + neighbors. Use before concluding.",
        InspectEntityArgs,
        inspect_entity,
    ),
    (
        "store_observation",
        "Persist a key observation/hypothesis to agent memory so it can be reused across the investigation.",
        StoreObservationArgs,
        store_observation,
    ),
]


def build_default_tools(provider: Any = None, memory: Any = None) -> list[RegisteredTool]:
    """Build the default SRE investigation toolkit.

    ``provider`` / ``memory`` are accepted for call-site symmetry with the
    runtime: handlers are pure ``(args, provider, memory) -> dict`` and the
    dispatch layer supplies the bound provider/memory at invocation time (they
    are not captured here). Passing them is optional but conventional.
    """
    tools: list[RegisteredTool] = []
    for name, desc, args_model, handler in _TOOL_DEFS:
        spec = ToolSpec(name=name, description=desc, args_model=args_model)
        tools.append(RegisteredTool(spec=spec, handler=handler))
    return tools
