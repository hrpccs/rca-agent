"""Frozen contracts for the RCA agent.

These are the integration seams every module programs against. **Do not edit in
worker PRs** — propose changes to the coordinator. All contracts depend only on
stdlib + pydantic, never on an implementation module.
"""
from __future__ import annotations

from ._primitives import EntityRef, Modality, Severity, TimeWindow, utcnow
from .context import ContextManager, ContextState, ToolMessage, TurnRecord
from .dataset import Case, Task, Topology, TopologyEdge, TopologyEntity
from .llm import DeltaKind, LLMClient, LLMRequest, LLMStreamDelta
from .memory import MemoryItem, MemoryQuery, MemoryStore
from .provider import (
    AlertFilter,
    CloudEvent,
    DataProvider,
    EventFilter,
    K8sEvent,
    LogFilter,
    LogLine,
    MetricFilter,
    MetricSeries,
    Span,
    Trace,
    TraceFilter,
    TopologyFilter,
    TopologySubgraph,
)
from .rca import RootCause, RcaReport, RcaStep, RcaTrace, StepKind
from .streaming import SSEEvent, SSEEventKind, SSEDelta, sse_format
from .tools import (
    RegisteredTool,
    ToolCall,
    ToolHandler,
    ToolResult,
    ToolSpec,
    build_openai_tools,
    validate_tool_call,
)

__all__ = [
    # primitives
    "Modality",
    "Severity",
    "TimeWindow",
    "EntityRef",
    "utcnow",
    # dataset
    "Task",
    "Topology",
    "TopologyEntity",
    "TopologyEdge",
    "Case",
    # provider
    "MetricFilter",
    "LogFilter",
    "TraceFilter",
    "EventFilter",
    "AlertFilter",
    "TopologyFilter",
    "MetricSeries",
    "LogLine",
    "Span",
    "Trace",
    "K8sEvent",
    "CloudEvent",
    "TopologySubgraph",
    "DataProvider",
    # tools
    "ToolResult",
    "ToolCall",
    "ToolSpec",
    "ToolHandler",
    "RegisteredTool",
    "build_openai_tools",
    "validate_tool_call",
    # memory
    "MemoryItem",
    "MemoryQuery",
    "MemoryStore",
    # context
    "ToolMessage",
    "TurnRecord",
    "ContextState",
    "ContextManager",
    # llm
    "DeltaKind",
    "LLMStreamDelta",
    "LLMRequest",
    "LLMClient",
    # rca
    "StepKind",
    "RcaStep",
    "RootCause",
    "RcaReport",
    "RcaTrace",
    # streaming
    "SSEEventKind",
    "SSEDelta",
    "SSEEvent",
    "sse_format",
]
