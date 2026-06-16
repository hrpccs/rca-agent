"""Shared pytest fixtures and hooks for the RCA agent test suite."""
from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from rca_agent.contracts import (
    Case,
    ContextState,
    EntityRef,
    MemoryItem,
    MemoryQuery,
    Modality,
    Task,
    TimeWindow,
    Topology,
)


def pytest_collection_modifyitems(config, items):
    """Auto-skip `@pytest.mark.live` tests when no DeepSeek key / infra is set."""
    live_key = os.getenv("RCA_DEEPSEEK_API_KEY", "")
    has_live = bool(live_key) and not live_key.startswith("sk-x")
    skip_live = pytest.mark.skip(
        reason="live test: set RCA_DEEPSEEK_API_KEY (and bring up infra) to run"
    )
    for item in items:
        if "live" in item.keywords and not has_live:
            item.add_marker(skip_live)


# --------------------------------------------------------------------------- #
# Fakes (reusable across workers; do not depend on any implementation module)
# --------------------------------------------------------------------------- #
@pytest.fixture
def time_window() -> TimeWindow:
    from datetime import datetime, timezone

    return TimeWindow(
        start=datetime(2026, 4, 25, 5, 18, 12, tzinfo=timezone.utc),
        end=datetime(2026, 4, 25, 5, 28, 12, tzinfo=timezone.utc),
        start_us=1777094292716735,
        end_us=1777094892716735,
    )


@pytest.fixture
def sample_task(time_window: TimeWindow) -> Task:
    return Task(
        task_id="t001",
        alert_title="checkout 错误次数告警",
        alert_window=time_window,
        alert_entity={"entity_id": "d219413245b68b297976412bbee076cf", "entity_type": "apm.operation"},
        prompt_text="帮我分析下根因。",
        available_modalities=[Modality.METRICS, Modality.LOGS, Modality.TRACES],
    )


@pytest.fixture
def sample_topology(time_window: TimeWindow) -> Topology:
    return Topology(case_id="t001", window=time_window)


@pytest.fixture
def sample_case(sample_task: Task, sample_topology: Topology) -> Case:
    return Case(
        task=sample_task,
        topology=sample_topology,
        case_dir="/tmp/fake-case-t001",
        modalities=sample_task.available_modalities,
    )


class FakeMemoryStore:
    """Minimal in-memory MemoryStore for tool/agent tests."""

    def __init__(self) -> None:
        self._items: dict[str, list[MemoryItem]] = {}

    def index(self, items: list[MemoryItem]) -> None:
        for it in items:
            self._items.setdefault(it.case_id, []).append(it)

    def retrieve(self, q: MemoryQuery) -> list[MemoryItem]:
        pool = self._items.get(q.case_id, []) + self._items.get("__global__", [])
        if q.kind:
            pool = [i for i in pool if i.kind == q.kind]
        if q.text:
            tl = q.text.lower()
            scored = [(i, i.content.lower().count(tl)) for i in pool]
            pool = [i for i, _ in sorted(scored, key=lambda x: -x[1]) if any(tl in s.lower() for s in [i.content])]
        return pool[: q.top_k]

    def retrieve_for_context(self, case_id, query, top_k=8):
        return self.retrieve(MemoryQuery(case_id=case_id, text=query, top_k=top_k))

    def clear(self, case_id=None):
        if case_id is None:
            self._items.clear()
        else:
            self._items.pop(case_id, None)


@pytest.fixture
def fake_memory() -> FakeMemoryStore:
    return FakeMemoryStore()


@pytest.fixture
def new_step_id() -> Iterator[str]:
    """Deterministic-ish step id generator for tests."""
    counter = 0

    def gen() -> str:
        nonlocal counter
        counter += 1
        return f"step-{counter}-{uuid.uuid4().hex[:6]}"

    yield gen
