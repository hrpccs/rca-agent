"""Offline tests for the eval runner.

Injects a fake agent (no LLM, no network) via ``run_eval(agent_factory=...)`` so
the per-case metrics + ``runs/eval_summary.{json,csv}`` shape can be asserted
without any real DeepSeek call. The fake agent yields a small trace (2 steps +
1 report) per case, exercising the same async-iteration path the real agent
uses.
"""
from __future__ import annotations

import csv
import json
from collections.abc import AsyncIterator
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

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
from rca_agent.eval import runner


# --------------------------------------------------------------------------- #
# Fixtures: a minimal Case + a fake agent factory
# --------------------------------------------------------------------------- #
def _make_case(case_id: str) -> Case:
    from datetime import datetime

    tw = TimeWindow(
        start=datetime(2026, 4, 25, 5, 18, 12, tzinfo=UTC),
        end=datetime(2026, 4, 25, 5, 28, 12, tzinfo=UTC),
    )
    task = Task(
        task_id=case_id,
        alert_title=f"alert for {case_id}",
        alert_window=tw,
        prompt_text="analyze root cause",
        available_modalities=[Modality.METRICS, Modality.LOGS],
    )
    return Case(task=task, topology=Topology(case_id=case_id, window=tw), case_dir="/tmp/fake")


class _FakeAgent:
    """Yields 2 RcaStep (1 reasoning + 1 tool_call) then 1 RcaReport.

    The report's root_cause is populated so the metrics dict exercises every
    richness field (fault_type, entities, evidence, actions).
    """

    def __init__(self, case_id: str, *, fault_type: str | None = "k8s.pod_crashloop") -> None:
        self.case_id = case_id
        self._fault_type = fault_type

    async def run(self, case: Case) -> AsyncIterator[RcaStep | RcaReport]:  # type: ignore[override]
        cid = self.case_id
        yield RcaStep(
            step_id=f"{cid}-s1",
            case_id=cid,
            step_kind=StepKind.REASONING,
            thought="investigating",
        )
        yield RcaStep(
            step_id=f"{cid}-s2",
            case_id=cid,
            step_kind=StepKind.TOOL_CALL,
            tool_name="query_logs",
        )
        yield RcaReport(
            case_id=cid,
            task_id=cid,
            alert_title=f"alert for {cid}",
            root_cause=RootCause(
                summary="bad pod",
                entity_refs=[{"entity_id": "pod-1", "entity_name": "checkout-pod"}],
                fault_type=self._fault_type,
                evidence=["step-1", "step-2"],
                confidence=0.8,
                recommended_actions=["restart pod"],
            ),
            steps=[
                RcaStep(step_id=f"{cid}-s1", case_id=cid, step_kind=StepKind.REASONING),
                RcaStep(step_id=f"{cid}-s2", case_id=cid, step_kind=StepKind.TOOL_CALL, tool_name="query_logs"),
            ],
            token_usage={"total_tokens": 42},
            status="completed",
        )


def _fake_factory(fault_type: str | None = "k8s.pod_crashloop"):
    """Build a (case, fake-agent) pair for a given case id — matches the
    AgentFactory Protocol."""

    def _factory(case_id: str, backend: str = "parquet") -> tuple[Case, _FakeAgent]:
        return _make_case(case_id), _FakeAgent(case_id, fault_type=fault_type)

    return _factory


# --------------------------------------------------------------------------- #
# Per-case metrics + summary artifacts
# --------------------------------------------------------------------------- #
class TestRunEvalMetrics:
    async def test_single_case_metrics_shape(self, tmp_path: Path):
        results = await runner.run_eval(
            cases=["t-fake-1"],
            out_dir=str(tmp_path),
            agent_factory=_fake_factory(),
        )
        assert len(results) == 1
        m = results[0]
        # Per-case metric keys the runner must record (per the eval contract)
        assert m["case_id"] == "t-fake-1"
        assert m["task_id"] == "t-fake-1"
        assert m["status"] == "completed"
        assert m["confidence"] == pytest.approx(0.8)
        assert m["has_fault_type"] is True
        assert m["fault_type"] == "k8s.pod_crashloop"
        assert m["n_entities"] == 1
        assert m["n_evidence"] == 2
        assert m["n_steps"] == 2
        assert m["n_tool_calls"] == 1
        assert m["tokens"] == 42
        assert "elapsed_s" in m and isinstance(m["elapsed_s"], float)

    async def test_two_cases_each_recorded(self, tmp_path: Path):
        results = await runner.run_eval(
            cases=["t-fake-1", "t-fake-2"],
            out_dir=str(tmp_path),
            agent_factory=_fake_factory(),
        )
        assert [r["case_id"] for r in results] == ["t-fake-1", "t-fake-2"]
        assert all(r["status"] == "completed" for r in results)

    async def test_no_fault_type_flagged(self, tmp_path: Path):
        results = await runner.run_eval(
            cases=["t-fake-1"],
            out_dir=str(tmp_path),
            agent_factory=_fake_factory(fault_type=None),
        )
        assert results[0]["has_fault_type"] is False
        assert results[0]["fault_type"] is None


class TestRunEvalArtifacts:
    async def test_summary_json_created_with_right_shape(self, tmp_path: Path):
        await runner.run_eval(
            cases=["t-fake-1", "t-fake-2"],
            out_dir=str(tmp_path),
            agent_factory=_fake_factory(),
        )
        summary_path = tmp_path / "eval_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert set(summary.keys()) == {"aggregate", "results"}
        agg = summary["aggregate"]
        # Aggregate must summarize what the per-case metrics expose
        assert agg["n_cases"] == 2
        assert agg["n_completed"] == 2
        assert agg["n_errors"] == 0
        assert agg["avg_confidence"] == pytest.approx(0.8)
        assert agg["avg_entities"] == pytest.approx(1.0)
        assert agg["avg_evidence"] == pytest.approx(2.0)
        assert agg["pct_has_fault_type"] == pytest.approx(100.0)
        assert len(summary["results"]) == 2

    async def test_summary_csv_created_with_header_and_rows(self, tmp_path: Path):
        await runner.run_eval(
            cases=["t-fake-1"],
            out_dir=str(tmp_path),
            agent_factory=_fake_factory(),
        )
        csv_path = tmp_path / "eval_summary.csv"
        assert csv_path.exists()
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        row = rows[0]
        assert row["case_id"] == "t-fake-1"
        assert row["status"] == "completed"
        assert row["confidence"] == "0.8"
        # CSV uses the canonical column set (extrasaction='ignore')
        for col in ["case_id", "status", "confidence", "has_fault_type", "fault_type",
                    "n_entities", "n_evidence", "n_steps", "n_tool_calls", "tokens",
                    "elapsed_s", "error"]:
            assert col in row

    async def test_per_case_report_json_written(self, tmp_path: Path):
        await runner.run_eval(
            cases=["t-fake-1"],
            out_dir=str(tmp_path),
            agent_factory=_fake_factory(),
        )
        report_path = tmp_path / "t-fake-1.report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["case_id"] == "t-fake-1"
        assert report["root_cause"]["summary"] == "bad pod"


# --------------------------------------------------------------------------- #
# Error path + factory injection
# --------------------------------------------------------------------------- #
class TestRunEvalErrorHandling:
    async def test_case_exception_recorded_not_raised(self, tmp_path: Path):
        def boom_factory(case_id: str, backend: str = "parquet"):
            raise RuntimeError("agent blew up")

        results = await runner.run_eval(
            cases=["t-fake-1"],
            out_dir=str(tmp_path),
            agent_factory=boom_factory,
        )
        assert len(results) == 1
        m = results[0]
        assert m["status"] == "error"
        # case_id is in scope at the call site even when the agent blew up
        # before emitting a report; it must be carried into the metrics row so
        # the failed case is identifiable in eval_summary.csv (not a blank cell).
        assert m["case_id"] == "t-fake-1"
        assert "RuntimeError" in m["error"]
        assert "agent blew up" in m["error"]
        assert isinstance(m["elapsed_s"], float)

    async def test_failed_case_appears_in_csv_with_case_id(self, tmp_path: Path):
        def boom_factory(case_id: str, backend: str = "parquet"):
            raise RuntimeError("nope")

        await runner.run_eval(
            cases=["t-fake-err"],
            out_dir=str(tmp_path),
            agent_factory=boom_factory,
        )
        csv_path = tmp_path / "eval_summary.csv"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["case_id"] == "t-fake-err"
        assert rows[0]["status"] == "error"

    async def test_factory_default_when_none(self, tmp_path: Path, monkeypatch):
        # When agent_factory is None, the runner falls back to build_agent_for_case.
        # We monkeypatch that import target so the default path is exercised
        # without a real LLM.
        called: dict[str, Any] = {}

        def fake_build(case_id, backend="parquet"):
            called["case_id"] = case_id
            called["backend"] = backend
            return _make_case(case_id), _FakeAgent(case_id)

        import rca_agent.eval.runner as runner_mod

        monkeypatch.setattr(runner_mod, "build_agent_for_case", fake_build)
        await runner.run_eval(
            cases=["t-fake-1"],
            out_dir=str(tmp_path),
            agent_factory=None,
        )
        assert called == {"case_id": "t-fake-1", "backend": "parquet"}
