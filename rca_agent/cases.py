"""Benchmark case discovery + loading (shared infrastructure).

Multiple modules (providers, loader, agent, eval, cli) need to materialize a
:class:`rca_agent.contracts.Case` from the on-disk benchmark layout, so this
lives in the foundation rather than being reimplemented per-module. It only
parses JSON metadata (task.json + topology.json) — reading parquet is the
provider's job.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import get_settings
from .contracts import Case, Modality, Task, TimeWindow, Topology, TopologyEdge, TopologyEntity


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    # Python 3.11 fromisoformat handles offsets + fractional seconds.
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _window_from_task(d: dict) -> TimeWindow:
    start = _parse_iso(d.get("start")) or datetime.fromtimestamp(0)
    end = _parse_iso(d.get("end")) or start
    return TimeWindow(start=start, end=end)


def _window_from_topology(d: dict) -> TimeWindow:
    start = _parse_iso(d.get("start_iso")) or _parse_iso(d.get("start")) or datetime.fromtimestamp(0)
    end = _parse_iso(d.get("end_iso")) or _parse_iso(d.get("end")) or start
    return TimeWindow(
        start=start,
        end=end,
        start_us=d.get("start_us"),
        end_us=d.get("end_us"),
    )


def list_cases(cases_dir: Path | None = None) -> list[str]:
    """Return sorted case ids (directory names that contain a task.json)."""
    root = Path(cases_dir) if cases_dir else get_settings().cases_dir
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "task.json").exists()
    )


def case_dir(case_id: str, cases_dir: Path | None = None) -> Path:
    root = Path(cases_dir) if cases_dir else get_settings().cases_dir
    return root / case_id


def case_file(case_id: str, name: str, cases_dir: Path | None = None) -> Path:
    """Path to a file in the case dir, e.g. case_file('t001', 'logs.parquet')."""
    return case_dir(case_id, cases_dir) / name


def load_task(case_id: str, cases_dir: Path | None = None) -> Task:
    d = json.loads(case_file(case_id, "task.json", cases_dir).read_text())
    modalities = [Modality(m) for m in d.get("available_modalities", [])]
    return Task(
        task_id=d["task_id"],
        task_version=d.get("task_version"),
        alert_event_id=d.get("alert_event_id"),
        alert_title=d.get("alert_title", ""),
        alert_trigger_time=d.get("alert_trigger_time", ""),
        alert_window=_window_from_task(d.get("alert_window", {})),
        alert_entity=d.get("alert_entity") or {},
        prompt_text=d.get("prompt_text", ""),
        workspace=d.get("workspace"),
        region_id=d.get("region_id"),
        available_modalities=modalities,
        scoring_note=d.get("scoring_note"),
    )


def load_topology(case_id: str, cases_dir: Path | None = None) -> Topology:
    d = json.loads(case_file(case_id, "topology.json", cases_dir).read_text())
    entities = [TopologyEntity(**e) for e in d.get("entities", [])]
    edges = [TopologyEdge(**e) for e in d.get("edges", [])]
    return Topology(
        case_id=d.get("case_id", case_id),
        source=d.get("source"),
        window=_window_from_topology(d.get("window", {})),
        cluster_id=d.get("cluster_id"),
        entities=entities,
        edges=edges,
        stats=d.get("stats", {}),
    )


def load_case(case_id: str, cases_dir: Path | None = None) -> Case:
    """Materialize a Case (task + topology + on-disk paths) for a benchmark case."""
    task = load_task(case_id, cases_dir)
    topology = load_topology(case_id, cases_dir)
    cdir = str(case_dir(case_id, cases_dir))
    return Case(
        task=task,
        topology=topology,
        case_dir=cdir,
        modalities=task.available_modalities or list(Modality),
    )


__all__ = ["list_cases", "case_dir", "case_file", "load_task", "load_topology", "load_case"]
