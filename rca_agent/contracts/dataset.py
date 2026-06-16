"""Data models mirroring the on-disk benchmark case (task.json + topology.json).

Pure data; I/O lives in :mod:`rca_agent.providers`. Matches the verified schemas
of the rca100 dataset.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ._primitives import Modality, TimeWindow


class Task(BaseModel):
    task_id: str
    task_version: str | None = None
    alert_event_id: str | None = None
    alert_title: str
    alert_trigger_time: str = ""
    alert_window: TimeWindow
    alert_entity: dict[str, Any] = Field(
        default_factory=dict
    )  # {entity_id,name,type,domain} — any may be null
    prompt_text: str
    workspace: str | None = None
    region_id: str | None = None
    available_modalities: list[Modality] = Field(default_factory=list)
    scoring_note: str | None = None


class TopologyEntity(BaseModel):
    id: str
    type: str
    name: str
    first_observed: int | None = None
    last_observed: int | None = None
    props: dict[str, Any] = Field(default_factory=dict)


class TopologyEdge(BaseModel):
    src: str
    src_type: str
    dst: str
    dst_type: str
    relation: str
    first_observed: int | None = None
    last_observed: int | None = None


class Topology(BaseModel):
    case_id: str
    source: str | None = None
    window: TimeWindow
    cluster_id: str | None = None
    entities: list[TopologyEntity] = Field(default_factory=list)
    edges: list[TopologyEdge] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class Case(BaseModel):
    """A bundle representing a single benchmark case on disk."""

    task: Task
    topology: Topology
    case_dir: str  # absolute path to the case folder (parquet root)
    modalities: list[Modality] = Field(default_factory=list)


__all__ = ["Task", "TopologyEntity", "TopologyEdge", "Topology", "Case"]
