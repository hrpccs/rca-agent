"""Unit tests for rca_agent.cases (discovery + task/topology loading).

These tests build a synthetic on-disk cases dir under ``tmp_path`` mirroring the
real rca100 layout (``<case_id>/{task.json, topology.json, *.parquet}``) and
exercise ``list_cases`` / ``load_task`` / ``load_topology`` / ``load_case`` /
``case_dir`` / ``case_file``. They never touch the real ``RCA_CASES_DIR`` and
never read parquet (loading only parses JSON metadata, as the module docstring
documents).

Malformed-input cases assert that clear, typed errors surface:
  * missing topology.json -> FileNotFoundError from load_case/load_topology
  * bad ISO-8601 time window in task.json -> ValueError (datetime.fromisoformat)
  * missing task.json dir -> silently skipped by list_cases, but load_task on a
    present-but-empty dir -> FileNotFoundError
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from rca_agent.cases import (
    case_dir,
    case_file,
    list_cases,
    load_case,
    load_task,
    load_topology,
)
from rca_agent.config import get_settings
from rca_agent.contracts import Case, Modality, Task, Topology

# --------------------------------------------------------------------------- #
# Synthetic on-disk case factory (mirrors the real rca100 schema).
# --------------------------------------------------------------------------- #
_ISO_START = "2026-04-25T05:18:12.716735+00:00"
_ISO_END = "2026-04-25T05:28:12.716735+00:00"
_US_START = 1777094292716735
_US_END = 1777094892716735


def _task_dict(case_id: str, *, modalities: list[str] | None = None) -> dict:
    """A task.json payload matching the real rca100 schema (subset of fields the
    loader actually reads, plus a few it ignores)."""
    return {
        "task_id": case_id,
        "task_version": "1.0",
        "alert_event_id": f"evt-{case_id}",
        "alert_title": f"{case_id} 错误次数告警",
        "alert_trigger_time": "",
        "alert_window": {"start": _ISO_START, "end": _ISO_END},
        "alert_entity": {
            "entity_id": "d219413245b68b297976412bbee076cf",
            "entity_name": "checkout::/oteldemo.CheckoutService/PlaceOrder",
            "entity_type": "apm.operation",
            "entity_domain": "apm",
        },
        "prompt_text": "帮我分析下根因。",
        "workspace": "rca-benchmark",
        "region_id": "cn-hongkong",
        "available_modalities": modalities
        if modalities is not None
        else ["metrics", "logs", "traces", "events", "alerts", "topology"],
        "scoring_note": "n/a",
    }


def _topology_dict(case_id: str) -> dict:
    """A topology.json payload matching the real rca100 schema."""
    return {
        "case_id": case_id,
        "source": "umodel_v3",
        "window": {
            "start_iso": _ISO_START,
            "end_iso": _ISO_END,
            "start_us": _US_START,
            "end_us": _US_END,
        },
        "cluster_id": "cfbbc0eabc19d43c0a6fb6889b4451ad0",
        "entities": [
            {
                "id": "70af2dee3d886988a1f1baefbf5fc400",
                "type": "apm.service",
                "name": "recommendation",
                "first_observed": 0,
                "last_observed": 1777000496,
                "props": {"service": "recommendation", "language": "python"},
            },
            {
                "id": "11111111111111111111111111111111",
                "type": "apm.operation",
                "name": "checkout::/oteldemo.CheckoutService/PlaceOrder",
                "first_observed": 1777000000,
                "last_observed": 1777009999,
                "props": {},
            },
        ],
        "edges": [
            {
                "src": "70af2dee3d886988a1f1baefbf5fc400",
                "src_type": "apm.service",
                "dst": "11111111111111111111111111111111",
                "dst_type": "apm.operation",
                "relation": "contains",
                "first_observed": 1777000000,
                "last_observed": 1777009999,
            },
        ],
        "stats": {
            "entities_total": 2,
            "edges_total": 1,
            "entities_by_type": {"apm.service": 1, "apm.operation": 1},
            "edges_by_relation": {"contains": 1},
        },
    }


def _write_case(
    root: Path,
    case_id: str,
    *,
    task: dict | None = None,
    topology: dict | None = None,
    parquet_names: tuple[str, ...] = ("metrics.parquet", "logs.parquet"),
    write_task: bool = True,
    write_topology: bool = True,
) -> Path:
    """Create <root>/<case_id>/ with task.json/topology.json + zero-byte parquet
    placeholders (loader never reads parquet, so empty files are fine)."""
    d = root / case_id
    d.mkdir(parents=True, exist_ok=True)
    if write_task:
        (d / "task.json").write_text(json.dumps(task or _task_dict(case_id)))
    if write_topology:
        (d / "topology.json").write_text(json.dumps(topology or _topology_dict(case_id)))
    # Parquet placeholders: the real layout has these siblings; presence asserts
    # list_cases / case_file resolve them by name without parsing them.
    for name in parquet_names:
        (d / name).write_bytes(b"")
    return d


@pytest.fixture
def cases_root(tmp_path: Path) -> Path:
    """A tmp cases dir with two synthetic cases mirroring the real layout, plus
    a stray dir/file that must be ignored by discovery."""
    root = tmp_path / "cases"
    root.mkdir()
    _write_case(root, "t001")
    _write_case(root, "t002", task=_task_dict("t002", modalities=["metrics", "logs"]))
    # Stray dir WITHOUT task.json -> must be skipped by list_cases.
    (root / "not-a-case").mkdir()
    (root / "not-a-case" / "README.md").write_text("ignore me")
    # Stray loose file at the root -> must be skipped.
    (root / "loose.txt").write_text("ignore me too")
    return root


# The ``_clear_settings_cache`` autouse fixture lives in tests/cases/conftest.py.
def test_list_cases_returns_sorted_case_ids(cases_root: Path):
    ids = list_cases(cases_root)
    assert ids == ["t001", "t002"]
    assert ids == sorted(ids)


def test_list_cases_ignores_dirs_without_task_json(cases_root: Path):
    ids = set(list_cases(cases_root))
    assert "not-a-case" not in ids


def test_list_cases_explicit_param_overrides_settings_env(monkeypatch, cases_root: Path):
    """Even if RCA_CASES_DIR points elsewhere, the explicit arg wins."""
    monkeypatch.setenv("RCA_CASES_DIR", "/nonexistent/path/from/env")
    get_settings.cache_clear()
    assert list_cases(cases_root) == ["t001", "t002"]


def test_list_cases_falls_back_to_settings_cases_dir(monkeypatch, cases_root: Path):
    monkeypatch.setenv("RCA_CASES_DIR", str(cases_root))
    get_settings.cache_clear()
    assert list_cases() == ["t001", "t002"]


def test_list_cases_nonexistent_dir_returns_empty(monkeypatch):
    monkeypatch.setenv("RCA_CASES_DIR", "/definitely/does/not/exist/xyz")
    get_settings.cache_clear()
    assert list_cases() == []


def test_list_cases_empty_root(tmp_path: Path):
    assert list_cases(tmp_path) == []


# --------------------------------------------------------------------------- #
# case_dir / case_file
# --------------------------------------------------------------------------- #
def test_case_dir_resolves_subpath(cases_root: Path):
    assert case_dir("t001", cases_root) == cases_root / "t001"


def test_case_dir_falls_back_to_settings(monkeypatch, cases_root: Path):
    monkeypatch.setenv("RCA_CASES_DIR", str(cases_root))
    get_settings.cache_clear()
    assert case_dir("t001") == cases_root / "t001"


def test_case_file_joins_name(cases_root: Path):
    assert case_file("t001", "logs.parquet", cases_root) == cases_root / "t001" / "logs.parquet"


# --------------------------------------------------------------------------- #
# load_task
# --------------------------------------------------------------------------- #
def test_load_task_round_trips_fields(cases_root: Path):
    t = load_task("t001", cases_root)
    assert isinstance(t, Task)
    assert t.task_id == "t001"
    assert t.task_version == "1.0"
    assert t.alert_event_id == "evt-t001"
    assert t.alert_title == "t001 错误次数告警"
    assert t.prompt_text == "帮我分析下根因。"
    assert t.workspace == "rca-benchmark"
    assert t.region_id == "cn-hongkong"
    assert t.scoring_note == "n/a"
    assert t.available_modalities == list(Modality)  # t001 has all six


def test_load_task_alert_window_parsed_as_datetimes(cases_root: Path):
    t = load_task("t001", cases_root)
    assert isinstance(t.alert_window.start, datetime)
    assert isinstance(t.alert_window.end, datetime)
    assert t.alert_window.start.tzinfo is not None  # tz-aware
    assert t.alert_window.end > t.alert_window.start


def test_load_task_modalities_subset(cases_root: Path):
    t = load_task("t002", cases_root)
    assert t.available_modalities == [Modality.METRICS, Modality.LOGS]


def test_load_task_missing_file_raises_filenotfound(cases_root: Path):
    # t001 has task.json but load_task on a non-existent id must raise.
    with pytest.raises(FileNotFoundError):
        load_task("nope", cases_root)


# --------------------------------------------------------------------------- #
# load_topology
# --------------------------------------------------------------------------- #
def test_load_topology_round_trips_fields(cases_root: Path):
    topo = load_topology("t001", cases_root)
    assert isinstance(topo, Topology)
    assert topo.case_id == "t001"
    assert topo.source == "umodel_v3"
    assert topo.cluster_id == "cfbbc0eabc19d43c0a6fb6889b4451ad0"
    assert len(topo.entities) == 2
    assert len(topo.edges) == 1
    assert topo.entities[0].name == "recommendation"
    assert topo.entities[0].props["language"] == "python"
    assert topo.edges[0].relation == "contains"
    assert topo.stats["entities_total"] == 2


def test_load_topology_window_carries_us_fields(cases_root: Path):
    topo = load_topology("t001", cases_root)
    assert topo.window.start_us == _US_START
    assert topo.window.end_us == _US_END
    assert isinstance(topo.window.start, datetime)
    assert isinstance(topo.window.end, datetime)


def test_load_topology_missing_file_raises_filenotfound(cases_root: Path):
    # Build a case dir with task.json but NO topology.json.
    _write_case(cases_root, "t-no-topo", write_topology=False)
    with pytest.raises(FileNotFoundError):
        load_topology("t-no-topo", cases_root)


# --------------------------------------------------------------------------- #
# load_case
# --------------------------------------------------------------------------- #
def test_load_case_round_trips(cases_root: Path):
    c = load_case("t001", cases_root)
    assert isinstance(c, Case)
    assert c.task.task_id == "t001"
    assert c.topology.case_id == "t001"
    assert c.case_dir == str(cases_root / "t001")
    assert c.modalities == list(Modality)


def test_load_case_modalities_default_to_all_when_task_empty(cases_root: Path):
    """Contract: when task.available_modalities is empty, Case.modalities falls
    back to ALL modalities (list(Modality))."""
    _write_case(
        cases_root,
        "t-empty-mods",
        task=_task_dict("t-empty-mods", modalities=[]),
    )
    c = load_case("t-empty-mods", cases_root)
    assert c.task.available_modalities == []
    assert c.modalities == list(Modality)


def test_load_case_uses_task_modalities_when_present(cases_root: Path):
    c = load_case("t002", cases_root)
    assert c.modalities == [Modality.METRICS, Modality.LOGS]


def test_load_case_missing_topology_raises_filenotfound(cases_root: Path):
    """Missing topology.json must surface a clear, typed FileNotFoundError from
    load_case (not a KeyError / AttributeError)."""
    _write_case(cases_root, "t-no-topo2", write_topology=False)
    with pytest.raises(FileNotFoundError):
        load_case("t-no-topo2", cases_root)


# --------------------------------------------------------------------------- #
# Malformed input: bad ISO time window
# --------------------------------------------------------------------------- #
def test_load_task_bad_iso_window_raises_valueerror(cases_root: Path):
    """A non-ISO timestamp in alert_window.start must raise a clear ValueError
    (datetime.fromisoformat rejects it) — not silently coerce to None."""
    bad_task = _task_dict("t-bad-iso")
    bad_task["alert_window"]["start"] = "not-a-real-timestamp"
    _write_case(cases_root, "t-bad-iso", task=bad_task)
    with pytest.raises(ValueError):
        load_task("t-bad-iso", cases_root)


def test_load_case_bad_iso_window_raises_valueerror(cases_root: Path):
    bad_task = _task_dict("t-bad-iso2")
    bad_task["alert_window"]["end"] = "2026/13/45 99:99:99"  # not ISO-8601
    _write_case(cases_root, "t-bad-iso2", task=bad_task)
    with pytest.raises(ValueError):
        load_case("t-bad-iso2", cases_root)


def test_load_topology_bad_iso_window_raises_valueerror(cases_root: Path):
    bad_topo = _topology_dict("t-bad-topo-iso")
    bad_topo["window"]["start_iso"] = "garbage"
    _write_case(cases_root, "t-bad-topo-iso", topology=bad_topo)
    with pytest.raises(ValueError):
        load_topology("t-bad-topo-iso", cases_root)


# --------------------------------------------------------------------------- #
# Optional-field fallbacks the loader implements
# --------------------------------------------------------------------------- #
def test_load_task_optional_fields_default_when_absent(cases_root: Path):
    """task.json with several optional keys omitted still loads with defaults."""
    minimal = {
        "task_id": "t-min",
        "alert_title": "x",
        "alert_window": {"start": _ISO_START, "end": _ISO_END},
        "prompt_text": "p",
    }
    _write_case(cases_root, "t-min", task=minimal)
    t = load_task("t-min", cases_root)
    assert t.task_id == "t-min"
    assert t.task_version is None
    assert t.alert_event_id is None
    assert t.workspace is None
    assert t.region_id is None
    assert t.scoring_note is None
    assert t.alert_entity == {}
    assert t.available_modalities == []


def test_load_topology_optional_fields_default_when_absent(cases_root: Path):
    """topology.json with optional keys omitted still loads with defaults."""
    minimal = {"case_id": "t-min-topo", "window": {"start_iso": _ISO_START}}
    _write_case(cases_root, "t-min-topo", topology=minimal)
    topo = load_topology("t-min-topo", cases_root)
    assert topo.case_id == "t-min-topo"
    assert topo.source is None
    assert topo.cluster_id is None
    assert topo.entities == []
    assert topo.edges == []
    assert topo.stats == {}
    # end defaults to start when missing.
    assert topo.window.end == topo.window.start


def test_window_z_suffix_parsed(cases_root: Path):
    """A trailing 'Z' (UTC Zulu) must parse, not crash — the loader swaps Z for
    +00:00 before fromisoformat."""
    task = _task_dict("t-z")
    task["alert_window"] = {"start": "2026-04-25T05:18:12Z", "end": "2026-04-25T05:28:12Z"}
    _write_case(cases_root, "t-z", task=task)
    t = load_task("t-z", cases_root)
    assert t.alert_window.start.tzinfo is not None
    # Zulu suffix must parse to a zero UTC offset (not naive, not a wrong zone).
    assert t.alert_window.start.utcoffset() == timedelta(0)
