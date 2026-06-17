"""Unit tests for ``scripts/analyze_eval.py`` (I1: baseline-diff mode).

Constructs tiny tmp runs-dirs + baseline JSONs and exercises:
  * ``--baseline`` prints a Δ table and exits 1 on a token regression.
  * ``--baseline`` with no regression exits 0 and still prints the Δ table.
  * Omitting ``--baseline`` keeps the original analyzer output unchanged.
  * Robustness: missing baseline file / missing aggregate keys degrade
    gracefully instead of crashing.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Load scripts/analyze_eval.py as a module (it's not on the package path).
# --------------------------------------------------------------------------- #
def _load_analyze(tmp_path: Path):
    """Import scripts/analyze_eval.py by path so tests can call its helpers."""
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "scripts" / "analyze_eval.py"
    assert src.exists(), f"analyzer not found at {src}"
    spec = importlib.util.spec_from_file_location("analyze_eval", src)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Insert into sys.modules so any internal imports resolve.
    sys.modules["analyze_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def _report(
    case_id: str,
    *,
    status: str = "completed",
    confidence: float = 0.85,
    n_steps: int = 10,
    n_tools: int = 5,
    total_tokens: int = 1000,
    reasoning_tokens: int = 0,
    n_entities: int = 2,
    n_evidence: int = 3,
) -> dict:
    """Build a minimal report.json shaped like rca_agent.eval.runner output."""
    return {
        "case_id": case_id,
        "status": status,
        "root_cause": {
            "confidence": confidence if status == "completed" else 0.0,
            "fault_type": "dependency.timeout" if status == "completed" else None,
            "entity_refs": [{"entity_name": f"e{i}"} for i in range(n_entities)],
            "evidence": [f"ev{i}" for i in range(n_evidence)],
        },
        "steps": [
            {"step_kind": "tool_call", "tool_name": "query_logs"}
            for _ in range(n_tools)
        ] + [{"step_kind": "reasoning"} for _ in range(n_steps - n_tools)],
        "token_usage": {
            "prompt_tokens": total_tokens // 2,
            "completion_tokens": total_tokens // 2,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
        },
    }


def _write_runs_dir(tmp_path: Path, reports: list[dict]) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    for r in reports:
        (runs / f"{r['case_id']}.report.json").write_text(json.dumps(r))
    return runs


def _baseline(aggregate: dict, metadata: dict | None = None) -> dict:
    return {
        "metadata": metadata or {"date": "2026-06-01", "git_sha": "abc1234", "model": "deepseek-reasoner"},
        "aggregate": aggregate,
        "per_case": [],
    }


# --------------------------------------------------------------------------- #
# --baseline regression detection
# --------------------------------------------------------------------------- #
def test_baseline_flags_token_regression_and_exits_1(tmp_path, capsys) -> None:
    """A >20% token rise vs baseline must print a Δ table, flag it with ⚠, and
    exit 1 (so the analyzer can gate CI)."""
    mod = _load_analyze(tmp_path)
    runs = _write_runs_dir(
        tmp_path,
        [_report("t001", total_tokens=10_000)],  # current: 10k
    )
    base = _baseline({
        "n_cases": 1, "n_completed": 1, "n_truncated": 0, "n_errors": 0,
        "avg_confidence": 0.85, "avg_steps": 10.0, "avg_tool_calls": 5.0,
        "avg_total_tokens": 1000.0,  # baseline: 1k -> 10x rise, well over 20%
        "avg_entities": 2.0, "avg_evidence": 3.0,
    })
    base_path = tmp_path / "baseline.json"
    base_path.write_text(json.dumps(base))

    with pytest.raises(SystemExit) as ei:
        mod.main(["--runs-dir", str(runs), "--baseline", str(base_path)])
    assert ei.value.code == 1

    out = capsys.readouterr().out
    assert "== baseline diff ==" in out
    assert "avg_total_tokens" in out
    assert "1000.0" in out and "10000.0" in out
    assert "⚠" in out
    assert "regression" in out.lower()


def test_baseline_no_regression_exits_0(tmp_path, capsys) -> None:
    """When current matches or beats baseline on every rule, exit 0 and report
    no regressions — but still print the Δ table."""
    mod = _load_analyze(tmp_path)
    runs = _write_runs_dir(
        tmp_path,
        [_report("t001", total_tokens=1000, confidence=0.90)],
    )
    base = _baseline({
        "n_cases": 1, "n_completed": 1, "n_truncated": 0, "n_errors": 0,
        "avg_confidence": 0.85, "avg_steps": 10.0, "avg_tool_calls": 5.0,
        "avg_total_tokens": 1000.0,
        "avg_entities": 2.0, "avg_evidence": 3.0,
    })
    base_path = tmp_path / "baseline.json"
    base_path.write_text(json.dumps(base))

    # Should NOT raise SystemExit(1).
    mod.main(["--runs-dir", str(runs), "--baseline", str(base_path)])

    out = capsys.readouterr().out
    assert "== baseline diff ==" in out
    assert "no regressions flagged" in out


def test_baseline_flags_convergence_drop(tmp_path, capsys) -> None:
    """A completed→truncated shift (fewer completions, more truncations) is a
    regression on convergence_rate and exits 1."""
    mod = _load_analyze(tmp_path)
    runs = _write_runs_dir(
        tmp_path,
        [
            _report("t001", status="completed"),
            _report("t002", status="truncated", confidence=0.0),
        ],
    )
    base = _baseline({
        "n_cases": 2, "n_completed": 2, "n_truncated": 0, "n_errors": 0,
        "avg_confidence": 0.85, "avg_steps": 10.0, "avg_tool_calls": 5.0,
        "avg_total_tokens": 1000.0,
        "avg_entities": 2.0, "avg_evidence": 3.0,
    })
    base_path = tmp_path / "baseline.json"
    base_path.write_text(json.dumps(base))

    with pytest.raises(SystemExit) as ei:
        mod.main(["--runs-dir", str(runs), "--baseline", str(base_path)])
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "convergence_rate" in out
    assert "⚠" in out


def test_baseline_flags_confidence_drop(tmp_path, capsys) -> None:
    """avg_confidence dropping by more than 0.05 is flagged."""
    mod = _load_analyze(tmp_path)
    runs = _write_runs_dir(
        tmp_path,
        [_report("t001", confidence=0.50)],  # baseline 0.90 -> drop of 0.40
    )
    base = _baseline({
        "n_cases": 1, "n_completed": 1, "n_truncated": 0, "n_errors": 0,
        "avg_confidence": 0.90, "avg_steps": 10.0, "avg_tool_calls": 5.0,
        "avg_total_tokens": 1000.0,
        "avg_entities": 2.0, "avg_evidence": 3.0,
    })
    base_path = tmp_path / "baseline.json"
    base_path.write_text(json.dumps(base))

    with pytest.raises(SystemExit) as ei:
        mod.main(["--runs-dir", str(runs), "--baseline", str(base_path)])
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "avg_confidence" in out
    assert "⚠" in out


# --------------------------------------------------------------------------- #
# No --baseline: unchanged behavior
# --------------------------------------------------------------------------- #
def test_no_baseline_unchanged_output(tmp_path, capsys) -> None:
    """Without --baseline the analyzer prints only the original per-case +
    aggregate report and exits normally (no Δ table)."""
    mod = _load_analyze(tmp_path)
    runs = _write_runs_dir(
        tmp_path,
        [_report("t001"), _report("t002", status="truncated", confidence=0.0)],
    )
    mod.main(["--runs-dir", str(runs)])
    out = capsys.readouterr().out
    assert "== baseline diff ==" not in out
    # Original output sections are still present.
    assert "== 2 cases ==" in out
    assert "== convergence ==" in out
    assert "== token efficiency ==" in out


def test_no_runs_dir_prints_message_and_returns(tmp_path, capsys) -> None:
    """An empty runs-dir prints a friendly message and exits 0 (no crash)."""
    mod = _load_analyze(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    mod.main(["--runs-dir", str(empty)])
    out = capsys.readouterr().out
    assert "no *.report.json" in out


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def test_baseline_missing_file_does_not_crash(tmp_path, capsys) -> None:
    """A nonexistent --baseline path degrades to a warning (no diff, exit 0)."""
    mod = _load_analyze(tmp_path)
    runs = _write_runs_dir(tmp_path, [_report("t001")])
    bogus = tmp_path / "does-not-exist.json"
    mod.main(["--runs-dir", str(runs), "--baseline", str(bogus)])
    out = capsys.readouterr()
    assert "== baseline diff ==" not in out.out  # no diff printed
    assert "could not load baseline" in out.err


def test_baseline_missing_keys_tolerated(tmp_path, capsys) -> None:
    """A baseline missing some aggregate keys must still diff (treat missing as
    0 / not-comparable) without raising."""
    mod = _load_analyze(tmp_path)
    runs = _write_runs_dir(tmp_path, [_report("t001", total_tokens=1000)])
    # Baseline has only n_cases + n_completed — everything else missing.
    base = _baseline({"n_cases": 1, "n_completed": 1})
    base_path = tmp_path / "baseline.json"
    base_path.write_text(json.dumps(base))

    mod.main(["--runs-dir", str(runs), "--baseline", str(base_path)])
    out = capsys.readouterr().out
    assert "== baseline diff ==" in out
    # With baseline total_tokens=0 and current=1000, the >20% rule doesn't
    # fire (guard: b_tok > 0). convergence matches. No regression expected.
    assert "no regressions flagged" in out


def test_diff_metrics_unit_smoke(tmp_path) -> None:
    """Direct unit test of _diff_metrics classification rules.

    Loads the analyzer fresh via _load_analyze (not importlib.import_module) so
    this test is order-independent and passes in isolation."""
    mod = _load_analyze(tmp_path)
    base_agg = {
        "n_cases": 10, "n_completed": 9, "n_truncated": 1, "n_errors": 0,
        "avg_confidence": 0.85, "avg_total_tokens": 1000.0,
    }
    cur_agg = {
        "n_cases": 10, "n_completed": 9, "n_truncated": 1, "n_errors": 0,
        "avg_confidence": 0.80, "avg_total_tokens": 1100.0,  # +10%, under 20%
    }
    diffs = mod._diff_metrics(cur_agg, base_agg)
    by_name = {d.name: d for d in diffs}
    # +10% tokens is under the 20% threshold -> not a regression.
    assert not by_name["avg_total_tokens"].regressed
    # Confidence dropped 0.05 exactly — rule is "< -0.05" so 0.05 is NOT a reg.
    assert not by_name["avg_confidence"].regressed
    # Push confidence to -0.06 -> regression.
    diffs2 = mod._diff_metrics(
        {**cur_agg, "avg_confidence": 0.79}, base_agg,
    )
    assert any(d.regressed for d in diffs2 if d.name == "avg_confidence")
