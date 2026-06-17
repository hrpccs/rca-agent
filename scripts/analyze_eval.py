#!/usr/bin/env python
"""Analyze persisted RCA eval reports into aggregate capability statistics.

The eval runner (``rca_agent.eval.runner``) writes raw per-case
``<cid>.report.json`` files + an ``eval_summary.{json,csv}`` but ships no
analysis layer. This script reads those artifacts and prints a per-case table,
convergence rate, token/step/tool distributions (mean/p50/p90/max), tool-usage
patterns, token efficiency, and flags the thinking-cost accounting gap.

Re-run it any time the case set grows::

    uv run python scripts/analyze_eval.py [--runs-dir runs]

It deliberately computes everything from the persisted artifacts (no live
calls), so it doubles as a regression-comparison input source: snapshot the
output (or the ``eval_summary.json``) per milestone and diff.

Baseline-diff mode
------------------
With ``--baseline <path>`` it loads a committed baseline JSON (the
``eval_baselines/*.json`` shape: ``metadata`` + ``aggregate`` + optional
``per_case``) and prints a Δ table comparing current vs baseline for the key
metrics, flagging regressions (convergence drop, confidence drop >0.05, token
rise >20%, completed→error/truncated shift). Exits non-zero if any regression
is flagged, so it can gate CI.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _agg(xs: list) -> dict:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 1),
        "p50": round(_pct(xs, 50), 1),
        "p90": round(_pct(xs, 90), 1),
        "max": round(max(xs), 1),
    }


def _load_elapsed(runs_dir: Path) -> dict[str, float]:
    """elapsed_s isn't on RcaReport; read it from eval_summary.json if present."""
    summ = runs_dir / "eval_summary.json"
    out: dict[str, float] = {}
    if summ.exists():
        try:
            for r in json.loads(summ.read_text()).get("results", []):
                if r.get("case_id") and r.get("elapsed_s") is not None:
                    out[r["case_id"]] = r["elapsed_s"]
        except (json.JSONDecodeError, OSError):
            pass
    return out


def _load_rows(runs_dir: Path) -> list[dict[str, Any]]:
    """Read every ``*.report.json`` under ``runs_dir`` into a flat row dict."""
    files = sorted(runs_dir.glob("*.report.json"))
    elapsed_map = _load_elapsed(runs_dir)
    rows: list[dict[str, Any]] = []
    for fp in files:
        r = json.loads(fp.read_text())
        rc = r.get("root_cause", {}) or {}
        steps = r.get("steps", []) or []
        tools = Counter(s.get("tool_name") for s in steps if s.get("step_kind") == "tool_call")
        tu = r.get("token_usage", {}) or {}
        cid = r.get("case_id", fp.stem)
        rows.append({
            "case": cid,
            "status": r.get("status", "?"),
            "conf": rc.get("confidence"),
            "fault_type": rc.get("fault_type"),
            "n_ent": len(rc.get("entity_refs", [])),
            "n_ev": len(rc.get("evidence", [])),
            "n_steps": len(steps),
            "n_tools": sum(tools.values()),
            "tools": dict(tools),
            "prompt_tok": tu.get("prompt_tokens", 0),
            "total_tok": tu.get("total_tokens", 0),
            "reasoning_tok": tu.get("reasoning_tokens", 0),
            "elapsed": elapsed_map.get(cid),
        })
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the aggregate stats block (same shape as eval_baselines/*.json).

    Robust to missing/None fields: every metric falls back to 0 when no cases
    contribute a numeric value.
    """
    n_cases = len(rows)
    if not n_cases:
        return {
            "n_cases": 0,
            "n_completed": 0,
            "n_truncated": 0,
            "n_errors": 0,
            "avg_confidence": 0.0,
            "avg_steps": 0.0,
            "avg_tool_calls": 0.0,
            "avg_total_tokens": 0.0,
            "avg_entities": 0.0,
            "avg_evidence": 0.0,
        }
    status_counts = Counter(x["status"] for x in rows)
    n_completed = status_counts.get("completed", 0)
    n_truncated = status_counts.get("truncated", 0)
    n_errors = status_counts.get("error", 0) + status_counts.get("failed", 0)

    # Confidence is averaged over completed cases only (truncated reports a 0.0
    # placeholder that would drag the mean down and isn't a real signal).
    confs = [
        x["conf"] for x in rows
        if x["conf"] is not None and x["status"] == "completed"
    ]
    steps_all = [x["n_steps"] for x in rows]
    tools_all = [x["n_tools"] for x in rows]
    tok_all = [x["total_tok"] for x in rows]
    ent_all = [x["n_ent"] for x in rows]
    ev_all = [x["n_ev"] for x in rows]

    def _mean(xs: list) -> float:
        xs = [float(x) for x in xs if x is not None]
        return round(statistics.mean(xs), 2) if xs else 0.0

    return {
        "n_cases": n_cases,
        "n_completed": n_completed,
        "n_truncated": n_truncated,
        "n_errors": n_errors,
        "avg_confidence": _mean(confs),
        "avg_steps": _mean(steps_all),
        "avg_tool_calls": _mean(tools_all),
        "avg_total_tokens": _mean(tok_all),
        "avg_entities": _mean(ent_all),
        "avg_evidence": _mean(ev_all),
    }


# --------------------------------------------------------------------------- #
# Baseline diff
# --------------------------------------------------------------------------- #
@dataclass
class _MetricDiff:
    name: str
    baseline: float | str
    current: float | str
    delta: float | None  # None when not numeric / not comparable
    regressed: bool = False
    note: str = ""


def _num(v: Any) -> float | None:
    """Coerce a baseline/current metric value to float; None if not comparable."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_get(d: dict[str, Any] | None, key: str, default: Any = 0) -> Any:
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    return default if v is None else v


def _convergence_rate(agg: dict[str, Any]) -> float:
    """Fraction of cases that reached a conclusion (completed / n_cases)."""
    n = int(_safe_get(agg, "n_cases", 0) or 0)
    if n <= 0:
        return 0.0
    completed = int(_safe_get(agg, "n_completed", 0) or 0)
    return completed / n


def _diff_metrics(
    cur_agg: dict[str, Any],
    base_agg: dict[str, Any],
) -> list[_MetricDiff]:
    """Build the per-metric diff list and classify regressions.

    Regression rules (a metric is flagged ⚠ when ANY holds):
      * convergence rate drops at all (cur < base)
      * avg_confidence drops by more than 0.05
      * avg_total_tokens rises by more than 20%
      * a completed→error or completed→truncated shift (fewer completions while
        errors or truncations grew)
    Non-regressing numeric deltas are reported with their Δ for visibility.
    """
    cur_conv = _convergence_rate(cur_agg)
    base_conv = _convergence_rate(base_agg)

    metric_pairs: list[tuple[str, str]] = [
        ("convergence_rate", "__computed__"),
        ("avg_confidence", "avg_confidence"),
        ("avg_steps", "avg_steps"),
        ("avg_tool_calls", "avg_tool_calls"),
        ("avg_total_tokens", "avg_total_tokens"),
        ("avg_entities", "avg_entities"),
        ("avg_evidence", "avg_evidence"),
    ]

    out: list[_MetricDiff] = []
    for display, key in metric_pairs:
        if key == "__computed__":
            b_val: Any = round(base_conv, 4)
            c_val: Any = round(cur_conv, 4)
        else:
            b_val = _safe_get(base_agg, key, 0)
            c_val = _safe_get(cur_agg, key, 0)
        bn, cn = _num(b_val), _num(c_val)
        delta = (cn - bn) if (bn is not None and cn is not None) else None
        out.append(_MetricDiff(name=display, baseline=b_val, current=c_val, delta=delta))

    # Classify regressions.
    # Convergence rate drop.
    if cur_conv < base_conv:
        out[0].regressed = True
        out[0].note = "convergence dropped"
    # Confidence drop > 0.05.
    if out[1].delta is not None and out[1].delta < -0.05:
        out[1].regressed = True
        out[1].note = "confidence dropped >0.05"
    # Token rise > 20% (only meaningful when baseline > 0).
    b_tok = _num(_safe_get(base_agg, "avg_total_tokens", 0))
    c_tok = _num(_safe_get(cur_agg, "avg_total_tokens", 0))
    if (
        b_tok is not None
        and c_tok is not None
        and b_tok > 0
        and c_tok > b_tok * 1.20
    ):
        out[4].regressed = True
        out[4].note = "tokens rose >20%"
    # completed→error/truncated shift.
    b_completed = int(_safe_get(base_agg, "n_completed", 0) or 0)
    c_completed = int(_safe_get(cur_agg, "n_completed", 0) or 0)
    b_err = int(_safe_get(base_agg, "n_errors", 0) or 0)
    c_err = int(_safe_get(cur_agg, "n_errors", 0) or 0)
    b_trunc = int(_safe_get(base_agg, "n_truncated", 0) or 0)
    c_trunc = int(_safe_get(cur_agg, "n_truncated", 0) or 0)
    if c_completed < b_completed and (c_err > b_err or c_trunc > b_trunc):
        out[0].regressed = True
        if "completed→error/truncated shift" not in out[0].note:
            out[0].note = (out[0].note + "; " if out[0].note else "") + \
                "completed→error/truncated shift"
    return out


def _load_baseline(path: Path) -> dict[str, Any] | None:
    """Load a baseline JSON file; return None + warn on any failure.

    Robust to missing files and malformed JSON so a misconfigured ``--baseline``
    path degrades to "no baseline" rather than crashing the analyzer.
    """
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: could not load baseline {path}: {e}", file=sys.stderr)
        return None
    if not isinstance(raw, dict):
        print(f"warning: baseline {path} is not a JSON object", file=sys.stderr)
        return None
    return raw


def _print_diff(
    cur_agg: dict[str, Any],
    baseline: dict[str, Any],
    out=sys.stdout,
) -> bool:
    """Print the baseline-vs-current Δ table. Return True if any regression.

    Missing keys in either file are tolerated (treated as 0 / not-comparable).
    """
    base_meta = baseline.get("metadata") if isinstance(baseline.get("metadata"), dict) else {}
    base_agg = baseline.get("aggregate")
    if not isinstance(base_agg, dict):
        print(
            "warning: baseline has no 'aggregate' block; cannot diff",
            file=sys.stderr,
        )
        return False

    print("", file=out)
    print("== baseline diff ==", file=out)
    if base_meta:
        bdate = base_meta.get("date", "?")
        bsha = base_meta.get("git_sha", "?")
        bmodel = base_meta.get("model", "?")
        print(
            f"  baseline: {bdate} sha={bsha} model={bmodel}",
            file=out,
        )

    diffs = _diff_metrics(cur_agg, base_agg)
    any_regression = False
    print(
        f"  {'metric':22} {'baseline':>14} {'current':>14} {'Δ':>14}  flag",
        file=out,
    )
    for d in diffs:
        flag = "⚠" if d.regressed else ""
        if flag:
            any_regression = True
        delta_str = f"{d.delta:+.4g}" if d.delta is not None else "n/a"
        note = f"  {d.note}" if d.note else ""
        print(
            f"  {d.name:22} {str(d.baseline):>14} {str(d.current):>14} "
            f"{delta_str:>14}  {flag}{note}",
            file=out,
        )
    if any_regression:
        print("\n  ⚠ regression(s) detected — see flagged rows above.", file=out)
    else:
        print("\n  no regressions flagged.", file=out)
    return any_regression


# --------------------------------------------------------------------------- #
# Report printing
# --------------------------------------------------------------------------- #
def _print_report(rows: list[dict[str, Any]]) -> None:
    """Print the full per-case + aggregate report (the original analyzer output).

    Kept identical to the pre-``--baseline`` behavior so omitting ``--baseline``
    is a strict no-op.
    """
    print(f"== {len(rows)} cases ==")
    print(f"{'case':6} {'status':10} {'conf':>5} {'steps':>5} {'tools':>5} "
          f"{'total_tok':>10} {'elapsed':>7}  fault_type")
    for x in rows:
        print(f"{x['case']:6} {x['status']:10} {str(x['conf']):>5} {x['n_steps']:>5} "
              f"{x['n_tools']:>5} {x['total_tok']:>10} {str(x['elapsed']):>7}  {x['fault_type']}")

    st = Counter(x["status"] for x in rows)
    print("\n== convergence ==")
    for k, v in st.most_common():
        print(f"  {k}: {v} ({round(100 * v / len(rows))}%)")

    done = [x for x in rows if x["status"] in ("completed", "truncated")]
    print(f"\n== distributions (n={len(done)} completed/truncated) ==")
    for name, key in [("confidence", "conf"), ("total_tokens", "total_tok"),
                      ("prompt_tokens", "prompt_tok"), ("steps", "n_steps"),
                      ("tool_calls", "n_tools"), ("elapsed_s", "elapsed")]:
        a = _agg([x[key] for x in done])
        print(f"  {name:14} mean={a['mean']:>10}  p50={a['p50']:>10}  "
              f"p90={a['p90']:>10}  max={a['max']:>10}")

    tc = Counter()
    for x in rows:
        for t, c in x["tools"].items():
            tc[t] += c
    print("\n== tool usage (total calls across cases) ==")
    for t, c in tc.most_common():
        print(f"  {t:18} {c:>4}  ({round(c / max(1, len(done)), 1)}/case)")

    tot_tok = sum(x["total_tok"] for x in done)
    tot_tools = sum(x["n_tools"] for x in done)
    print("\n== token efficiency ==")
    print(f"  total tokens (completed): {tot_tok:,}")
    print(f"  total tool calls:         {tot_tools}")
    if tot_tools:
        print(f"  tokens / tool call:       {round(tot_tok / tot_tools):,}")
    zreason = sum(1 for x in rows if not x["reasoning_tok"])
    print(f"  cases w/ reasoning_tokens=0: {zreason}/{len(rows)}  "
          "(thinking-cost accounting gap)")

    confs = [x["conf"] for x in done if x["conf"] is not None and x["status"] != "truncated"]
    if confs:
        print("\n== confidence (NOTE: uncalibrated — no ground truth to check accuracy) ==")
        print(f"  range: {min(confs)}-{max(confs)}  mean={round(statistics.mean(confs), 2)}")
        hi = sum(1 for c in confs if c >= 0.8)
        print(f"  >=0.8 (high): {hi}/{len(confs)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Path to a baseline JSON (eval_baselines/*.json shape) to diff "
            "against. When given, prints a Δ table and exits 1 on regression."
        ),
    )
    args = ap.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    files = sorted(runs_dir.glob("*.report.json"))
    if not files:
        print(f"no *.report.json in {runs_dir}")
        return

    rows = _load_rows(runs_dir)
    _print_report(rows)

    if args.baseline is not None:
        baseline = _load_baseline(args.baseline)
        if baseline is not None:
            cur_agg = _aggregate(rows)
            regressed = _print_diff(cur_agg, baseline)
            if regressed:
                sys.exit(1)


if __name__ == "__main__":
    main()
