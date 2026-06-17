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
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    files = sorted(runs_dir.glob("*.report.json"))
    if not files:
        print(f"no *.report.json in {runs_dir}")
        return
    elapsed_map = _load_elapsed(runs_dir)

    rows = []
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


if __name__ == "__main__":
    main()
