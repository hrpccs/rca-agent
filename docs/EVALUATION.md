# RCA Agent — Capability Evaluation & Evaluation-Infrastructure Assessment

*Baseline run: 2026-06-18 · model `deepseek-reasoner` (reasoning_effort=high) ·
parquet backend · max_steps=25, max_tokens=8192 · 8-case sample
(t001, t002, t004, t020, t040, t060, t080, t100). Raw artifacts in `runs/`;
regression baseline in `eval_baselines/baseline-2026-06-18.json`; analysis tool
`scripts/analyze_eval.py`.*

---

## 1. Executive summary

We measured the agent's **operational/behavioral** RCA capability on an 8-case
sample of the rca100 benchmark. Headline result: the agent **converges with a
confident, well-evidenced root cause in most cases** (7/8 completed, mean
confidence **0.83**, mean **5.6 evidence pointers** and **3.3 entities** per
conclusion), but it is **expensive** (mean **562K tokens/case**, p90 **839K**,
max **1.13M**) and occasionally **fails to converge** (1/8 truncated — 768K
tokens spent with no conclusion).

**The single most important limitation is not in the agent — it is in the
benchmark: rca100 ships no ground truth.** Each case's `task.json` carries only
the alert and topology; `scoring_note` reads *"Output contract and fault
taxonomy will be published in a follow-up release."* Consequently **RCA
*correctness* cannot be scored today** — only convergence, confidence, cost,
and structural richness. `rca_agent/eval/scoring.py` already implements
entity-set P/R/F1 and fault-type match, but there is nothing to match against.
**Publishing the answer key and wiring it into scoring is the #1 infrastructure
priority** (see §4). Until then, every capability number below is a *behavioral*
proxy, not an *accuracy* measure.

---

## 2. Methodology

- **Harness:** `rca eval --cases … --backend parquet` → `rca_agent.eval.runner.run_eval`,
  which drives `build_agent_for_case` per case, drains the agent's step/report
  stream, and writes `runs/<cid>.report.json` + `runs/eval_summary.{json,csv}`.
- **Sample:** 8 cases spread across the t001–t103 range (not random; chosen for
  coverage). Not statistically representative of all 103 — see §5.
- **Cost ceiling:** at DeepSeek-reasoner pricing the 8 cases cost on the order
  of a few USD; a full 103-case sweep is therefore ~$10–20 plus several hours,
  gateway-throttled (≤~3 concurrent streams). Continuous accrual (§5) is the
  intended path, not a one-shot full sweep.
- **What is objectively measurable (no GT):** convergence status, confidence,
  step/tool counts, token cost, latency, fault-type *coverage*, entity/evidence
  *richness*, tool-usage mix, failure modes.
- **What is NOT measurable (needs GT):** root-cause correctness, entity P/R/F1,
  fault-type accuracy, confidence calibration.

---

## 3. Capability findings (n=8)

### 3.1 Convergence — strong, with one truncation failure

| status | count | share |
|---|---|---|
| completed | 7 | 88% |
| truncated | 1 | 12% |
| error | 0 | 0% |

The one truncation (**t080**) is the clearest failure mode: the agent ran **26
reasoning turns / 35 tool calls / 96 steps / 768K tokens** and was still
mid-investigation (last steps were `tool_call`/`tool_result`) when it hit the
25-turn step cap — it emitted **no conclusion and confidence 0.0**. That is a
full-cost run yielding no answer. *Recommendation: a "force best-hypothesis"
fallback near the step cap so truncated runs still return a low-confidence
answer (§4.10).*

### 3.2 Confidence — high but uncalibrated

Completed cases report confidence **0.75–0.85 (mean 0.83)**; 5/7 ≥ 0.8. With no
ground truth we **cannot** say whether 0.83 confidence corresponds to ~83%
correct — it may be over- or under-confident. Note the agent rarely produces
low-confidence completed answers, so the scale is compressed (no signal below
0.75).

### 3.3 Token cost — the dominant constraint; context grows unboundedly

| metric | mean | p50 | p90 | max |
|---|---|---|---|---|
| total tokens | 588K | 551K | 839K | **1.13M** (t060) |
| prompt tokens | 580K | 546K | 831K | 1.12M |
| steps | 92 | 96 | 109 | 135 |
| tool calls | 37 | 38 | 45 | 54 |

**~16,050 tokens are added to the context per tool call** (each tool result is
appended and re-sent every subsequent turn). This is the cost driver *and* a
capability ceiling: longer investigations compound exponentially in prompt size,
which is exactly what produces 1.13M-token runs and the truncation in §3.1.

**Accounting gap:** `reasoning_tokens` is **0 in 8/8 cases** despite
thinking-mode. The client dumps the raw usage chunk, but the agent's accumulator
reads a top-level `reasoning_tokens` key that DeepSeek does not populate (it
nests thinking tokens under `completion_tokens_details`). So **true cost is
undercounted** — the thinking budget is invisible.

### 3.4 Latency — acceptable per case, blocking at scale

72–198s/case (mean ~130s). Sequential eval → ~15 min for 6 cases; a full sweep
is multi-hour. Parallelism is gateway-limited (§4.8).

### 3.5 Tool-usage mix — leans on metrics/logs/traces

| tool | calls/case |
|---|---|
| query_metrics | 8.8 |
| query_logs | 8.1 |
| query_traces | 7.5 |
| inspect_entity | 3.6 |
| store_observation | 3.4 |
| query_events | 2.6 |
| query_alerts | 1.5 |
| get_topology | 1.1 |

The agent spends heavily on metrics/logs/traces (the three richest modalities)
and lightly on topology/alerts. Whether this is *optimal* per case is unknown
without correctness signal, but the volume (≈24 metrics+logs+traces calls/case)
plus §3.3 suggests possible **over-investigation** before concluding.

### 3.6 Fault-type attribution — clusters on `dependency.timeout`

| fault_type | count |
|---|---|
| dependency.timeout | 5/8 |
| app.exception | 1 |
| infra.database_slow_query | 1 |
| (truncated) | 1 |

5/8 conclusions are typed `dependency.timeout`. This **could** reflect the true
distribution *or* an over-attribution bias — indistinguishable without ground
truth. This clustering is itself an argument for per-stratum reporting (§4.9).

### 3.7 Trace quality (now durable via Wave-4)

With Wave-4, every step (reasoning, tool_call + args, tool_result + rendered
text, memory-retrieval, conclude) is persisted (`runs/<cid>.report.json` →
`root_cause.steps`, and via the server to the `rca_steps` table). Qualitatively,
the traces read as genuine ReAct investigations: hypothesis → targeted query →
evidence → refined hypothesis. The evidence pointers in conclusions cite real
step content. *This persistence is what makes the continuous, trace-driven
evaluation in §5 possible.*

---

## 4. Evaluation-infrastructure assessment — prioritized recommendations

Ordered by leverage on "ability to continuously, quantitatively evaluate and
improve the agent."

**P0 — publish ground truth + wire correctness scoring.** The benchmark is
blind; nothing else matters until RCA correctness is measurable. `scoring.py`
already has `entity_precision/recall/f1` + `fault_type_match`. Needed: the
rca100 answer key (root-cause entity set + fault type per case) exposed under
each case dir, and the runner computing P/R/F1 + fault-type accuracy into
`eval_summary`. This converts every number in §3 from a behavioral proxy into an
accuracy measure and unlocks calibration (§4.6).

**P1 — fix token accounting.** Capture `completion_tokens_details.reasoning_tokens`
(DepSeek nests it). Every cost figure currently undercounts the thinking budget.
Small, high-value fix in `llm/deepseek_client.py` usage extraction.

**P1 — bound context growth.** ~16K tokens/tool call, up to 1.13M/case. Options:
summarize/compact older tool results past a window, cap tool-result text size
in the LLM context (the persisted trace can keep the full text), or a
sliding-window context manager. Directly lowers cost *and* the truncation rate.

**P1 — regression baseline + diff harness.** A baseline snapshot is now committed
(`eval_baselines/baseline-2026-06-18.json`) and `scripts/analyze_eval.py`
recomputes the same stats from artifacts. Add a `--baseline <file>` diff mode
that flags material regressions (convergence ↓, cost ↑, confidence drift) on
every change — the foundation for "continuously improve without regressing."

**P2 — per-module cost/latency in the eval output.** OTel already records
`rca_provider_query_duration_seconds` and per-tool counts, but `eval_summary`
carries only aggregate counts/tokens. Surface per-tool and per-modality latency
+ token share so bottlenecks (e.g. query_traces volume) are visible per run.

**P2 — confidence calibration.** Once GT exists: reliability diagram
(confidence bucket vs empirical accuracy), and optionally a calibration map.
Today confidence is a free-floating 0.75–0.85.

**P2 — queryable metrics store.** Eval writes files; OTel is ephemeral; the new
`rca_steps`/`rca_runs` trace store is not used by the eval path. Persist the
per-case metrics into `rca_runs` (it already has status/model/token_usage
columns) so historical runs are queryable, not just the latest files. (Wave-4's
trace API already provides the read side — `GET /runs`.)

**P2 — parallel eval.** Runner is sequential. A gateway-safe worker pool (≤3
concurrent) with per-worker `out_dir` (then merge) would ~3× throughput. Also
expose `--out-dir` (the CLI currently lacks it), `--concurrency`, and
stratified `--sample`.

**P2 — difficulty/type stratification.** Tag cases by fault type / difficulty;
report per-stratum. The §3.6 clustering shows aggregate numbers can hide
per-type behavior.

**P2 — convergence / stopping policy.** Replace the hard truncation (t080: 768K
tokens, no answer) with a "force best-hypothesis" fallback near the step cap so
truncated runs still return a low-confidence root cause instead of nothing.

**P3 — eval CLI ergonomics.** Add `--out-dir`, `--concurrency`, `--sample`,
`--baseline`; emit the analysis table (`scripts/analyze_eval.py`) at run end.

---

## 5. Baseline & continuous evaluation

- **Baseline:** `eval_baselines/baseline-2026-06-18.json` — aggregate +
  per-case metrics + run metadata (model, settings, git sha). Diff future runs
  against it.
- **Analysis:** `scripts/analyze_eval.py` — re-run after any eval batch to
  recompute convergence, distributions, tool mix, token efficiency from
  `runs/*.report.json` (no live calls).
- **Continuous accrual:** run additional paced batches (≤3 concurrent) to grow
  the sample toward the full 103; each batch refreshes `runs/` and the analysis.
  Re-snapshot a baseline at each meaningful milestone (model/prompt change).
- **Per-case traces:** `runs/<cid>.report.json` (full step trace) and, via the
  server, `GET /runs/{run_id}` / `cli trace <run_id>` (Wave-4) — the substrate
  for trace-level failure analysis (e.g. *why* did t080 not converge?).

### Reproduce

```bash
# clear the dev proxy first (breaks httpx) — cli.py also does this at import
unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
set -a; . .env; set +a; unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
uv run python -m rca_agent.cli eval --cases t001,t002,t004,t020,t040,t060,t080,t100 --backend parquet
uv run python scripts/analyze_eval.py --runs-dir runs
```

> **Note:** `t001`'s data has since moved under `cases/t001/t001_backup/`, so
> `load_case("t001")` now raises `FileNotFoundError`. `list_cases()` returns the
> **102** cases that still have a top-level `task.json` (t002–t103); use those
> for accrual.

---

## 6. Post-improvement validation (I1–I4)

Four improvements were implemented from the §4 recommendations and re-measured
on a 7-case re-run of the same set (`runs_post/`; `t001` skipped per the note
above). `eval_baselines/baseline-2026-06-18-post.json` is the post-improvement
snapshot; the pre-improvement `baseline-2026-06-18.json` is retained for the
historical comparison.

**Validated wins:**

- **I1 — token accounting (FIXED, confirmed).** `reasoning_tokens` is now
  captured (avg **3,698/case**) vs **0 in 8/8** pre-improvement. The
  thinking-cost budget is no longer invisible. (`0/7` cases now report zero.)
- **I2 — force-conclude fallback (WORKS, confirmed on the failure case).** The
  pre-improvement `t080` truncated to **confidence 0.0 / no answer** after 768K
  tokens. Post-improvement `t080` still reaches the step cap (`status=truncated`)
  but **recovers a real root cause at confidence 0.55** ("inventory endpoint
  traffic dropped 53%, self-recovered; likely upstream cart reduced calls —
  transient"). Exactly the intended rescue: a full-cost run no longer returns
  nothing.
- **I3 — per-module cost/latency + eval flags (used here).** The `--out-dir`
  flag produced this very `runs_post/` comparison without disturbing `runs/`;
  `eval_summary` now carries per-tool latency + modality breakdown +
  `tool_call_p90`.
- **I4 — context bounding (opt-in, default OFF).** Available via
  `RCA_CONTEXT_TOOL_RESULT_MAX_CHARS` / `RCA_CONTEXT_MAX_TOOL_MESSAGES`; not
  exercised live yet (default-off preserves behavior; unit-tested).

**Caveat — single-seed noise (a new infra finding).** The aggregate before/after
is **not** a clean signal: `t060` swung 0.78→0.30 (confidence) on a single
re-run, and the baseline-diff tool flagged `convergence_rate`/`avg_confidence`
"regressions" that are **DeepSeek non-determinism + the missing t001**, not
real effects. I2 only touches the truncated path and I4 is off by default, so
they cannot explain a completed-case confidence drop. **Implication: the eval
must run multiple seeds per case and report mean±std** before any
improvement-attribution or regression-gating is trustworthy. This raises the
priority of a **multi-seed / repeated-runs** mode (add to §4 as P1): `--seeds N`
in the runner, accrue N runs/case, and have the analyzer report per-case
variance + a noise-aware diff.

**Continuous evaluation (E4):** a paced accrual loop runs ~2 new cases per
~30 min against the 102 valid cases, appending to `runs/` and re-running the
analyzer + post-improvement baseline diff. It auto-expires after 7 days and
stops accruing past a coverage target (~50 cases) to bound cost.
