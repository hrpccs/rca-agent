import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchCases,
  fetchRun,
  fetchRuns,
  openRcaStream,
  startRca,
  type RcaStreamHandle,
  type StreamHandlers,
} from "./api";
import type { Backend, RcaReport, RcaStep, RunSummary } from "./types";
import { CasePicker } from "./components/CasePicker";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ReportCard } from "./components/ReportCard";
import { RunHistory } from "./components/RunHistory";
import { RunPanel } from "./components/RunPanel";
import { TraceTimeline } from "./components/TraceTimeline";
import "./App.css";

type RunStatus = "idle" | "starting" | "running" | "done" | "error";

/**
 * The trace state for a single case. Replacing the old single `steps`/`report`
 * pair with a per-case cache is what lets switching cases preserve the previous
 * case's trace instead of wiping it (the original `resetRun()` bug): each case
 * owns its own entry, and selecting a case just swaps which entry is rendered.
 */
interface CaseTrace {
  steps: RcaStep[];
  report: RcaReport | null;
  runId: string | null;
  status: RunStatus;
  error: string | null;
  elapsedMs: number;
  /**
   * Set when the trace being shown is a *persisted* run replayed from the
   * backend (via RunHistory click or a disconnect recovery), NOT the live run.
   * Distinguishes the two so the UI can show a "replay" badge / "back to live"
   * affordance. When null, the displayed trace is the live (or last live) run.
   */
  replayingRunId: string | null;
  /**
   * Banner shown above the timeline for best-effort disconnect recovery
   * ("connection lost — loaded server-saved trace (N steps) · retry" when a
   * persisted trace was fetched, or "connection lost — partial trace shown"
   * when no runId/fetchRun was available). Null when no such banner applies.
   * Distinct from `error` (which is the hard-failure line).
   */
  disconnectBanner: string | null;
  /**
   * Shadow copy of the LIVE run's trace, captured when a replay overlay is
   * applied so "Back to live" can restore it. While `replayingRunId != null`,
   * `steps`/`report`/`status` hold the REPLAYED (persisted) run; these fields
   * hold the live run so it isn't destroyed by viewing history. Null when no
   * replay is active (no shadow needed). Live SSE steps arriving during a
   * replay are appended to `liveSteps` (not `steps`) so they survive.
   */
  liveSteps: RcaStep[] | null;
  liveReport: RcaReport | null;
  liveStatus: RunStatus;
}

function emptyTrace(): CaseTrace {
  return {
    steps: [],
    report: null,
    runId: null,
    status: "idle",
    error: null,
    elapsedMs: 0,
    replayingRunId: null,
    disconnectBanner: null,
    liveSteps: null,
    liveReport: null,
    liveStatus: "idle",
  };
}

export default function App() {
  const [cases, setCases] = useState<string[]>([]);
  const [casesError, setCasesError] = useState<string | null>(null);
  const [selectedCase, setSelectedCase] = useState<string | null>(null);
  const [backend, setBackend] = useState<Backend>("parquet");

  // Per-case trace cache. A ref holds the mutable map; a version counter
  // forces a re-render whenever an entry mutates (React does not observe ref
  // mutations). This keeps the displayed trace (`cache.get(selectedCase)`) in
  // sync with stream callbacks that append steps, without re-creating the
  // stream handle on every step. The version is also a dependency of the
  // `current` memo so the recomputed trace is read after every bump.
  const cacheRef = useRef<Map<string, CaseTrace>>(new Map());
  const [version, setVersion] = useState(0);
  // Bump the render version. Called after every ref mutation that should be
  // reflected in the UI (step append, status change, banner set, etc.).
  const bump = useCallback(() => setVersion((v) => (v + 1) & 0x7fffffff), []);

  // Per-case persisted-run summaries for the RunHistory panel. Keyed by case
  // id so switching cases shows that case's runs (re-fetched on select).
  const runsByCaseRef = useRef<Map<string, RunSummary[]>>(new Map());
  const [runsLoading, setRunsLoading] = useState(false);

  // The run currently being replayed (for highlight in RunHistory). Lives in
  // the CaseTrace too (replayingRunId), but kept as top-level state for a
  // stable identity across renders.
  const [replayRunId, setReplayRunId] = useState<string | null>(null);

  const eventSourceRef = useRef<RcaStreamHandle | null>(null);
  const tickerRef = useRef<number | null>(null);
  // Per-run start timestamps, keyed by case id. The live elapsed ticker reads
  // the start time for the case it is ticking (not a single shared ref), so
  // switching cases mid-run no longer produces a non-monotonic elapsed jump:
  // each case's elapsed is measured from its own start.
  const startedAtByCaseRef = useRef<Map<string, number>>(new Map());
  // Mounted guard: in-flight fetchRun promises (disconnect recovery, replay)
  // check this before mutating the cache, so an unmount can't trigger a
  // React "state update on unmounted component" warning or a stale write.
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Load case list on mount.
  useEffect(() => {
    let cancelled = false;
    setCasesError(null);
    fetchCases()
      .then((c) => {
        if (!cancelled) setCases(c);
      })
      .catch((e: Error) => {
        if (!cancelled) setCasesError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /** The trace to render for the selected case (never null after selection). */
  const current: CaseTrace = useMemo(() => {
    // version is read so this memo recomputes after every bump() (which is
    // how ref mutations become visible to the render). eslint-disable is
    // intentional: we DO want to recompute on version even though we don't
    // "use" it for a value.
    void version;
    if (!selectedCase) return emptyTrace();
    return cacheRef.current.get(selectedCase) ?? emptyTrace();
  }, [selectedCase, version]);

  const runsForSelected: RunSummary[] = useMemo(() => {
    void version;
    if (!selectedCase) return [];
    return runsByCaseRef.current.get(selectedCase) ?? [];
  }, [selectedCase, version]);

  const closeStream = useCallback(() => {
    if (eventSourceRef.current) {
      // dispose() closes the EventSource AND cancels the SSE idle watchdog,
      // so a stopped run can never fire a spurious onError ~60s later.
      eventSourceRef.current.dispose();
      eventSourceRef.current = null;
    }
    if (tickerRef.current != null) {
      window.clearInterval(tickerRef.current);
      tickerRef.current = null;
    }
  }, []);

  // Clean up on unmount.
  useEffect(() => closeStream, [closeStream]);

  /**
   * Mutate a SPECIFIC case's cache entry and bump the render. This is the
   * single mutation primitive: every cache write goes through here. Taking
   * the case id explicitly (rather than reading `selectedCase` at call time)
   * is load-bearing — stream handlers, the post-startRca await, the
   * onTransportClosed recovery, and replay fetches all resolve LATER than the
   * user action that started them. If they read `selectedCase` at resolution
   * time, a case switch in between would write the step/trace into the WRONG
   * case's entry. Callers capture the case id at creation time and pass it
   * here, so a step from case A's stream always lands in case A even after the
   * user has moved on to case B.
   */
  const updateForCase = useCallback(
    (caseId: string, fn: (t: CaseTrace) => CaseTrace) => {
      const prev = cacheRef.current.get(caseId) ?? emptyTrace();
      cacheRef.current.set(caseId, fn(prev));
      bump();
    },
    [bump],
  );

  /** Convenience: mutate the currently-selected case's entry. */
  const updateCurrent = useCallback(
    (fn: (t: CaseTrace) => CaseTrace) => {
      if (!selectedCase) return;
      updateForCase(selectedCase, fn);
    },
    [selectedCase, updateForCase],
  );

  /**
   * Best-effort load the persisted-run list for a case into RunHistory.
   * Failures are swallowed (history is non-critical). Declared before the
   * handlers that use it so the useCallback deps close over a stable ref.
   */
  const loadRuns = useCallback(
    async (caseId: string) => {
      setRunsLoading(true);
      try {
        const runs = await fetchRuns(caseId);
        runsByCaseRef.current.set(caseId, runs);
        bump();
      } catch {
        // Run history is best-effort; ignore failures (the user can still run).
      } finally {
        setRunsLoading(false);
      }
    },
    [bump],
  );

  /**
   * Live elapsed-time ticker. Ticks the currently-selected case's elapsedMs
   * while its run is active. Uses the per-case start timestamp
   * (startedAtByCaseRef) so a case switch never produces a non-monotonic jump:
   * each case's elapsed is measured from its own start. Re-arms only when the
   * selected case or its status changes, so a background case's ticker is
   * correctly paused while the user views another case.
   */
  useEffect(() => {
    if (selectedCase && current.status === "running") {
      const startedAt = startedAtByCaseRef.current.get(selectedCase);
      if (startedAt != null) {
        tickerRef.current = window.setInterval(() => {
          updateForCase(selectedCase, (t) => ({
            ...t,
            elapsedMs: Date.now() - startedAt,
          }));
        }, 200);
      }
    }
    return () => {
      if (tickerRef.current != null) {
        window.clearInterval(tickerRef.current);
        tickerRef.current = null;
      }
    };
  }, [selectedCase, current.status, updateForCase]);

  const handleRun = useCallback(async () => {
    if (!selectedCase) return;
    // Capture the case id at call time. EVERY async/stream callback below
    // threads `runCase` into updateForCase so a step or recovery for this run
    // always lands in THIS case's entry — even if the user switches cases
    // before the POST resolves or before a step arrives.
    const runCase = selectedCase;
    closeStream();
    setReplayRunId(null);
    // Reset ONLY this case's cache entry to a fresh run. We do NOT touch other
    // cases' entries — that is the whole point of the per-case cache (switching
    // cases must not wipe a sibling trace).
    cacheRef.current.set(runCase, { ...emptyTrace(), status: "starting" });
    bump();
    const startedAt = Date.now();
    startedAtByCaseRef.current.set(runCase, startedAt);

    const handlers: StreamHandlers = {
      onStep: (step) => {
        // If the user has since replayed a historical run for this case, keep
        // the live step in the live-shadow (not the displayed replayed steps)
        // so it survives a later "back to live".
        updateForCase(runCase, (t) => {
          if (t.replayingRunId != null) {
            // Append to the live shadow, not the displayed replay.
            const liveSteps = t.liveSteps ?? [];
            return { ...t, liveSteps: [...liveSteps, step] };
          }
          return {
            ...t,
            status: t.status === "starting" ? "running" : t.status,
            steps: [...t.steps, step],
          };
        });
      },
      onReport: (rep) => {
        updateForCase(runCase, (t) =>
          t.replayingRunId != null
            ? { ...t, liveReport: rep }
            : { ...t, report: rep },
        );
      },
      onDone: () => {
        const startedAtForCase = startedAtByCaseRef.current.get(runCase);
        if (startedAtForCase != null) {
          updateForCase(runCase, (t) => {
            if (t.replayingRunId != null) {
              return { ...t, liveStatus: "done" };
            }
            return { ...t, elapsedMs: Date.now() - startedAtForCase, status: "done" };
          });
        } else {
          updateForCase(runCase, (t) =>
            t.replayingRunId != null ? { ...t, liveStatus: "done" } : { ...t, status: "done" },
          );
        }
        closeStream();
        // Refresh run history so the just-completed run appears in the
        // RunHistory panel (the backend flushes the run row at completion;
        // the run-start fetch happened before the row existed).
        void loadRuns(runCase);
      },
      onError: (msg) => {
        updateForCase(runCase, (t) => {
          if (t.replayingRunId != null) {
            // A stream error while replaying history shouldn't flip the
            // displayed replay to error; record on the live shadow only.
            return { ...t, liveStatus: "error" };
          }
          return {
            ...t,
            error: msg,
            status: t.status === "running" || t.status === "starting" ? "error" : t.status,
          };
        });
        closeStream();
      },
    };

    let runId: string | null = null;
    try {
      const resp = await startRca(runCase, backend);
      runId = resp.run_id ?? null;
      // Record runId + flip to "running" eagerly (the old single-state model
      // did this right after POST). Eager "running" means the elapsed ticker
      // and the live indicator start as soon as the POST resolves, not only on
      // the first step (which can lag behind LLM warmup).
      updateForCase(runCase, (t) =>
        t.replayingRunId != null
          ? { ...t, runId, liveStatus: "running" }
          : { ...t, runId, status: t.status === "starting" ? "running" : t.status },
      );
    } catch {
      // POST may fail in some setups; the SSE stream is what matters. Continue
      // without a runId — disconnect recovery will then be unavailable.
      // Still flip to running so the ticker starts for an in-flight stream.
      updateForCase(runCase, (t) =>
        t.replayingRunId != null ? t : { ...t, status: t.status === "starting" ? "running" : t.status },
      );
    }

    eventSourceRef.current = openRcaStream(runCase, backend, handlers, {
      runId,
      // onRecover fires on EVERY mid-run drop path: idle-timeout (the most
      // common — a long DeepSeek turn that stops emitting pings), transport
      // CLOSED (network/CORS), and a server-sent error event. In all three
      // cases the live partial trace may be incomplete while the server has a
      // fuller persisted one, so we best-effort fetch it via fetchRun and show
      // it + a banner instead of a bare broken timeline. This is the fix for
      // "stream drops and the user sees an empty trace."
      onRecover: (knownRunId) => {
        // Best-effort fetch the persisted trace and replace what's shown so the
        // user keeps their in-progress work instead of a bare error. Explicitly
        // best-effort: if the network is also down (or no runId is known), we
        // show the in-memory partial trace + a recovery banner.
        const recoverId = knownRunId ?? runId;
        if (!recoverId) {
          updateForCase(runCase, (t) => ({
            ...t,
            disconnectBanner: "connection lost — partial trace shown · retry · 连接中断，显示部分追踪",
          }));
          return;
        }
        fetchRun(recoverId)
          .then((run) => {
            // mountedRef guards the post-unmount write; runCase pins the target
            // so a case switch between drop and resolution can't misroute.
            if (!mountedRef.current) return;
            // Build the recovery banner once: both the replay-shadow branch
            // and the live-trace branch show the SAME banner (only the trace
            // they write into differs), so we hoist it out of the branch to
            // avoid the two copies drifting. 连接断开 — 已加载服务器保存的轨迹（N 步）· 重试
            const stepWord = run.steps.length === 1 ? "step" : "steps";
            const disconnectBanner = `connection lost — loaded server-saved trace (${run.steps.length} ${stepWord}) · retry · 连接断开，已加载服务器保存的轨迹（${run.steps.length} 步）`;
            updateForCase(runCase, (t) => {
              // Don't clobber a replay the user started after the drop: if a
              // replay is now active, write the recovery into the live shadow.
              if (t.replayingRunId != null) {
                return {
                  ...t,
                  liveSteps: run.steps.length > 0 ? run.steps : (t.liveSteps ?? []),
                  liveReport: extractReport(run) ?? t.liveReport,
                  liveStatus: run.status === "completed" ? "done" : "error",
                  disconnectBanner,
                };
              }
              return {
                ...t,
                // Prefer the persisted (server-authoritative) steps/report, but
                // keep the in-memory ones if the server has none yet (the run
                // may not have been flushed before the drop).
                steps: run.steps.length > 0 ? run.steps : t.steps,
                report: extractReport(run) ?? t.report,
                // The run's terminal status is server-authoritative here:
                // completed -> done; anything else (running/error/...) -> the
                // user should see error + the banner. We do NOT branch on
                // t.status (which onError may have already flipped to "error").
                status: run.status === "completed" ? "done" : "error",
                error: run.status === "completed" ? null : t.error,
                disconnectBanner,
              };
            });
            // Refresh run history so the dropped run appears with its final
            // status (the backend marks it interrupted/error on its side).
            void loadRuns(runCase);
          })
          .catch(() => {
            if (!mountedRef.current) return;
            updateForCase(runCase, (t) => ({
              ...t,
              disconnectBanner: "connection lost — partial trace shown · retry · 连接中断，显示部分追踪",
            }));
          });
      },
    });
  }, [selectedCase, backend, closeStream, updateForCase, bump, loadRuns]);

  const handleStop = useCallback(() => {
    closeStream();
    updateCurrent((t) => ({
      ...t,
      status: t.status === "running" || t.status === "starting" ? "done" : t.status,
    }));
  }, [closeStream, updateCurrent]);

  const handleSelectCase = useCallback(
    (id: string) => {
      closeStream();
      // NOTE: deliberately do NOT wipe. If a cached trace exists for `id`,
      // it is shown as-is; otherwise an empty/idle trace is shown. This is
      // the fix for "trace lost on case switch".
      setSelectedCase(id);
      setReplayRunId(null);
      // Best-effort populate RunHistory for the newly-selected case.
      void loadRuns(id);
    },
    [closeStream, loadRuns],
  );

  /**
   * Replay a persisted run's trace into the timeline (RunHistory click).
   * The live run's in-memory trace is saved into the live-shadow fields so a
   * later "Back to live" restores it (and so live SSE steps arriving during
   * the replay aren't lost). The replay overlay is only applied on a
   * successful fetch — a failure surfaces an error WITHOUT setting
   * replayRunId, so the RunHistory highlight can't get stuck on a run whose
   * trace never loaded.
   */
  const handleSelectRun = useCallback(
    async (runId: string) => {
      if (!selectedCase) return;
      const replayCase = selectedCase;
      closeStream();
      try {
        const run = await fetchRun(runId);
        if (!mountedRef.current) return;
        updateForCase(replayCase, (t) => {
          // Snapshot the live trace into the shadow BEFORE overlaying the
          // replay, so "Back to live" can restore it. Only snapshot if there
          // isn't already a shadow (nested replays don't clobber the original).
          const alreadyShadowing = t.replayingRunId != null;
          const shadowSteps = alreadyShadowing ? t.liveSteps : t.steps;
          const shadowReport = alreadyShadowing ? t.liveReport : t.report;
          const shadowStatus = alreadyShadowing ? t.liveStatus : t.status;
          return {
            ...t,
            // Displayed (replayed) trace:
            steps: run.steps,
            report: extractReport(run),
            status: run.status === "completed" ? "done" : "idle",
            error: null,
            disconnectBanner: null,
            replayingRunId: runId,
            // Live shadow (restored on "Back to live"):
            liveSteps: shadowSteps,
            liveReport: shadowReport,
            liveStatus: shadowStatus,
          };
        });
        setReplayRunId(runId);
      } catch (e) {
        // Replay fetch failed: surface but don't crash, and don't set
        // replayRunId (so the history highlight doesn't get stuck).
        if (!mountedRef.current) return;
        updateForCase(replayCase, (t) => ({
          ...t,
          error: `replay failed: ${(e as Error).message}`,
        }));
      }
    },
    [selectedCase, closeStream, updateForCase],
  );

  /**
   * Return to the live (last live) trace for the selected case. Restores the
   * live-shadow snapshot taken when the replay was applied, so the user sees
   * the actual live run again (including any steps that arrived during the
   * replay window).
   */
  const handleBackToLive = useCallback(() => {
    setReplayRunId(null);
    updateCurrent((t) => {
      if (t.replayingRunId == null) return t;
      return {
        ...t,
        steps: t.liveSteps ?? [],
        report: t.liveReport,
        status: t.liveStatus,
        replayingRunId: null,
        liveSteps: null,
        liveReport: null,
        liveStatus: "idle",
      };
    });
  }, [updateCurrent]);

  const status = current.status;
  const replaying = current.replayingRunId != null;

  return (
    <div className="app">
      <header className="app__header">
        <h1>RCA Ops Console · 根因分析控制台</h1>
        <span className={`app__status app__status--${status}`}>
          {status === "idle" && "就绪 / idle"}
          {status === "starting" && "启动中 / starting…"}
          {status === "running" && "运行中 / running"}
          {status === "done" && "完成 / done"}
          {status === "error" && "错误 / error"}
        </span>
      </header>

      <div className="app__body">
        <aside className="app__sidebar">
          <CasePicker
            cases={cases}
            selected={selectedCase}
            onSelect={handleSelectCase}
            loading={cases.length === 0 && casesError == null}
            error={casesError}
          />
          {selectedCase && (
            <div className="app__sidebar-section">
              <RunHistory
                runs={runsForSelected}
                selectedRunId={replaying ? current.replayingRunId : replayRunId}
                onSelectRun={handleSelectRun}
                loading={runsLoading}
              />
            </div>
          )}
        </aside>

        <main className="app__main">
          <ErrorBoundary
            resetKey={selectedCase}
            fallback={(err, reset) => (
              <div className="app__error app__error--fatal" role="alert">
                <h2>Panel render failed · 面板渲染异常</h2>
                <p>{err.message}</p>
                <button type="button" onClick={reset}>
                  Retry · 重试
                </button>
              </div>
            )}
          >
            <RunPanel
              caseId={selectedCase}
              backend={backend}
              onBackendChange={setBackend}
              status={status}
              onRun={handleRun}
              onStop={handleStop}
              stepCount={current.steps.length}
              elapsedMs={current.elapsedMs}
            />

            {current.disconnectBanner && (
              <div className="app__banner app__banner--disconnect" role="status">
                ⚠ {current.disconnectBanner}{" "}
                <button type="button" className="app__banner-action" onClick={handleRun}>
                  Retry · 重试
                </button>
              </div>
            )}

            {current.error && !current.disconnectBanner && (
              <div className="app__error">⚠ {current.error}</div>
            )}

            {replaying && (
              <div className="app__banner app__banner--replay" role="status">
                ⟲ Replaying run{" "}
                <code>{current.replayingRunId}</code>
                <button
                  type="button"
                  className="app__banner-action"
                  onClick={handleBackToLive}
                >
                  Back to live · 返回实时
                </button>
              </div>
            )}

            <TraceTimeline
              steps={current.steps}
              running={status === "running"}
              replay={replaying}
            />

            {current.report && <ReportCard report={current.report} />}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}

/**
 * Derive a display RcaReport from a persisted Run, if the backend included one.
 * The Run type carries an optional `report` field (forwarded by fetchRun from
 * the body's sibling `report` key). We read it directly — no untyped cast — so
 * a missing/renamed field is caught by the type system rather than silently
 * returning null.
 */
function extractReport(run: { report?: RcaReport | null }): RcaReport | null {
  return run.report ?? null;
}
