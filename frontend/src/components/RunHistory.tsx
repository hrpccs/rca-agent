import type { RunSummary } from "../types";

interface RunHistoryProps {
  /** Persisted runs for the currently-selected case (most-recent first). */
  runs: RunSummary[];
  /** The run currently being replayed (live run has no selection here). */
  selectedRunId?: string | null;
  /** Invoked when the user clicks a run row to replay its trace. */
  onSelectRun: (runId: string) => void;
  loading?: boolean;
}

function formatStartedAt(iso?: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

/**
 * Sidebar panel: list of persisted runs for the selected case. Purely
 * presentational — the parent owns fetching (`runs`) and replay wiring
 * (`onSelectRun`). Clicking a row loads that run's persisted trace into the
 * timeline; the active/selected run is highlighted and distinguished from the
 * live run (which has no row selected).
 */
export function RunHistory({ runs, selectedRunId, onSelectRun, loading }: RunHistoryProps) {
  return (
    <section className="run-history" aria-label="Run history for selected case">
      <div className="run-history__head">
        <span className="run-history__title">Run History · 历史记录</span>
        <span className="run-history__count">
          {loading ? "loading…" : `${runs.length}`}
        </span>
      </div>

      {!loading && runs.length === 0 && (
        <div className="run-history__empty">No past runs yet. · 暂无历史。</div>
      )}

      <ul className="run-history__list">
        {runs.map((run) => {
          const active = selectedRunId === run.run_id;
          return (
            <li key={run.run_id}>
              <button
                type="button"
                className={`run-history__item ${active ? "run-history__item--active" : ""}`}
                onClick={() => onSelectRun(run.run_id)}
                aria-pressed={active}
              >
                <span className={`run-history__status run-history__status--${run.status}`}>
                  {run.status}
                </span>
                <span className="run-history__meta">
                  <span className="run-history__time">{formatStartedAt(run.started_at)}</span>
                  <span className="run-history__id" title={run.run_id}>
                    {run.run_id.length > 10 ? run.run_id.slice(0, 8) + "…" : run.run_id}
                  </span>
                  <span className="run-history__steps">{run.step_count} step{run.step_count === 1 ? "" : "s"}</span>
                </span>
                {active && <span className="run-history__badge">replay</span>}
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
