import type { RunSummary, TokenUsage } from "../types";

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

/** Summarize token usage as a compact "1.2k tok" string, or null when absent. */
function formatTokens(usage?: TokenUsage | null): string | null {
  if (!usage) return null;
  const total =
    typeof usage.total_tokens === "number"
      ? usage.total_tokens
      : (typeof usage.prompt_tokens === "number" && typeof usage.completion_tokens === "number"
          ? usage.prompt_tokens + usage.completion_tokens
          : null);
  // Reject null AND non-finite/non-positive values: a corrupted backend row
  // could carry NaN/Infinity/negative totals, which would otherwise render as
  // "NaN tok" / "Infinityk tok" / "-1 tok". Hide the field in those cases.
  if (total == null || !Number.isFinite(total) || total <= 0) return null;
  if (total >= 1000) return `${(total / 1000).toFixed(1)}k tok`;
  return `${total} tok`;
}

/**
 * Map a raw status string to the CSS modifier class used for the badge. Unknown
 * statuses (and the explicit `interrupted`/`error` kinds) get distinct colors
 * so a user can tell at a glance whether a run ended cleanly, was cut off
 * (`interrupted` — the idle/transport-drop outcome), or hard-failed (`error`).
 */
function statusClass(status: string): string {
  switch (status) {
    case "completed":
      return "run-history__status--completed";
    case "running":
      return "run-history__status--running";
    case "truncated":
      return "run-history__status--truncated";
    case "interrupted":
      return "run-history__status--interrupted";
    case "error":
      return "run-history__status--error";
    default:
      return "run-history__status--unknown";
  }
}

/**
 * Sidebar panel: list of persisted runs for the selected case. Purely
 * presentational — the parent owns fetching (`runs`) and replay wiring
 * (`onSelectRun`). Clicking a row loads that run's persisted trace into the
 * timeline; the active/selected run is highlighted and distinguished from the
 * live run (which has no row selected).
 *
 * Each row shows: status badge (color-coded; `interrupted` and `error` are
 * visually distinct from `completed`/`running`), started time, short run id,
 * step count, and token usage (when the backend recorded it). This is the
 * "运行记录" surfacing — the user's past runs are visible here so they never
 * lose track of what they ran.
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
          const tokens = formatTokens(run.token_usage);
          return (
            <li key={run.run_id}>
              <button
                type="button"
                className={`run-history__item ${active ? "run-history__item--active" : ""}`}
                onClick={() => onSelectRun(run.run_id)}
                aria-pressed={active}
                aria-label={`Run ${run.run_id}, status ${run.status}, ${run.step_count} steps`}
              >
                <span className={`run-history__status ${statusClass(run.status)}`}>
                  {run.status}
                </span>
                <span className="run-history__meta">
                  <span className="run-history__time">{formatStartedAt(run.started_at)}</span>
                  <span className="run-history__id-line">
                    <span className="run-history__id" title={run.run_id}>
                      {run.run_id.length > 10 ? run.run_id.slice(0, 8) + "…" : run.run_id}
                    </span>
                    <span className="run-history__steps">{run.step_count} step{run.step_count === 1 ? "" : "s"}</span>
                    {tokens && <span className="run-history__tokens">{tokens}</span>}
                  </span>
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
