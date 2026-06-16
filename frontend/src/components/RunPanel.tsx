import type { Backend } from "../types";

interface RunPanelProps {
  caseId: string | null;
  backend: Backend;
  onBackendChange: (b: Backend) => void;
  status: "idle" | "starting" | "running" | "done" | "error";
  onRun: () => void;
  onStop: () => void;
  stepCount: number;
  elapsedMs: number;
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${(s - m * 60).toFixed(0)}s`;
}

/** Top control bar: selected case, backend toggle, run/stop, live counters. */
export function RunPanel({
  caseId,
  backend,
  onBackendChange,
  status,
  onRun,
  onStop,
  stepCount,
  elapsedMs,
}: RunPanelProps) {
  const running = status === "running" || status === "starting";

  return (
    <section className="run-panel">
      <div className="run-panel__case">
        <span className="run-panel__label">Case</span>
        <code className="run-panel__id">{caseId ?? "—"}</code>
      </div>

      <div className="run-panel__backend" role="group" aria-label="Data backend">
        <span className="run-panel__label">Backend</span>
        <div className="run-panel__toggle">
          {(["parquet", "clickhouse"] as Backend[]).map((b) => (
            <button
              key={b}
              type="button"
              className={`run-panel__backend-btn ${backend === b ? "is-active" : ""}`}
              onClick={() => onBackendChange(b)}
              disabled={running}
              aria-pressed={backend === b}
            >
              {b}
            </button>
          ))}
        </div>
      </div>

      <div className="run-panel__metrics">
        <div className="run-panel__metric">
          <span className="run-panel__metric-label">Steps</span>
          <span className="run-panel__metric-value">{stepCount}</span>
        </div>
        <div className="run-panel__metric">
          <span className="run-panel__metric-label">Elapsed</span>
          <span className="run-panel__metric-value">{formatMs(elapsedMs)}</span>
        </div>
      </div>

      <div className="run-panel__actions">
        {!running ? (
          <button
            type="button"
            className="run-panel__run"
            onClick={onRun}
            disabled={!caseId}
          >
            ▶ Run RCA
          </button>
        ) : (
          <button type="button" className="run-panel__stop" onClick={onStop}>
            ■ Stop
          </button>
        )}
      </div>
    </section>
  );
}
