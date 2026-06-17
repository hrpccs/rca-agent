import { useEffect, useMemo, useRef, useState } from "react";
import type { RcaStep, StepKind } from "../types";

interface TraceTimelineProps {
  steps: RcaStep[];
  running: boolean;
  /**
   * When true, render a small "replay" badge in the header to indicate the
   * displayed trace is a persisted run being replayed (not the live run).
   * Optional and purely cosmetic; defaults to false so existing callers are
   * unaffected.
   */
  replay?: boolean;
}

/**
 * A timeline entry: either a standalone step, or a tool_call paired with its
 * matching tool_result (matched by tool_name proximity — the agent emits a
 * tool_call immediately followed by its tool_result).
 */
type Entry =
  | { type: "single"; step: RcaStep; index: number }
  | { type: "tool"; call: RcaStep; result: RcaStep | null; index: number };

const KIND_META: Record<StepKind, { label: string; cls: string; glyph: string }> = {
  reasoning: { label: "Reasoning / 推理", cls: "kind--reasoning", glyph: "💭" },
  observe: { label: "Observe / 观察", cls: "kind--observe", glyph: "👁" },
  hypothesize: { label: "Hypothesis / 假设", cls: "kind--hyp", glyph: "🔍" },
  investigate: { label: "Investigate / 调查", cls: "kind--investigate", glyph: "🧪" },
  tool_call: { label: "Tool", cls: "kind--tool", glyph: "🔧" },
  tool_result: { label: "Evidence", cls: "kind--tool", glyph: "🔧" },
  conclude: { label: "Conclusion / 结论", cls: "kind--conclude", glyph: "✅" },
  error: { label: "Error", cls: "kind--error", glyph: "⚠" },
};

/**
 * Group raw steps into timeline entries, pairing each tool_call with its
 * matching tool_result. The agent emits tool_call immediately followed by its
 * tool_result; a tool_call is paired only with the very next step if that step
 * is a tool_result. Intervening steps of other kinds are NEVER skipped — they
 * are emitted as standalone entries so the trace never loses a step.
 */
function groupSteps(steps: RcaStep[]): Entry[] {
  const entries: Entry[] = [];
  const used = new Set<number>();
  for (let i = 0; i < steps.length; i++) {
    if (used.has(i)) continue;
    const s = steps[i];
    if (s.step_kind === "tool_call") {
      // Pair only with an immediately-following tool_result.
      const next = steps[i + 1];
      if (next && next.step_kind === "tool_result") {
        used.add(i + 1);
        entries.push({ type: "tool", call: s, result: next, index: i });
      } else {
        entries.push({ type: "tool", call: s, result: null, index: i });
      }
    } else {
      entries.push({ type: "single", step: s, index: i });
    }
  }
  return entries;
}

function shortArgs(args: Record<string, unknown> | null | undefined): string {
  if (!args) return "";
  try {
    const s = JSON.stringify(args);
    return s.length > 400 ? s.slice(0, 400) + "…" : s;
  } catch {
    return String(args);
  }
}

function TruncatedText({
  text,
  max = 600,
}: {
  text: string | null | undefined;
  max?: number;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!text) return null;
  const tooLong = text.length > max;
  const shown = expanded || !tooLong ? text : text.slice(0, max) + " …";
  return (
    <div className="timeline__text">
      <pre>{shown}</pre>
      {tooLong && (
        <button
          type="button"
          className="timeline__expand"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "collapse" : `expand (+${text.length - max})`}
        </button>
      )}
    </div>
  );
}

/** The live streaming trace of RcaSteps, grouped by kind with tool pairing. */
export function TraceTimeline({ steps, running, replay = false }: TraceTimelineProps) {
  const entries = useMemo(() => groupSteps(steps), [steps]);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Auto-scroll to newest entry while streaming, unless the user scrolled up.
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [entries, autoScroll]);

  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    setAutoScroll(atBottom);
  };

  return (
    <section className="timeline" onScroll={onScroll}>
      <div className="timeline__head">
        <h2>Trace Timeline · 追踪时间线</h2>
        <span className="timeline__count">
          {steps.length} step{steps.length === 1 ? "" : "s"}
          {replay && <span className="timeline__replay">⟲ replay</span>}
          {running && <span className="timeline__live">● live</span>}
        </span>
      </div>

      {entries.length === 0 && (
        <div className="timeline__empty">
          {running ? "Waiting for the first step…" : "Select a case and run RCA to see the live trace."}
        </div>
      )}

      <ol className="timeline__list">
        {entries.map((entry) => {
          if (entry.type === "tool") {
            const call = entry.call;
            const meta = KIND_META[call.step_kind];
            return (
              <li key={call.step_id ?? `tool-${entry.index}`} className={`timeline__item ${meta.cls} animate-in`}>
                <span className="timeline__glyph">{meta.glyph}</span>
                <div className="timeline__content">
                  <div className="timeline__row">
                    <span className="timeline__kind">{meta.label}</span>
                    <code className="timeline__tool">{call.tool_name ?? "tool"}</code>
                    {call.ts && <span className="timeline__ts">{new Date(call.ts).toLocaleTimeString()}</span>}
                  </div>
                  {call.tool_args && (
                    <div className="timeline__args">
                      <span className="timeline__args-label">args:</span>
                      <code>{shortArgs(call.tool_args)}</code>
                    </div>
                  )}
                  {entry.result && (
                    <div className="timeline__evidence">
                      <span className="timeline__evidence-label">evidence:</span>
                      <TruncatedText text={entry.result.tool_result_text} />
                    </div>
                  )}
                </div>
              </li>
            );
          }

          const step = entry.step;
          const meta = KIND_META[step.step_kind];
          return (
            <li key={step.step_id ?? `s-${entry.index}`} className={`timeline__item ${meta.cls} animate-in`}>
              <span className="timeline__glyph">{meta.glyph}</span>
              <div className="timeline__content">
                <div className="timeline__row">
                  <span className="timeline__kind">{meta.label}</span>
                  {step.ts && <span className="timeline__ts">{new Date(step.ts).toLocaleTimeString()}</span>}
                </div>

                {step.step_kind === "reasoning" && step.thought && (
                  <TruncatedText text={step.thought} />
                )}

                {(step.step_kind === "hypothesize" || step.step_kind === "conclude") && step.hypothesis && (
                  <div className="timeline__hypothesis">
                    <p>{step.hypothesis}</p>
                    {step.confidence != null && (
                      <div className="timeline__conf">
                        <span className="timeline__conf-label">confidence</span>
                        <span className="timeline__conf-value">{(step.confidence * 100).toFixed(0)}%</span>
                      </div>
                    )}
                  </div>
                )}

                {step.step_kind === "conclude" && step.thought && (
                  <TruncatedText text={step.thought} />
                )}

                {step.entities && step.entities.length > 0 && (
                  <div className="timeline__entities">
                    {step.entities.map((e) => (
                      <span key={e} className="chip">{e}</span>
                    ))}
                  </div>
                )}

                {step.step_kind === "error" && (
                  <div className="timeline__text">
                    <pre>{step.thought ?? "error"}</pre>
                  </div>
                )}
              </div>
            </li>
          );
        })}
        <div ref={bottomRef} />
      </ol>
    </section>
  );
}
