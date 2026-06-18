// TypeScript mirrors of the backend RCA contracts
// (see rca_agent/contracts/rca.py + rca_agent/contracts/streaming.py).
// These are the source of truth for the frontend.

/** A single step kind in the agent trace. Mirrors contracts.StepKind. */
export type StepKind =
  | "observe"
  | "hypothesize"
  | "investigate"
  | "tool_call"
  | "tool_result"
  | "reasoning"
  | "conclude"
  | "error";

/** A single agent step. Mirrors contracts.RcaStep. */
export interface RcaStep {
  step_id: string;
  case_id: string;
  step_kind: StepKind;
  /** reasoning_content excerpt (optional, for display). */
  thought?: string | null;
  tool_name?: string | null;
  tool_args?: Record<string, unknown> | null;
  tool_result?: Record<string, unknown> | null;
  /** pre-rendered text fed back to the LLM (the human-readable evidence). */
  tool_result_text?: string | null;
  hypothesis?: string | null;
  /** 0..1 */
  confidence?: number | null;
  entities?: string[];
  /** ISO timestamp. */
  ts?: string;
}

/** Reference to an entity implicated in the root cause. */
export interface EntityRef {
  entity_name: string;
  entity_type?: string;
  entity_domain?: string;
  [k: string]: unknown;
}

/** The root cause of an incident. Mirrors contracts.RootCause. */
export interface RootCause {
  summary: string;
  entity_refs?: EntityRef[];
  fault_type?: string | null;
  evidence?: string[];
  confidence: number;
  contributing_factors?: string[];
  recommended_actions?: string[];
}

/** Token usage for the run. */
export interface TokenUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  [k: string]: unknown;
}

/**
 * Run status. Mirrors the backend's run-row status field. `interrupted` is
 * emitted when a run ended without a clean `done` (idle-drop, transport drop,
 * or a server-side cancel) — distinct from `error` (a hard failure) and
 * `truncated` (the agent hit its step budget). Kept permissive (`| string`)
 * so unknown backend statuses don't break the union at the type boundary.
 *
 * NOTE: the literal members are listed for IDE autocomplete/documentation of
 * the known statuses; because `string` is in the union, assignment of any
 * string is accepted (the literals don't narrow the effective type).
 */
export type ReportStatus =
  | "completed"
  | "truncated"
  | "interrupted"
  | "error"
  | string;

/** The final RCA report. Mirrors contracts.RcaReport. */
export interface RcaReport {
  case_id: string;
  task_id: string;
  alert_title: string;
  root_cause: RootCause;
  steps?: RcaStep[];
  started_at?: string;
  finished_at?: string | null;
  model?: string | null;
  token_usage?: TokenUsage | null;
  status: ReportStatus;
}

/** SSE event kinds emitted by the server. Mirrors contracts.SSEEventKind. */
export type SseEventKind = "step" | "delta" | "report" | "error" | "done" | "ping";

/** Generic SSE envelope: `{event, case_id, data, seq}`. Mirrors contracts.SSEEvent. */
export interface SseEvent<T = unknown> {
  event: SseEventKind;
  case_id: string;
  data: T;
  seq: number;
}

/** A fine-grained streaming token (optional; not all runs emit these). */
export interface SseDelta {
  kind: "text" | "reasoning" | "tool_call";
  text?: string | null;
  step_id?: string | null;
}

/** Response of `GET /cases`. */
export interface CasesResponse {
  cases: string[];
}

/** Response of `POST /rca/{case_id}`. */
export interface StartRcaResponse {
  case_id: string;
  backend: string;
  stream_url: string;
  /**
   * The persisted run id assigned by the backend. Present when the store is
   * available; the frontend threads it onto the stream URL and uses it to
   * best-effort recover the persisted trace if the SSE transport drops
   * mid-run (GET /runs/{run_id}). Optional for backward compat with older
   * backends that did not persist runs.
   */
  run_id?: string | null;
}

/**
 * Summary of a persisted run, as returned by `GET /runs` (list view).
 * Mirrors the backend's run-list projection: the heavy `steps` array is NOT
 * included — load it on demand via {@link Run} / `GET /runs/{run_id}`.
 */
export interface RunSummary {
  run_id: string;
  case_id: string;
  status: string;
  model?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  token_usage?: TokenUsage | null;
  step_count: number;
}

/**
 * Full run record, as returned by `GET /runs/{run_id}`: the {@link RunSummary}
 * fields plus the persisted `steps` array. The backend returns these as two
 * sibling keys (`run` + `steps`) in the JSON body; {@link fetchRun} merges them.
 *
 * `report` is optional: when the backend persists the final RcaReport alongside
 * the run (a sibling `report` key in the body), {@link fetchRun} surfaces it
 * here so the UI can render the ReportCard on replay / disconnect recovery
 * without a separate fetch. Absent for backends/runs that don't store it.
 */
export interface Run extends RunSummary {
  steps: RcaStep[];
  report?: RcaReport | null;
}

/** Data backend selector. */
export type Backend = "parquet" | "clickhouse";
