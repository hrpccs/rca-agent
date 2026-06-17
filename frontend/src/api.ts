import type {
  Backend,
  CasesResponse,
  RcaReport,
  RcaStep,
  Run,
  RunSummary,
  StartRcaResponse,
  SseDelta,
  SseEvent,
  SseEventKind,
} from "./types";

/**
 * API client for the RCA backend.
 *
 * In dev, all paths are relative and proxied to http://localhost:8000 by Vite
 * (see vite.config.ts). In production behind the same origin, relative paths
 * also work. To target a remote backend instead, set VITE_API_BASE.
 */
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

function withBase(path: string): string {
  if (!API_BASE) return path;
  return `${API_BASE.replace(/\/$/, "")}${path}`;
}

/** Fetch the available case ids. */
export async function fetchCases(): Promise<string[]> {
  const res = await fetch(withBase("/cases"));
  if (!res.ok) {
    throw new Error(`fetchCases failed: ${res.status} ${res.statusText}`);
  }
  const body = (await res.json()) as CasesResponse;
  return body.cases ?? [];
}

/** Start (or re-fetch) an RCA run, returning the SSE stream URL (+ run_id). */
export async function startRca(caseId: string, backend: Backend = "parquet"): Promise<StartRcaResponse> {
  const res = await fetch(withBase(`/rca/${encodeURIComponent(caseId)}?backend=${backend}`), {
    method: "POST",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`startRca failed: ${res.status} ${res.statusText} ${text}`);
  }
  return (await res.json()) as StartRcaResponse;
}

/**
 * List persisted runs. Pass a `caseId` to filter to one case
 * (`GET /runs?case_id=…`); omit it for the global list. Returns the `runs`
 * array (never null) so callers can `.map` without a guard.
 *
 * Envelope shape: `{ runs: RunSummary[] }`.
 */
export async function fetchRuns(caseId?: string | null): Promise<RunSummary[]> {
  const path =
    caseId != null && caseId !== ""
      ? `/runs?case_id=${encodeURIComponent(caseId)}`
      : "/runs";
  const res = await fetch(withBase(path));
  if (!res.ok) {
    throw new Error(`fetchRuns failed: ${res.status} ${res.statusText}`);
  }
  const body = (await res.json()) as { runs?: RunSummary[] };
  return body.runs ?? [];
}

/**
 * Fetch a single persisted run with its steps. The backend returns
 * `{ run: RunSummary, steps: RcaStep[], report?: RcaReport }`; we merge the
 * three into a {@link Run} so callers get one typed object. Steps default to
 * `[]` when absent; `report` is forwarded when the backend includes it.
 */
export async function fetchRun(runId: string): Promise<Run> {
  const res = await fetch(withBase(`/runs/${encodeURIComponent(runId)}`));
  if (!res.ok) {
    throw new Error(`fetchRun failed: ${res.status} ${res.statusText}`);
  }
  const body = (await res.json()) as {
    run: RunSummary;
    steps?: RcaStep[];
    report?: RcaReport | null;
  };
  return {
    ...body.run,
    steps: body.steps ?? [],
    ...(body.report !== undefined ? { report: body.report } : {}),
  };
}

/** Fetch only the persisted steps for a run (`GET /runs/{runId}/steps`). */
export async function fetchRunSteps(runId: string): Promise<RcaStep[]> {
  const res = await fetch(withBase(`/runs/${encodeURIComponent(runId)}/steps`));
  if (!res.ok) {
    throw new Error(`fetchRunSteps failed: ${res.status} ${res.statusText}`);
  }
  const body = (await res.json()) as { steps?: RcaStep[] };
  return body.steps ?? [];
}

/** Callbacks invoked as SSE events are parsed from the stream. */
export interface StreamHandlers {
  onStep?: (step: RcaStep, ev: SseEvent<RcaStep>) => void;
  onDelta?: (delta: SseDelta, ev: SseEvent<SseDelta>) => void;
  onReport?: (report: RcaReport, ev: SseEvent<RcaReport>) => void;
  onDone?: (status: unknown, ev: SseEvent) => void;
  onError?: (message: string, ev: SseEvent<{ error?: string }>) => void;
}

/** Options for {@link openRcaStream}. */
export interface OpenStreamOptions {
  /**
   * If a positive, finite number, arm an inactivity watchdog: when no SSE
   * event is received for this many milliseconds, the stream is closed and a
   * terminal `onError` is fired. Each received event re-arms the timer.
   *
   * `Infinity`, `NaN`, and values `<= 0` disable the watchdog entirely
   * (Infinity is NOT clamped to 0 — that would kill the stream instantly).
   * Default: 60000 (60s).
   */
  idleTimeoutMs?: number;
  /**
   * Optional run id to thread onto the stream URL
   * (`…&run_id=…`). When the backend persisted a run for this invocation, pass
   * its id so the server can correlate the live SSE stream with the stored run
   * row (and so the client knows which run to recover via `fetchRun` if the
   * transport drops mid-run).
   */
  runId?: string | null;
  /**
   * Resilience hook: invoked ONCE, with the runId (if known), when the SSE
   * *transport* fails mid-run (native `error` event while the readyState is
   * CLOSED) — as opposed to an idle-timeout, a clean `done`, or a server-sent
   * `error` event. This is the distinct signal that "the connection itself
   * died and there may be a persisted trace worth fetching." Use it to
   * best-effort load the persisted trace via {@link fetchRun}; note that
   * `onError` ALSO fires with the `stream closed for case …` message, so this
   * callback is purely an extra *channel* to distinguish a transport drop from
   * a hard server error. If `runId` was not supplied, the argument is null and
   * the caller should fall back to the in-memory partial trace.
   */
  onTransportClosed?: (runId: string | null) => void;
}

/**
 * Handle returned by {@link openRcaStream}. Calling `dispose()` (or `close()`)
 * closes the underlying EventSource AND clears any armed idle watchdog and
 * marks the stream terminated, so a stopped run can never fire a spurious
 * terminal callback ~60s later.
 *
 * `es` is exposed for callers that need to inspect `readyState`; the wrapped
 * `close` delegates to `dispose` so existing `.close()` call sites keep the
 * disposal contract.
 */
export interface RcaStreamHandle {
  /** The underlying EventSource. Do not call `.close()` on it directly — use dispose(). */
  readonly es: EventSource;
  /** Close the stream and clear the idle watchdog. Safe to call repeatedly. */
  dispose: () => void;
  /** Alias for {@link RcaStreamHandle.dispose}. */
  close: () => void;
}

/**
 * A thin wrapper that funnels all teardown through one `dispose` path so the
 * idle watchdog timer is always cancelled. `close` is an alias for `dispose`
 * to keep EventSource-shaped call sites working.
 */
function wrapStream(es: EventSource, disposeImpl: () => void): RcaStreamHandle {
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    disposeImpl();
  };
  return { es, dispose, close: dispose };
}

/**
 * Open an SSE stream for an RCA run and dispatch typed events.
 *
 * The server emits SSE messages of the form:
 *   event: <step|report|done|error|delta|ping>
 *   data:  {"event":<kind>,"case_id":"t001","data":<payload>,"seq":<int>}
 *
 * The payload under `.data` is a serialized RcaStep / RcaReport / dict.
 *
 * Inactivity watchdog: when `idleTimeoutMs` is a positive finite number, a
 * timer fires `onError("stream idle …")` and closes the stream if no event
 * arrives within that window. Each received event re-arms the timer. The
 * timer is armed AFTER the native-transport early-return, so a reconnecting
 * EventSource hammering a dead endpoint does NOT keep re-arming the watchdog
 * on failed attempts (which would defeat it). Stopping the run via the
 * returned handle's `dispose()`/`close()` clears the timer and marks the
 * stream terminated, preventing any spurious late callback.
 */
export function openRcaStream(
  caseId: string,
  backend: Backend,
  handlers: StreamHandlers,
  options: OpenStreamOptions = {},
): RcaStreamHandle {
  const idleTimeoutMs = options.idleTimeoutMs ?? DEFAULT_IDLE_TIMEOUT_MS;
  const armIdle = shouldArmIdle(idleTimeoutMs);
  const runId = options.runId ?? null;

  // Build the stream URL. `run_id` is threaded through as a query param so the
  // backend can correlate this SSE stream with its persisted run row (the
  // sibling backend unit uses it to attach emitted steps to the right run).
  // Omitted entirely when no runId is known (older backends).
  const qs = `backend=${backend}${runId ? `&run_id=${encodeURIComponent(runId)}` : ""}`;
  const url = withBase(`/rca/${encodeURIComponent(caseId)}/stream?${qs}`);
  const es = new EventSource(url);

  // --- Idle watchdog state -------------------------------------------------
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  let terminated = false;

  const clearIdle = () => {
    if (idleTimer != null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
  };

  const rearmIdle = () => {
    // No-op if the watchdog is disabled OR the stream is already torn down.
    // The terminated check defends against a straggler named event arriving
    // after dispose/idle/done/error: without it, a fresh 60s timer would be
    // armed on a closed stream and never cleared (disposeInternal is then a
    // no-op), leaking the timer + its captured handlers/caseId closure.
    if (!armIdle || terminated) return;
    clearIdle();
    idleTimer = setTimeout(() => {
      // Fire-and-forget a terminal callback, then go through dispose() so the
      // EventSource is closed and the watchdog cannot fire again. try/finally
      // guarantees teardown even if the user's onError throws — matching the
      // done/error listener contract and preventing a dangling open stream.
      idleTimer = null;
      if (terminated) return;
      try {
        handlers.onError?.(`stream idle for case ${caseId} (no event in ${idleTimeoutMs}ms)`, {
          event: "error",
          case_id: caseId,
          data: { error: `stream idle (>${idleTimeoutMs}ms)` },
          seq: 0,
        });
      } finally {
        disposeInternal();
      }
    }, idleTimeoutMs);
  };

  // Single source of truth for teardown. Cancels the watchdog, marks the
  // stream terminated, and closes the EventSource. Idempotent.
  const disposeInternal = () => {
    if (terminated) return;
    terminated = true;
    clearIdle();
    es.close();
  };

  const makeListener = (kind: SseEventKind) => (msg: MessageEvent<string>) => {
    // Drop any event delivered after the stream was torn down (idle, transport
    // CLOSED, done, error, or dispose). This is the single fence that prevents
    // a straggler event from re-arming the watchdog and re-firing a terminal
    // callback on a closed stream.
    if (terminated) return;
    // A native EventSource transport error fires the "error" listener with no
    // `data` (msg.data === undefined). Don't try to parse it, and DO NOT arm
    // the idle watchdog here — a reconnecting EventSource retrying a dead
    // endpoint would otherwise re-arm the watchdog on every failed attempt and
    // hang the UI forever. Let es.onerror below handle native transport
    // failures.
    if (kind === "error" && (msg as MessageEvent).data == null) {
      return;
    }
    let envelope: SseEvent;
    try {
      envelope = JSON.parse(msg.data) as SseEvent;
    } catch (e) {
      handlers.onError?.(`failed to parse ${kind} SSE data: ${(e as Error).message}`, {
        event: kind,
        case_id: caseId,
        data: { error: msg.data },
        seq: 0,
      });
      return;
    }

    // We received a real event: re-arm the inactivity watchdog (no-op when
    // disabled). Done after the early-returns above per the watchdog contract.
    rearmIdle();

    const payload = envelope.data;
    switch (kind) {
      case "step":
        handlers.onStep?.(payload as RcaStep, envelope as SseEvent<RcaStep>);
        break;
      case "delta":
        handlers.onDelta?.(payload as SseDelta, envelope as SseEvent<SseDelta>);
        break;
      case "report":
        handlers.onReport?.(payload as RcaReport, envelope as SseEvent<RcaReport>);
        break;
      case "done":
        // Terminal: invoke the user callback, then ALWAYS tear down — even if
        // the callback throws — so the watchdog can never be left armed on a
        // stream the caller considers finished.
        try {
          handlers.onDone?.(payload, envelope);
        } finally {
          disposeInternal();
        }
        break;
      case "error": {
        const errMsg =
          (payload as { error?: string })?.error ??
          `stream error for case ${caseId}`;
        try {
          handlers.onError?.(errMsg, envelope as SseEvent<{ error?: string }>);
        } finally {
          disposeInternal();
        }
        break;
      }
      case "ping":
        // keep-alive; ignore (but the watchdog was re-armed above)
        break;
      default:
        break;
    }
  };

  (["step", "delta", "report", "done", "error", "ping"] as SseEventKind[]).forEach((k) => {
    es.addEventListener(k, makeListener(k) as EventListener);
  });

  // If the transport itself fails (network down, non-200, CORS), EventSource
  // fires `error` and auto-retries while CONNECTING, then ends in CLOSED.
  // Surface once when the connection has truly failed (CLOSED) so callers can
  // stop the run instead of waiting on retries that will never deliver data.
  // We ALSO fire `onTransportClosed` (once) with the known runId so the caller
  // can best-effort recover the persisted trace — this is distinct from the
  // idle-timeout, a clean `done`, and a server-sent `error` event, none of
  // which reach this CLOSED branch.
  let nativeErrorSurfaced = false;
  es.onerror = () => {
    if (nativeErrorSurfaced) return;
    // If the stream was already terminated cleanly (done/error/dispose/idle),
    // the EventSource may still emit a final CLOSED error during teardown.
    // Don't surface a spurious terminal callback after an orderly close.
    if (terminated) {
      nativeErrorSurfaced = true;
      return;
    }
    if (es.readyState === EventSource.CLOSED) {
      nativeErrorSurfaced = true;
      // Teardown FIRST so neither user callback can leave the EventSource
      // open (matching the listener contract: disposeInternal runs before the
      // user-facing terminal callback is invoked). Then fire BOTH callbacks
      // independently — one throwing must not suppress the other, since they
      // carry distinct signals (onTransportClosed => "try to fetch the
      // persisted trace"; onError => "show the error banner").
      disposeInternal(/* transport */);
      try {
        options.onTransportClosed?.(runId);
      } catch {
        // Swallow only here: onError below must still fire even if the
        // resilience hook throws. The hook is best-effort by design.
      }
      handlers.onError?.(`stream closed for case ${caseId}`, {
        event: "error",
        case_id: caseId,
        data: { error: "EventSource closed" },
        seq: 0,
      });
    }
  };

  // Arm the initial watchdog now that listeners are wired (no-op if disabled).
  rearmIdle();

  // Funnel all teardown through disposeInternal so the watchdog is cancelled.
  return wrapStream(es, () => disposeInternal(/* dispose */));
}

/** Default inactivity window before an idle stream is forcibly closed. */
export const DEFAULT_IDLE_TIMEOUT_MS = 60_000;

/**
 * Whether the idle watchdog should be armed for the given delay. Only positive
 * finite numbers qualify; `Infinity` is explicitly rejected (clamping it to 0
 * via setTimeout would kill the stream instantly), as is `NaN` (which must not
 * silently disable the feature by accident).
 */
export function shouldArmIdle(idleTimeoutMs: number): boolean {
  return Number.isFinite(idleTimeoutMs) && idleTimeoutMs > 0;
}
