import type {
  Backend,
  CasesResponse,
  RcaReport,
  RcaStep,
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

/** Start (or re-fetch) an RCA run, returning the SSE stream URL. */
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

/** Callbacks invoked as SSE events are parsed from the stream. */
export interface StreamHandlers {
  onStep?: (step: RcaStep, ev: SseEvent<RcaStep>) => void;
  onDelta?: (delta: SseDelta, ev: SseEvent<SseDelta>) => void;
  onReport?: (report: RcaReport, ev: SseEvent<RcaReport>) => void;
  onDone?: (status: unknown, ev: SseEvent) => void;
  onError?: (message: string, ev: SseEvent<{ error?: string }>) => void;
}

/**
 * Open an SSE stream for an RCA run and dispatch typed events.
 *
 * Returns the underlying EventSource so callers can `.close()` it. The server
 * emits SSE messages of the form:
 *   event: <step|report|done|error|delta|ping>
 *   data:  {"event":<kind>,"case_id":"t001","data":<payload>,"seq":<int>}
 *
 * The payload under `.data` is a serialized RcaStep / RcaReport / dict.
 */
export function openRcaStream(
  caseId: string,
  backend: Backend,
  handlers: StreamHandlers,
): EventSource {
  const url = withBase(`/rca/${encodeURIComponent(caseId)}/stream?backend=${backend}`);
  const es = new EventSource(url);

  const makeListener = (kind: SseEventKind) => (msg: MessageEvent<string>) => {
    // A native EventSource transport error fires the "error" listener with no
    // `data` (msg.data === undefined). Don't try to parse it; let es.onerror
    // below handle native transport failures.
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
        handlers.onDone?.(payload, envelope);
        es.close();
        break;
      case "error": {
        const errMsg =
          (payload as { error?: string })?.error ??
          `stream error for case ${caseId}`;
        handlers.onError?.(errMsg, envelope as SseEvent<{ error?: string }>);
        es.close();
        break;
      }
      case "ping":
        // keep-alive; ignore
        break;
      default:
        break;
    }
  };

  (["step", "delta", "report", "done", "error", "ping"] as SseEventKind[]).forEach((k) => {
    es.addEventListener(k, makeListener(k) as EventListener);
  });

  // If the transport itself fails (network, CORS, non-200), surface it once.
  es.onerror = () => {
    // Only surface if still connecting/open and not already closed by done/error.
    if (es.readyState === EventSource.CLOSED) {
      handlers.onError?.(`stream closed for case ${caseId}`, {
        event: "error",
        case_id: caseId,
        data: { error: "EventSource closed" },
        seq: 0,
      });
    }
  };

  return es;
}
