import type { SseEvent } from "../types";

/**
 * Build an SSE envelope in the backend wire format
 * `{event, case_id, data, seq}` for tests. Shared by api.test.ts and
 * App.test.tsx so the wire-format shape lives in one place.
 */
export function sseEnvelope<T>(
  event: SseEvent["event"],
  caseId: string,
  data: T,
  seq: number,
): SseEvent<T> {
  return { event, case_id: caseId, data, seq };
}
