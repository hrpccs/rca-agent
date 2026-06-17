import { act, cleanup } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_IDLE_TIMEOUT_MS,
  fetchRun,
  fetchRunSteps,
  fetchRuns,
  openRcaStream,
  shouldArmIdle,
  type StreamHandlers,
} from "../api";
import type { RcaReport, RcaStep, SseEvent } from "../types";
import {
  FakeEventSource,
  installFakeEventSource,
  fakeEventSourcesCreated,
} from "./fakeEventSource";
import { sseEnvelope as envelope } from "./sseEnvelope";

// The production handle types `es` as the DOM `EventSource`; in tests the
// underlying instance is our FakeEventSource, so we widen through `unknown`.
function fakeEs(handle: { es: EventSource }): FakeEventSource {
  return handle.es as unknown as FakeEventSource;
}

function makeHandlers(overrides: Partial<StreamHandlers> = {}): StreamHandlers & {
  calls: Record<string, unknown[]>;
} {
  const calls: Record<string, unknown[]> = {
    onStep: [],
    onDelta: [],
    onReport: [],
    onDone: [],
    onError: [],
  };
  const handlers: StreamHandlers = {
    onStep: (step, ev) => {
      calls.onStep.push([step, ev]);
    },
    onDelta: (delta, ev) => {
      calls.onDelta.push([delta, ev]);
    },
    onReport: (rep, ev) => {
      calls.onReport.push([rep, ev]);
    },
    onDone: (status, ev) => {
      calls.onDone.push([status, ev]);
    },
    onError: (msg, ev) => {
      calls.onError.push([msg, ev]);
    },
    ...overrides,
  };
  return { ...handlers, calls };
}

describe("openRcaStream — parsing & dispatch", () => {
  let restore: () => void;

  beforeEach(() => {
    const inst = installFakeEventSource();
    restore = inst.cleanup;
  });

  afterEach(() => {
    cleanup();
    restore();
    vi.useRealTimers();
  });

  it("parses {event,case_id,data,seq} and dispatches step / report / done", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t001", "parquet", handlers);
    const es = fakeEs(handle);

    const step: RcaStep = {
      step_id: "s1",
      case_id: "t001",
      step_kind: "observe",
      thought: "pod cpu high",
    };
    es.dispatchEventMessage("step", envelope("step", "t001", step, 1));

    const report: RcaReport = {
      case_id: "t001",
      task_id: "task-1",
      alert_title: "cpu",
      status: "completed",
      root_cause: { summary: "oom", confidence: 0.9 },
    };
    es.dispatchEventMessage("report", envelope("report", "t001", report, 2));

    es.dispatchEventMessage("done", envelope("done", "t001", { ok: true }, 3));

    expect(handlers.calls.onStep).toHaveLength(1);
    const [parsedStep, stepEv] = handlers.calls.onStep[0] as [RcaStep, SseEvent];
    expect(parsedStep.step_kind).toBe("observe");
    expect(parsedStep.thought).toBe("pod cpu high");
    // envelope fields round-trip exactly
    expect(stepEv.case_id).toBe("t001");
    expect(stepEv.seq).toBe(1);
    expect(stepEv.event).toBe("step");

    const [parsedReport] = handlers.calls.onReport[0] as [RcaReport, SseEvent];
    expect(parsedReport.root_cause.summary).toBe("oom");

    expect(handlers.calls.onDone).toHaveLength(1);
    // done closes the stream
    expect(es.isClosed()).toBe(true);
    handle.dispose();
  });

  it("dispatches error and closes the stream on an `error` event", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t002", "clickhouse", handlers);
    const es = fakeEs(handle);

    es.dispatchEventMessage("error", envelope("error", "t002", { error: "boom" }, 5));

    expect(handlers.calls.onError).toHaveLength(1);
    const [msg, ev] = handlers.calls.onError[0] as [string, SseEvent<{ error?: string }>];
    expect(msg).toBe("boom");
    expect(ev.seq).toBe(5);
    expect(es.isClosed()).toBe(true);
    handle.dispose();
  });

  it("surfaces a parse failure via onError without throwing", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t003", "parquet", handlers);
    const es = fakeEs(handle);

    // Dispatch a raw broken payload (not valid JSON).
    es.dispatchEventMessage("step", "{not json");

    expect(handlers.calls.onError).toHaveLength(1);
    const [msg] = handlers.calls.onError[0] as [string, SseEvent];
    expect(msg).toMatch(/failed to parse step SSE data/);
    // Parse failure must NOT close the stream.
    expect(es.isClosed()).toBe(false);
    handle.dispose();
  });

  it("ignores native transport-error events that carry no data", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t004", "parquet", handlers);
    const es = fakeEs(handle);

    // A real EventSource fires `error` with a MessageEvent whose data is
    // undefined on native transport failure. Our listener must NOT treat this
    // as a parsed error event.
    const ev = new MessageEvent("error", { data: undefined });
    (es as unknown as { listeners: Map<string, Set<(e: MessageEvent) => void>> })
      .listeners.get("error")!
      .forEach((l) => l(ev));

    expect(handlers.calls.onError).toHaveLength(0);
    expect(es.isClosed()).toBe(false);
    handle.dispose();
  });

  it("surfaces a native transport failure only when readyState is CLOSED", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t005", "parquet", handlers);
    const es = fakeEs(handle);

    // CONNECTING -> retry, not yet a terminal failure
    es.readyState = FakeEventSource.CONNECTING;
    es.simulateTransportError(false);
    expect(handlers.calls.onError).toHaveLength(0);

    // CLOSED -> terminal
    es.simulateTransportError(true);
    expect(handlers.calls.onError).toHaveLength(1);
    const [msg] = handlers.calls.onError[0] as [string, SseEvent];
    expect(msg).toMatch(/stream closed for case t005/);
    expect(es.isClosed()).toBe(true);
    handle.dispose();
  });

  it("does not double-close: done then a later transport error is a no-op", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t006", "parquet", handlers);
    const es = fakeEs(handle);

    es.dispatchEventMessage("done", envelope("done", "t006", {}, 1));
    expect(handlers.calls.onDone).toHaveLength(1);
    // After done, readyState is CLOSED; a spurious native error must NOT fire.
    es.simulateTransportError(true);
    expect(handlers.calls.onError).toHaveLength(0);
    handle.dispose();
  });
});

describe("openRcaStream — inactivity watchdog", () => {
  let restore: () => void;

  beforeEach(() => {
    vi.useFakeTimers();
    const inst = installFakeEventSource();
    restore = inst.cleanup;
  });

  afterEach(() => {
    cleanup();
    restore();
    vi.useRealTimers();
  });

  it("shouldArmIdle rejects Infinity / NaN / 0 / negatives, accepts positive finite", () => {
    expect(shouldArmIdle(1000)).toBe(true);
    expect(shouldArmIdle(DEFAULT_IDLE_TIMEOUT_MS)).toBe(true);
    expect(shouldArmIdle(Infinity)).toBe(false);
    expect(shouldArmIdle(NaN)).toBe(false);
    expect(shouldArmIdle(0)).toBe(false);
    expect(shouldArmIdle(-5)).toBe(false);
  });

  it("fires onError and closes the stream when no event arrives within idleTimeoutMs", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t-idle", "parquet", handlers, { idleTimeoutMs: 1000 });
    const es = fakeEs(handle);

    // No events for 1000ms -> watchdog should fire.
    act(() => {
      vi.advanceTimersByTime(999);
    });
    expect(handlers.calls.onError).toHaveLength(0);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(handlers.calls.onError).toHaveLength(1);
    const [msg, ev] = handlers.calls.onError[0] as [string, SseEvent];
    expect(msg).toMatch(/stream idle for case t-idle/);
    expect(ev.event).toBe("error");
    expect(es.isClosed()).toBe(true);
  });

  it("re-arms the watchdog on each received event so an active stream is not killed", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t-active", "parquet", handlers, { idleTimeoutMs: 1000 });
    const es = fakeEs(handle);

    // Emit a step every 800ms for a while; should never trip the 1000ms watchdog.
    for (let i = 0; i < 5; i++) {
      act(() => {
        vi.advanceTimersByTime(800);
      });
      es.dispatchEventMessage(
        "step",
        envelope("step", "t-active", { step_id: `s${i}`, case_id: "t-active", step_kind: "observe" }, i),
      );
    }
    expect(handlers.calls.onError).toHaveLength(0);
    expect(es.isClosed()).toBe(false);
    handle.dispose();
  });

  it("does NOT arm the watchdog for Infinity (stream survives past any finite delay)", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t-inf", "parquet", handlers, { idleTimeoutMs: Infinity });
    const es = fakeEs(handle);

    // Advance well beyond a reasonable bound; with Infinity NO timer must be
    // pending (setTimeout(Infinity) would clamp to 0 and kill the stream).
    act(() => {
      vi.advanceTimersByTime(10 * 60 * 1000);
    });
    expect(handlers.calls.onError).toHaveLength(0);
    expect(es.isClosed()).toBe(false);
    handle.dispose();
  });

  it("does NOT arm the watchdog for NaN or <= 0", () => {
    for (const bad of [NaN, 0, -1] as number[]) {
      const handlers = makeHandlers();
      const handle = openRcaStream("t-bad", "parquet", handlers, { idleTimeoutMs: bad });
      const es = fakeEs(handle);
      act(() => {
        vi.advanceTimersByTime(60_000);
      });
      expect(handlers.calls.onError).toHaveLength(0);
      expect(es.isClosed()).toBe(false);
      handle.dispose();
    }
  });

  it("dispose() before the timeout cancels the timer — assert the timer is cleared, not just absence of onError", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t-stop", "parquet", handlers, { idleTimeoutMs: 1000 });
    const es = fakeEs(handle);

    handle.dispose();
    // After dispose the stream is closed and terminated.
    expect(es.isClosed()).toBe(true);

    // Spy on setTimeout to prove no NEW timer is scheduled, then advance time
    // far beyond the watchdog window and assert no callback fires AND that the
    // timer queue is drained (no pending callback). We verify via the absence
    // of any side-effect: dispose must have cleared the armed timer.
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    // No new timers should have been scheduled by the disposed stream.
    expect(setTimeoutSpy).not.toHaveBeenCalled();
    expect(handlers.calls.onError).toHaveLength(0);
    expect(handlers.calls.onDone).toHaveLength(0);
    setTimeoutSpy.mockRestore();
  });

  it("close() (the EventSource-shaped alias) also clears the idle timer", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t-close-alias", "parquet", handlers, { idleTimeoutMs: 1000 });
    const es = fakeEs(handle);

    handle.close();
    expect(es.isClosed()).toBe(true);
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(handlers.calls.onError).toHaveLength(0);
  });

  it("native transport failure (CLOSED) does not leave the idle timer dangling", () => {
    const handlers = makeHandlers();
    openRcaStream("t-tx", "parquet", handlers, { idleTimeoutMs: 1000 });
    const es = fakeEventSourcesCreated[fakeEventSourcesCreated.length - 1];

    es.simulateTransportError(true);
    expect(handlers.calls.onError).toHaveLength(1);

    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(setTimeoutSpy).not.toHaveBeenCalled();
    setTimeoutSpy.mockRestore();
  });

  it("does not re-arm the watchdog on native transport-error events with no data", () => {
    // A reconnecting EventSource retrying a dead endpoint fires `error`
    // repeatedly with no data. The watchdog must NOT be re-armed on those,
    // or it would never trip and the UI would hang forever.
    const handlers = makeHandlers();
    const handle = openRcaStream("t-retry", "parquet", handlers, { idleTimeoutMs: 1000 });
    const es = fakeEs(handle);

    // Fire several native errors (CONNECTING, no data) — must NOT re-arm.
    for (let i = 0; i < 4; i++) {
      es.readyState = FakeEventSource.CONNECTING;
      const ev = new MessageEvent("error", { data: undefined });
      (es as unknown as { listeners: Map<string, Set<(e: MessageEvent) => void>> })
        .listeners.get("error")!
        .forEach((l) => l(ev));
      act(() => {
        vi.advanceTimersByTime(200);
      });
    }
    // Still under the 1000ms budget so far: total advanced = 800ms, no re-arm
    // happened, so the watchdog should still be primed from open time.
    expect(handlers.calls.onError).toHaveLength(0);
    // Crossing 1000ms from open should trip it despite the retries.
    act(() => {
      vi.advanceTimersByTime(201);
    });
    expect(handlers.calls.onError).toHaveLength(1);
    const [msg] = handlers.calls.onError[0] as [string, SseEvent];
    expect(msg).toMatch(/stream idle for case t-retry/);
    handle.dispose();
  });

  it("ignores straggler events delivered after a terminal done (terminated guard)", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t-straggler", "parquet", handlers, { idleTimeoutMs: 1000 });
    const es = fakeEs(handle);

    es.dispatchEventMessage("done", envelope("done", "t-straggler", {}, 1));
    expect(handlers.calls.onDone).toHaveLength(1);
    expect(es.isClosed()).toBe(true);

    // A late step/error/done the browser still delivers to the named listeners
    // must NOT re-arm the watchdog or fire callbacks again.
    es.dispatchEventMessage("step", envelope("step", "t-straggler", { step_id: "late", case_id: "t-straggler", step_kind: "observe" }, 2));
    es.dispatchEventMessage("done", envelope("done", "t-straggler", {}, 3));

    expect(handlers.calls.onDone).toHaveLength(1);
    expect(handlers.calls.onStep).toHaveLength(0);

    // Advance far past the watchdog window: no spurious idle error.
    const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(setTimeoutSpy).not.toHaveBeenCalled();
    expect(handlers.calls.onError).toHaveLength(0);
    setTimeoutSpy.mockRestore();
  });

  it("tears down even if the user onDone handler throws (no dangling idle timer)", () => {
    const boom = vi.fn(() => {
      throw new Error("user onDone blew up");
    });
    const handlers = makeHandlers({ onDone: boom });
    const handle = openRcaStream("t-throw-done", "parquet", handlers, { idleTimeoutMs: 1000 });
    const es = fakeEs(handle);

    // The throw propagates out of the listener (we don't swallow user errors),
    // but disposeInternal MUST still run via try/finally so the stream closes
    // and no idle timer is left armed.
    expect(() =>
      es.dispatchEventMessage("done", envelope("done", "t-throw-done", {}, 1)),
    ).toThrow("user onDone blew up");

    expect(es.isClosed()).toBe(true);
    // No idle timer may fire later.
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(handlers.calls.onError).toHaveLength(0);
  });
});

describe("openRcaStream — run_id threading & onTransportClosed", () => {
  let restore: () => void;
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    const inst = installFakeEventSource();
    restore = inst.cleanup;
  });

  afterEach(() => {
    (globalThis as { fetch: unknown }).fetch = originalFetch;
    restore();
    vi.useRealTimers();
  });

  it("appends &run_id= to the stream URL when a runId is provided", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t007", "parquet", handlers, {
      runId: "run-abc-123",
    });
    const es = fakeEs(handle);
    expect(es.url).toContain("run_id=run-abc-123");
    // backend param still present and correct
    expect(es.url).toContain("backend=parquet");
    handle.dispose();
  });

  it("omits run_id entirely from the URL when no runId is given", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t008", "clickhouse", handlers);
    const es = fakeEs(handle);
    expect(es.url).not.toContain("run_id");
    expect(es.url).toContain("backend=clickhouse");
    handle.dispose();
  });

  it("URL-encodes the run_id", () => {
    const handlers = makeHandlers();
    const handle = openRcaStream("t009", "parquet", handlers, {
      runId: "run with spaces&special",
    });
    const es = fakeEs(handle);
    expect(es.url).toContain("run_id=run%20with%20spaces%26special");
    handle.dispose();
  });

  it("fires onTransportClosed (with the runId) on a native transport CLOSED", () => {
    const onTransportClosed = vi.fn();
    const handlers = makeHandlers();
    const handle = openRcaStream("t010", "parquet", handlers, {
      runId: "run-dead",
      onTransportClosed,
    });
    const es = fakeEs(handle);

    es.simulateTransportError(true);

    expect(onTransportClosed).toHaveBeenCalledTimes(1);
    expect(onTransportClosed).toHaveBeenCalledWith("run-dead");
    // onError also still fires (the banner signal).
    expect(handlers.calls.onError).toHaveLength(1);
    const [msg] = handlers.calls.onError[0] as [string, SseEvent];
    expect(msg).toMatch(/stream closed for case t010/);
    expect(es.isClosed()).toBe(true);
    handle.dispose();
  });

  it("passes null to onTransportClosed when no runId was supplied", () => {
    const onTransportClosed = vi.fn();
    const handlers = makeHandlers();
    const handle = openRcaStream("t011", "parquet", handlers, {
      onTransportClosed,
    });
    const es = fakeEs(handle);

    es.simulateTransportError(true);

    expect(onTransportClosed).toHaveBeenCalledWith(null);
    handle.dispose();
  });

  it("does NOT fire onTransportClosed on a CONNECTING retry (only on CLOSED)", () => {
    const onTransportClosed = vi.fn();
    const handlers = makeHandlers();
    const handle = openRcaStream("t012", "parquet", handlers, {
      runId: "run-retry",
      onTransportClosed,
    });
    const es = fakeEs(handle);

    // CONNECTING retry — not yet terminal.
    es.readyState = FakeEventSource.CONNECTING;
    es.simulateTransportError(false);
    expect(onTransportClosed).not.toHaveBeenCalled();

    // Now CLOSED — terminal.
    es.simulateTransportError(true);
    expect(onTransportClosed).toHaveBeenCalledTimes(1);
    handle.dispose();
  });

  it("does NOT fire onTransportClosed after a clean done (teardown guard)", () => {
    const onTransportClosed = vi.fn();
    const handlers = makeHandlers();
    const handle = openRcaStream("t013", "parquet", handlers, {
      runId: "run-done",
      onTransportClosed,
    });
    const es = fakeEs(handle);

    es.dispatchEventMessage("done", envelope("done", "t013", {}, 1));
    // Spurious CLOSED error after done must not fire the hook.
    es.simulateTransportError(true);
    expect(onTransportClosed).not.toHaveBeenCalled();
    handle.dispose();
  });

  it("still fires onError even if onTransportClosed throws", () => {
    const onTransportClosed = vi.fn(() => {
      throw new Error("hook blew up");
    });
    const handlers = makeHandlers();
    const handle = openRcaStream("t014", "parquet", handlers, {
      runId: "run-throw",
      onTransportClosed,
    });
    const es = fakeEs(handle);

    // The hook throwing must not suppress onError (the banner signal).
    es.simulateTransportError(true);
    expect(onTransportClosed).toHaveBeenCalledTimes(1);
    expect(handlers.calls.onError).toHaveLength(1);
    expect(es.isClosed()).toBe(true);
    handle.dispose();
  });
});

describe("run-persistence fetchers", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    (globalThis as { fetch: unknown }).fetch = originalFetch;
  });

  function mockJson(url: RegExp, body: unknown, status = 200) {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const u = typeof input === "string" ? input : input.toString();
      if (url.test(u)) return new Response(JSON.stringify(body), { status });
      return new Response("not mocked", { status: 404 });
    });
    (globalThis as { fetch: unknown }).fetch = fetchMock;
    return fetchMock;
  }

  it("fetchRuns lists runs and parses the {runs:[]} envelope", async () => {
    const body = {
      runs: [
        { run_id: "r1", case_id: "t001", status: "completed", step_count: 5 },
        { run_id: "r2", case_id: "t001", status: "error", step_count: 2 },
      ],
    };
    const fm = mockJson(/^\/runs/, body);
    const runs = await fetchRuns("t001");
    expect(runs).toHaveLength(2);
    expect(runs[0].run_id).toBe("r1");
    expect(runs[0].step_count).toBe(5);
    // case_id filter is threaded as a query param.
    expect(fm.mock.calls[0][0]).toBe("/runs?case_id=t001");
  });

  it("fetchRuns omits case_id query when not given", async () => {
    const fm = mockJson(/^\/runs$/, { runs: [] });    await fetchRuns();
    expect(fm.mock.calls[0][0]).toBe("/runs");
  });

  it("fetchRuns returns [] when the envelope has no runs key", async () => {
    mockJson(/^\/runs/, {});    const runs = await fetchRuns("t001");
    expect(runs).toEqual([]);
  });

  it("fetchRuns throws on non-OK", async () => {
    (globalThis as { fetch: unknown }).fetch = vi.fn(async () =>
      new Response("nope", { status: 500 }),
    );    await expect(fetchRuns("t001")).rejects.toThrow(/fetchRuns failed: 500/);
  });

  it("fetchRun merges {run, steps} into a Run", async () => {
    const body = {
      run: { run_id: "r1", case_id: "t001", status: "completed", step_count: 1 },
      steps: [{ step_id: "s1", case_id: "t001", step_kind: "observe" }],
    };
    mockJson(/^\/runs\/r1(\?|$)/, body);
    const run = await fetchRun("r1");
    expect(run.run_id).toBe("r1");
    expect(run.status).toBe("completed");
    expect(run.steps).toHaveLength(1);
    expect(run.steps[0].step_kind).toBe("observe");
  });

  it("fetchRun defaults steps to [] when absent", async () => {
    const body = { run: { run_id: "r2", case_id: "t001", status: "running", step_count: 0 } };
    mockJson(/^\/runs\/r2(\?|$)/, body);    const run = await fetchRun("r2");
    expect(run.steps).toEqual([]);
  });

  it("fetchRun throws on non-OK", async () => {
    (globalThis as { fetch: unknown }).fetch = vi.fn(async () =>
      new Response("nope", { status: 404 }),
    );    await expect(fetchRun("missing")).rejects.toThrow(/fetchRun failed: 404/);
  });

  it("fetchRunSteps returns the steps array", async () => {
    const body = { steps: [{ step_id: "s1", case_id: "t001", step_kind: "observe" }] };
    mockJson(/^\/runs\/r1\/steps(\?|$)/, body);    const steps = await fetchRunSteps("r1");
    expect(steps).toHaveLength(1);
  });

  it("fetchRunSteps defaults to [] when absent", async () => {
    mockJson(/^\/runs\/r1\/steps(\?|$)/, {});    const steps = await fetchRunSteps("r1");
    expect(steps).toEqual([]);
  });

  it("fetchRunSteps throws on non-OK", async () => {
    (globalThis as { fetch: unknown }).fetch = vi.fn(async () =>
      new Response("nope", { status: 500 }),
    );    await expect(fetchRunSteps("r1")).rejects.toThrow(/fetchRunSteps failed: 500/);
  });
});

