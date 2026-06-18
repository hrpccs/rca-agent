import { act, render, screen, waitFor, fireEvent } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { RcaReport, RcaStep } from "../types";
import { installFakeEventSource, fakeEventSourcesCreated } from "./fakeEventSource";
import { sseEnvelope as envelope } from "./sseEnvelope";

describe("App — SSE integration", () => {
  let restoreEventSource: () => void;
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    const inst = installFakeEventSource();
    restoreEventSource = inst.cleanup;
  });

  afterEach(() => {
    (globalThis as { fetch: unknown }).fetch = originalFetch;
    restoreEventSource();
    fakeEventSourcesCreated.length = 0;
    vi.useRealTimers();
  });

  function mockFetchCases(cases: string[]) {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.startsWith("/cases")) {
        return new Response(JSON.stringify({ cases }), { status: 200 });
      }
      if (url.includes("/rca/") && url.includes("/stream")) {
        // openRcaStream uses EventSource, not fetch; never reached.
        return new Response("", { status: 404 });
      }
      // POST /rca/{case_id}
      return new Response(
        JSON.stringify({ case_id: "t001", backend: "parquet", stream_url: "/rca/t001/stream" }),
        { status: 200 },
      );
    });
    (globalThis as { fetch: unknown }).fetch = fetchMock;
    return fetchMock;
  }

  it("loads the case list, runs an RCA via SSE, and renders the report", async () => {
    mockFetchCases(["t001", "t002"]);

    render(<App />);

    // Wait for the case list.
    await waitFor(() => {
      expect(screen.getByText("t001")).toBeInTheDocument();
    });

    // Select a case.
    fireEvent.click(screen.getByText("t001"));

    // Run.
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));

    // The stream should have been opened.
    await waitFor(() => {
      expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1);
    });
    const es = fakeEventSourcesCreated[fakeEventSourcesCreated.length - 1];
    es.simulateOpen();

    // Emit a step (reasoning renders its thought text, so we can assert it).
    const step: RcaStep = {
      step_id: "s1",
      case_id: "t001",
      step_kind: "reasoning",
      thought: "pod restarting",
    };
    act(() => {
      es.dispatchEventMessage("step", envelope("step", "t001", step, 1));
    });

    // The run status should have advanced to running and the step rendered.
    await waitFor(() => {
      expect(screen.getByText(/running/i)).toBeInTheDocument();
    });
    expect(screen.getByText("pod restarting")).toBeInTheDocument();

    // Emit the report + done.
    const report: RcaReport = {
      case_id: "t001",
      task_id: "task-1",
      alert_title: "Pod CPU 使用率过高",
      status: "completed",
      root_cause: { summary: "OOM kill due to low memory limit", confidence: 0.9 },
    };
    act(() => {
      es.dispatchEventMessage("report", envelope("report", "t001", report, 2));
      es.dispatchEventMessage("done", envelope("done", "t001", { ok: true }, 3));
    });

    await waitFor(() => {
      expect(screen.getByText(/Root Cause Report/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/OOM kill due to low memory limit/i)).toBeInTheDocument();
    expect(screen.getByText(/done/i)).toBeInTheDocument();
  });

  it("Stop closes the stream and does not fire a spurious error after the idle window", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    mockFetchCases(["t001"]);

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText("t001")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("t001"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));

    await waitFor(() => {
      expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1);
    });
    const es = fakeEventSourcesCreated[fakeEventSourcesCreated.length - 1];

    // Stop before any event arrives.
    fireEvent.click(screen.getByRole("button", { name: /Stop/i }));
    expect(es.isClosed()).toBe(true);

    // Advance well past the idle window; no error should surface.
    act(() => {
      vi.advanceTimersByTime(70_000);
    });
    expect(screen.queryByText(/stream idle/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/stream closed/i)).not.toBeInTheDocument();
  });

  it("shows a cases-load error when fetchCases fails", async () => {
    (globalThis as { fetch: unknown }).fetch = vi.fn(async () =>
      new Response("nope", { status: 500 }),
    );

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Failed to load cases/i)).toBeInTheDocument();
    });
  });
});

/**
 * Integration coverage for the T4 per-case trace cache, run history loading,
 * and disconnect-recovery paths. These mock both fetch (for the REST surface:
 * /cases, /rca POST, /runs) and EventSource (for the SSE surface) and drive
 * the real App component.
 */
describe("App — per-case trace cache & disconnect recovery", () => {
  let restoreEventSource: () => void;
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    const inst = installFakeEventSource();
    restoreEventSource = inst.cleanup;
  });

  afterEach(() => {
    (globalThis as { fetch: unknown }).fetch = originalFetch;
    restoreEventSource();
    fakeEventSourcesCreated.length = 0;
    vi.useRealTimers();
  });

  /**
   * A fetch mock that:
   *  - serves /cases
   *  - POST /rca/{case_id} returns a run_id (so disconnect recovery is wired)
   *  - GET /runs returns an empty list by default
   * The caller can override the run_id per-case or stub /runs responses.
   */
  function mockFetchAll(opts: {
    cases: string[];
    runIdFor?: (caseId: string) => string | null;
    runsFor?: (caseId: string) => unknown;
    runDetail?: (runId: string) => unknown;
  }) {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = init?.method ?? "GET";
      if (url.startsWith("/cases")) {
        return new Response(JSON.stringify({ cases: opts.cases }), { status: 200 });
      }
      if (url.includes("/rca/") && method === "POST") {
        const caseId = url.split("/rca/")[1]?.split("?")[0] ?? "x";
        const runId = opts.runIdFor?.(caseId) ?? null;
        return new Response(
          JSON.stringify({
            case_id: caseId,
            backend: "parquet",
            stream_url: `/rca/${caseId}/stream`,
            ...(runId ? { run_id: runId } : {}),
          }),
          { status: 200 },
        );
      }
      if (url.startsWith("/runs/") && url.includes("/steps")) {
        const runId = url.split("/runs/")[1]?.split("/")[0] ?? "x";
        const detail = opts.runDetail?.(runId);
        return new Response(
          JSON.stringify({ steps: (detail as { steps?: unknown[] })?.steps ?? [] }),
          { status: 200 },
        );
      }
      if (url.startsWith("/runs/")) {
        const runId = url.split("/runs/")[1]?.split("?")[0] ?? "x";
        const detail =
          opts.runDetail?.(runId) ?? {
            run: { run_id: runId, case_id: "x", status: "completed", step_count: 0 },
            steps: [],
          };
        return new Response(JSON.stringify(detail), { status: 200 });
      }
      if (url.startsWith("/runs")) {
        const m = url.match(/case_id=([^&]+)/);
        const caseId = m ? decodeURIComponent(m[1]) : "";
        const body = opts.runsFor?.(caseId) ?? { runs: [] };
        return new Response(JSON.stringify(body), { status: 200 });
      }
      return new Response("", { status: 404 });
    });
    (globalThis as { fetch: unknown }).fetch = fetchMock;
    return fetchMock;
  }

  function latestEs() {
    return fakeEventSourcesCreated[fakeEventSourcesCreated.length - 1];
  }

  it("switching cases preserves the previous case's trace (cache, not wipe)", async () => {
    mockFetchAll({ cases: ["t004", "t010"] });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t004")).toBeInTheDocument());

    // Select t004 and run it.
    fireEvent.click(screen.getByText("t004"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));
    await waitFor(() => expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1));
    latestEs().simulateOpen();

    // Emit a step whose text is unique to t004.
    act(() => {
      latestEs().dispatchEventMessage(
        "step",
        envelope("step", "t004", {
          step_id: "s-t004",
          case_id: "t004",
          step_kind: "reasoning",
          thought: "t004-unique-trace-marker",
        }, 1),
      );
    });
    await waitFor(() =>
      expect(screen.getByText("t004-unique-trace-marker")).toBeInTheDocument(),
    );

    // Switch to t010 (would have wiped the trace under the old resetRun model).
    fireEvent.click(screen.getByText("t010"));
    // t010 has no trace yet — empty state shown.
    await waitFor(() =>
      expect(screen.getByText(/Select a case and run RCA/i)).toBeInTheDocument(),
    );

    // Switch BACK to t004 — its trace must still be there.
    fireEvent.click(screen.getByText("t004"));
    await waitFor(() =>
      expect(screen.getByText("t004-unique-trace-marker")).toBeInTheDocument(),
    );
  });

  it("loads run history (fetchRuns) when a case is selected", async () => {
    const fm = mockFetchAll({
      cases: ["t020"],
      runsFor: (caseId) =>
        caseId === "t020"
          ? { runs: [{ run_id: "r-old", case_id: "t020", status: "completed", step_count: 4 }] }
          : { runs: [] },
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t020")).toBeInTheDocument());

    fireEvent.click(screen.getByText("t020"));

    // fetchRuns should have been called with ?case_id=t020.
    await waitFor(() => {
      const calledWithCase = fm.mock.calls.some(([u]) => {
        const url = u as string;
        return url.startsWith("/runs") && url.includes("case_id=t020");
      });
      expect(calledWithCase).toBe(true);
    });
    // The persisted run appears in the history panel.
    await waitFor(() => expect(screen.getByText("r-old")).toBeInTheDocument());
  });

  it("on a transport-CLOSED mid-run, fetches the persisted trace and shows the recovery banner", async () => {
    mockFetchAll({
      cases: ["t030"],
      runIdFor: () => "run-disconnect",
      runDetail: () => ({
        run: {
          run_id: "run-disconnect",
          case_id: "t030",
          status: "completed",
          step_count: 1,
        },
        steps: [
          {
            step_id: "s-persisted",
            case_id: "t030",
            step_kind: "reasoning",
            thought: "persisted-trace-after-drop",
          },
        ],
      }),
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t030")).toBeInTheDocument());

    fireEvent.click(screen.getByText("t030"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));
    await waitFor(() => expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1));
    latestEs().simulateOpen();

    // Simulate the transport dropping mid-run (CLOSED).
    act(() => {
      latestEs().simulateTransportError(true);
    });

    // App should best-effort fetch the persisted trace and show it + the banner.
    await waitFor(() =>
      expect(screen.getByText(/connection lost — loaded server-saved trace/i)).toBeInTheDocument(),
    );
    await waitFor(() =>
      expect(screen.getByText("persisted-trace-after-drop")).toBeInTheDocument(),
    );
  });

  it("on disconnect with no known runId, shows a partial-trace banner (best-effort)", async () => {
    // No run_id returned from POST /rca -> recovery cannot fetch a persisted trace.
    mockFetchAll({
      cases: ["t031"],
      runIdFor: () => null,
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t031")).toBeInTheDocument());

    fireEvent.click(screen.getByText("t031"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));
    await waitFor(() => expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1));
    latestEs().simulateOpen();

    // Emit one step before the drop so we have a partial trace.
    act(() => {
      latestEs().dispatchEventMessage(
        "step",
        envelope("step", "t031", {
          step_id: "s-partial",
          case_id: "t031",
          step_kind: "reasoning",
          thought: "partial-before-drop",
        }, 1),
      );
    });
    await waitFor(() =>
      expect(screen.getByText("partial-before-drop")).toBeInTheDocument(),
    );

    // Drop the transport with no runId known.
    act(() => {
      latestEs().simulateTransportError(true);
    });

    // Partial-trace banner appears (no persisted fetch since runId is null).
    await waitFor(() =>
      expect(screen.getByText(/connection lost — partial trace shown/i)).toBeInTheDocument(),
    );
    // The partial step is still on screen.
    expect(screen.getByText("partial-before-drop")).toBeInTheDocument();
  });

  it("disconnect recovery writes into the ORIGINAL case, not a case switched to after the drop", async () => {
    // Regression guard: onTransportClosed's fetchRun resolves asynchronously.
    // If the user switches cases between the transport drop and the fetch
    // resolving, the recovered trace must still land in the dropped run's case
    // (captured at stream-open time), not the now-selected case. Under the old
    // updateCurrent model the recovery would clobber the wrong case's entry.
    mockFetchAll({
      cases: ["t040", "t041"],
      runIdFor: () => "run-t040",
      runDetail: () => ({
        run: { run_id: "run-t040", case_id: "t040", status: "completed", step_count: 1 },
        steps: [
          { step_id: "s-t040-recovered", case_id: "t040", step_kind: "reasoning", thought: "t040-recovered-marker" },
        ],
      }),
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t040")).toBeInTheDocument());
    fireEvent.click(screen.getByText("t040"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));
    await waitFor(() => expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1));
    latestEs().simulateOpen();

    // Drop the transport (kicks off fetchRun("run-t040")).
    act(() => {
      latestEs().simulateTransportError(true);
    });

    // Switch to t041 WHILE the recovery fetch is in flight (resolve ordering is
    // not deterministic, so we just switch immediately; the fetch is async).
    const t041InPicker = screen.getAllByText("t041").find((el) =>
      el.closest(".case-picker"),
    );
    fireEvent.click(t041InPicker!);

    // t041 must NOT receive t040's recovery (no banner, no recovered step).
    // Wait long enough for the recovery fetch to have resolved.
    await waitFor(
      () => {
        // The recovery banner would appear if it landed in the displayed case.
        expect(screen.queryByText(/connection lost/i)).not.toBeInTheDocument();
      },
      { timeout: 1500 },
    );

    // Switch back to t040 — the recovery MUST have landed there.
    const t040InPicker = screen.getAllByText("t040").find((el) =>
      el.closest(".case-picker"),
    );
    fireEvent.click(t040InPicker!);
    await waitFor(() =>
      expect(screen.getByText("t040-recovered-marker")).toBeInTheDocument(),
    );
    expect(screen.getByText(/connection lost — loaded server-saved trace/i)).toBeInTheDocument();
  });

  it("on an idle-drop with a known runId, fetches the persisted trace and shows the recovery banner (F2)", async () => {
    // Regression test for bug #2: the IDLE-watchdog drop path (the most common
    // drop during long DeepSeek turns) must trigger persisted-trace recovery,
    // not just the transport-CLOSED path. Under the old code only
    // onTransportClosed fired, so an idle drop showed an empty timeline
    // instead of the server-saved trace. The default idle timeout is 60s; we
    // drive it by advancing fake timers past the window.
    vi.useFakeTimers({ shouldAdvanceTime: true });
    mockFetchAll({
      cases: ["t050"],
      runIdFor: () => "run-idle-drop",
      runDetail: () => ({
        run: {
          run_id: "run-idle-drop",
          case_id: "t050",
          status: "interrupted",
          step_count: 1,
        },
        steps: [
          {
            step_id: "s-idle-persisted",
            case_id: "t050",
            step_kind: "reasoning",
            thought: "idle-drop-persisted-trace",
          },
        ],
      }),
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t050")).toBeInTheDocument());
    fireEvent.click(screen.getByText("t050"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));
    await waitFor(() => expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1));
    latestEs().simulateOpen();

    // Advance past the 60s idle window with NO events delivered -> the
    // watchdog fires, which must kick off fetchRun("run-idle-drop").
    act(() => {
      vi.advanceTimersByTime(60_500);
    });

    // The persisted trace + bilingual banner must appear (not an empty timeline).
    await waitFor(() =>
      expect(screen.getByText(/connection lost — loaded server-saved trace/i)).toBeInTheDocument(),
    );
    await waitFor(() =>
      expect(screen.getByText("idle-drop-persisted-trace")).toBeInTheDocument(),
    );
    // The banner reports the persisted step count (scoped to the banner text
    // so it doesn't collide with the RunHistory row that also shows "1 step").
    expect(screen.getByText(/server-saved trace \(1 step\)/i)).toBeInTheDocument();
  });

  it("flips status to running as soon as POST /rca resolves (not only on first step)", async () => {
    // Regression guard: the old single-state model called setStatus("running")
    // right after startRca. The per-case cache must preserve that so the
    // elapsed ticker / live indicator start before the first SSE step.
    mockFetchAll({ cases: ["t042"] });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t042")).toBeInTheDocument());
    fireEvent.click(screen.getByText("t042"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));

    // After POST resolves but before any step, status should be "running".
    await waitFor(() => {
      expect(screen.getByText(/运行中|running/i)).toBeInTheDocument();
    });
  });

  it("disconnect recovery does not clobber a replay started after the drop", async () => {
    // Regression guard: the fetchRun recovery promise resolves asynchronously;
    // if the user clicks a historical run in between, the recovery must write
    // into the live shadow, not overwrite the displayed replay.
    mockFetchAll({
      cases: ["t043"],
      runIdFor: () => "run-drop",
      runsFor: () => ({
        runs: [
          { run_id: "r-replay", case_id: "t043", status: "completed", step_count: 1 },
        ],
      }),
      runDetail: (rid) => {
        if (rid === "r-replay") {
          return {
            run: { run_id: "r-replay", case_id: "t043", status: "completed", step_count: 1 },
            steps: [
              { step_id: "s-replay", case_id: "t043", step_kind: "reasoning", thought: "replay-marker-step" },
            ],
          };
        }
        // The dropped live run's persisted trace:
        return {
          run: { run_id: "run-drop", case_id: "t043", status: "completed", step_count: 1 },
          steps: [
            { step_id: "s-live-drop", case_id: "t043", step_kind: "reasoning", thought: "live-drop-marker-step" },
          ],
        };
      },
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t043")).toBeInTheDocument());
    fireEvent.click(screen.getByText("t043"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));
    await waitFor(() => expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1));
    latestEs().simulateOpen();

    // Drop the transport; recovery fetch for run-drop is kicked off.
    act(() => {
      latestEs().simulateTransportError(true);
    });

    // Wait for run history to load so the replay row is clickable.
    await waitFor(() => expect(screen.getByText("r-replay")).toBeInTheDocument());

    // Click replay (its fetch may race with the still-pending drop recovery).
    fireEvent.click(screen.getByText("r-replay"));

    // The replayed step should ultimately be shown (recovery must not clobber).
    await waitFor(() =>
      expect(screen.getByText("replay-marker-step")).toBeInTheDocument(),
    );
    // And the live-drop marker (the recovery target) must NOT have overwritten
    // the displayed replay.
    expect(screen.queryByText("live-drop-marker-step")).not.toBeInTheDocument();
  });

  it("Back to live restores the live trace after replaying a historical run", async () => {
    // Regression guard: replay snapshots the live trace into a shadow; back to
    // live must restore it (not leave the replayed steps shown as if live).
    mockFetchAll({
      cases: ["t044"],
      runsFor: () => ({
        runs: [
          { run_id: "r-old", case_id: "t044", status: "completed", step_count: 1 },
        ],
      }),
      runDetail: () => ({
        run: { run_id: "r-old", case_id: "t044", status: "completed", step_count: 1 },
        steps: [
          { step_id: "s-replay", case_id: "t044", step_kind: "reasoning", thought: "replay-only-step" },
        ],
      }),
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("t044")).toBeInTheDocument());
    fireEvent.click(screen.getByText("t044"));
    fireEvent.click(screen.getByRole("button", { name: /Run RCA/i }));
    await waitFor(() => expect(fakeEventSourcesCreated.length).toBeGreaterThanOrEqual(1));
    latestEs().simulateOpen();

    // Emit a LIVE step first.
    act(() => {
      latestEs().dispatchEventMessage(
        "step",
        envelope("step", "t044", {
          step_id: "s-live",
          case_id: "t044",
          step_kind: "reasoning",
          thought: "live-step-must-survive",
        }, 1),
      );
    });
    await waitFor(() => expect(screen.getByText("live-step-must-survive")).toBeInTheDocument());

    // Replay a historical run — its step replaces the displayed trace.
    await waitFor(() => expect(screen.getByText("r-old")).toBeInTheDocument());
    fireEvent.click(screen.getByText("r-old"));
    await waitFor(() => expect(screen.getByText("replay-only-step")).toBeInTheDocument());
    // The live step is hidden during replay.
    expect(screen.queryByText("live-step-must-survive")).not.toBeInTheDocument();

    // Back to live — the live step must be restored.
    fireEvent.click(screen.getByRole("button", { name: /Back to live/i }));
    await waitFor(() =>
      expect(screen.getByText("live-step-must-survive")).toBeInTheDocument(),
    );
    expect(screen.queryByText("replay-only-step")).not.toBeInTheDocument();
  });
});
