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
