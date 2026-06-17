import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TraceTimeline } from "../components/TraceTimeline";
import type { RcaStep } from "../types";

function step(over: Partial<RcaStep>): RcaStep {
  return {
    step_id: over.step_id ?? "x",
    case_id: "t001",
    step_kind: "observe",
    ...over,
  };
}

describe("TraceTimeline", () => {
  it("renders the empty state when there are no steps and not running", () => {
    render(<TraceTimeline steps={[]} running={false} />);
    expect(
      screen.getByText(/Select a case and run RCA to see the live trace/i),
    ).toBeInTheDocument();
  });

  it("renders the waiting state when running with no steps", () => {
    render(<TraceTimeline steps={[]} running={true} />);
    expect(screen.getByText(/Waiting for the first step/i)).toBeInTheDocument();
    expect(screen.getByText(/live/i)).toBeInTheDocument();
  });

  it("renders a standalone reasoning step and its thought", () => {
    const steps = [
      step({ step_id: "r1", step_kind: "reasoning", thought: "analyzing the pod" }),
    ];
    render(<TraceTimeline steps={steps} running={false} />);
    expect(screen.getByText("analyzing the pod")).toBeInTheDocument();
    expect(screen.getByText(/Reasoning/i)).toBeInTheDocument();
  });

  it("groups a tool_call with its immediately-following tool_result", () => {
    const steps: RcaStep[] = [
      step({
        step_id: "tc1",
        step_kind: "tool_call",
        tool_name: "query_metrics",
        tool_args: { metric: "cpu" },
      }),
      step({
        step_id: "tr1",
        step_kind: "tool_result",
        tool_name: "query_metrics",
        tool_result_text: "cpu=99% for 10m",
      }),
    ];
    render(<TraceTimeline steps={steps} running={false} />);
    // tool name rendered once for the paired group
    expect(screen.getByText("query_metrics")).toBeInTheDocument();
    // evidence block rendered from tool_result_text
    expect(screen.getByText("cpu=99% for 10m")).toBeInTheDocument();
    // args block label is present (the args JSON is rendered in a <code>)
    expect(screen.getByText(/args:/i)).toBeInTheDocument();
    expect(screen.getByText(/"metric":"cpu"/)).toBeInTheDocument();
  });

  it("renders a tool_call as unpaired when the next step is not a tool_result", () => {
    const steps: RcaStep[] = [
      step({ step_id: "tc2", step_kind: "tool_call", tool_name: "fetch_logs" }),
      // an intervening reasoning step should NOT be skipped
      step({ step_id: "r2", step_kind: "reasoning", thought: "between" }),
    ];
    render(<TraceTimeline steps={steps} running={false} />);
    expect(screen.getByText("fetch_logs")).toBeInTheDocument();
    expect(screen.getByText("between")).toBeInTheDocument();
    // no evidence block (unpaired)
    expect(screen.queryByText(/evidence:/i)).not.toBeInTheDocument();
  });

  it("renders hypothesis with confidence", () => {
    const steps = [
      step({
        step_id: "h1",
        step_kind: "hypothesize",
        hypothesis: "OOM kill due to low limit",
        confidence: 0.73,
      }),
    ];
    render(<TraceTimeline steps={steps} running={false} />);
    expect(screen.getByText("OOM kill due to low limit")).toBeInTheDocument();
    expect(screen.getByText("73%")).toBeInTheDocument();
  });

  it("renders error step text", () => {
    const steps = [step({ step_id: "e1", step_kind: "error", thought: "upstream 500" })];
    render(<TraceTimeline steps={steps} running={false} />);
    expect(screen.getByText("upstream 500")).toBeInTheDocument();
  });

  it("shows the step count in the header", () => {
    const steps = [
      step({ step_id: "a", step_kind: "observe" }),
      step({ step_id: "b", step_kind: "observe" }),
    ];
    render(<TraceTimeline steps={steps} running={false} />);
    expect(screen.getByText(/2 steps/i)).toBeInTheDocument();
  });
});
