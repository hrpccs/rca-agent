import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunHistory } from "../components/RunHistory";
import type { RunSummary } from "../types";

function run(over: Partial<RunSummary> = {}): RunSummary {
  return {
    run_id: over.run_id ?? "r1",
    case_id: over.case_id ?? "t001",
    status: over.status ?? "completed",
    step_count: over.step_count ?? 3,
    started_at: over.started_at ?? "2026-06-17T10:00:00Z",
    ...over,
  };
}

describe("RunHistory", () => {
  it("renders an empty state when there are no runs and not loading", () => {
    render(<RunHistory runs={[]} onSelectRun={() => {}} />);
    expect(screen.getByText(/No past runs yet/i)).toBeInTheDocument();
  });

  it("renders a loading indicator while loading", () => {
    render(<RunHistory runs={[]} onSelectRun={() => {}} loading />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
    // empty-state copy is hidden while loading
    expect(screen.queryByText(/No past runs yet/i)).not.toBeInTheDocument();
  });

  it("renders each run's status, step count, and started time", () => {
    const runs = [
      run({ run_id: "r1", status: "completed", step_count: 5 }),
      run({ run_id: "r2", status: "error", step_count: 1 }),
    ];
    render(<RunHistory runs={runs} onSelectRun={() => {}} />);
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.getByText("error")).toBeInTheDocument();
    expect(screen.getByText(/5 steps/i)).toBeInTheDocument();
    expect(screen.getByText(/1 step/i)).toBeInTheDocument();
    // count badge shows the total number of runs
    const count = screen.getByText("2");
    expect(count).toBeInTheDocument();
  });

  it("calls onSelectRun with the run_id when a row is clicked", () => {
    const onSelect = vi.fn();
    const runs = [run({ run_id: "r-click" })];
    render(<RunHistory runs={runs} onSelectRun={onSelect} />);
    fireEvent.click(screen.getByText("completed"));
    expect(onSelect).toHaveBeenCalledWith("r-click");
  });

  it("marks the selected run as active with the replay badge", () => {
    const runs = [
      run({ run_id: "r-active" }),
      run({ run_id: "r-other" }),
    ];
    render(
      <RunHistory runs={runs} selectedRunId="r-active" onSelectRun={() => {}} />,
    );
    // replay badge only on the active row
    expect(screen.getByText(/replay/i)).toBeInTheDocument();
    // Exactly one button is marked pressed (the active one).
    const pressed = screen.getAllByRole("button", { pressed: true });
    expect(pressed).toHaveLength(1);
  });

  it("does not show the replay badge when no run is selected", () => {
    const runs = [run({ run_id: "r-none" })];
    render(<RunHistory runs={runs} selectedRunId={null} onSelectRun={() => {}} />);
    expect(screen.queryByText(/replay/i)).not.toBeInTheDocument();
  });

  it("renders — for a missing started_at", () => {
    const runs = [run({ run_id: "r-notime", started_at: null })];
    render(<RunHistory runs={runs} onSelectRun={() => {}} />);
    // The time slot shows — when started_at is absent.
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });
});
