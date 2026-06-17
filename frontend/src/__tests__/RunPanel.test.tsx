import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunPanel } from "../components/RunPanel";

const baseProps = {
  caseId: "t001" as string | null,
  backend: "parquet" as const,
  onBackendChange: vi.fn(),
  status: "idle" as const,
  onRun: vi.fn(),
  onStop: vi.fn(),
  stepCount: 0,
  elapsedMs: 0,
};

describe("RunPanel", () => {
  it("disables Run when no case is selected", () => {
    render(<RunPanel {...baseProps} caseId={null} />);
    const runBtn = screen.getByRole("button", { name: /Run RCA/i });
    expect(runBtn).toBeDisabled();
  });

  it("enables Run and calls onRun when a case is selected", () => {
    const onRun = vi.fn();
    render(<RunPanel {...baseProps} onRun={onRun} />);
    const runBtn = screen.getByRole("button", { name: /Run RCA/i });
    expect(runBtn).not.toBeDisabled();
    fireEvent.click(runBtn);
    expect(onRun).toHaveBeenCalledTimes(1);
  });

  it("switches to a Stop button while running and disables backend toggles", () => {
    const onStop = vi.fn();
    const onBackendChange = vi.fn();
    render(
      <RunPanel
        {...baseProps}
        status="running"
        onStop={onStop}
        onBackendChange={onBackendChange}
      />,
    );
    expect(screen.queryByRole("button", { name: /Run RCA/i })).not.toBeInTheDocument();
    const stopBtn = screen.getByRole("button", { name: /Stop/i });
    fireEvent.click(stopBtn);
    expect(onStop).toHaveBeenCalledTimes(1);

    const parquetBtn = screen.getByRole("button", { name: "parquet" });
    expect(parquetBtn).toBeDisabled();
  });

  it("changes the backend via the toggle when not running", () => {
    const onBackendChange = vi.fn();
    render(<RunPanel {...baseProps} backend="parquet" onBackendChange={onBackendChange} />);
    fireEvent.click(screen.getByRole("button", { name: "clickhouse" }));
    expect(onBackendChange).toHaveBeenCalledWith("clickhouse");
  });

  it("renders step count and elapsed time", () => {
    render(<RunPanel {...baseProps} stepCount={7} elapsedMs={4500} />);
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.getByText("4.5 s")).toBeInTheDocument();
  });
});
