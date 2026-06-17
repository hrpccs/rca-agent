import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CasePicker } from "../components/CasePicker";

describe("CasePicker", () => {
  it("lists all cases when there is no query", () => {
    const onSelect = vi.fn();
    render(
      <CasePicker cases={["t001", "t002", "case-foo"]} selected={null} onSelect={onSelect} />,
    );
    expect(screen.getByText("t001")).toBeInTheDocument();
    expect(screen.getByText("t002")).toBeInTheDocument();
    expect(screen.getByText("case-foo")).toBeInTheDocument();
    // count shows filtered/total
    expect(screen.getByText("3 / 3")).toBeInTheDocument();
  });

  it("filters cases by the search query (case-insensitive)", () => {
    render(<CasePicker cases={["t001", "t002", "case-foo"]} selected={null} onSelect={vi.fn()} />);
    const input = screen.getByLabelText(/Search cases/i);
    fireEvent.change(input, { target: { value: "T00" } });
    expect(screen.getByText("t001")).toBeInTheDocument();
    expect(screen.getByText("t002")).toBeInTheDocument();
    expect(screen.queryByText("case-foo")).not.toBeInTheDocument();
    expect(screen.getByText("2 / 3")).toBeInTheDocument();
  });

  it("shows the empty state when the query matches nothing", () => {
    render(<CasePicker cases={["t001", "t002"]} selected={null} onSelect={vi.fn()} />);
    fireEvent.change(screen.getByLabelText(/Search cases/i), { target: { value: "zzz" } });
    expect(screen.getByText(/No matching cases/i)).toBeInTheDocument();
  });

  it("shows a loading indicator and no empty state while loading", () => {
    render(<CasePicker cases={[]} selected={null} onSelect={vi.fn()} loading={true} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
    expect(screen.queryByText(/No matching cases/i)).not.toBeInTheDocument();
  });

  it("renders an error message when provided", () => {
    render(
      <CasePicker cases={[]} selected={null} onSelect={vi.fn()} error="network down" />,
    );
    expect(screen.getByText(/Failed to load cases: network down/i)).toBeInTheDocument();
  });

  it("calls onSelect with the case id and highlights the selected case", () => {
    const onSelect = vi.fn();
    render(<CasePicker cases={["t001", "t002"]} selected="t002" onSelect={onSelect} />);
    const t002Btn = screen.getByText("t002").closest("button")!;
    expect(t002Btn).toHaveClass("case-picker__item--active");
    fireEvent.click(t002Btn);
    expect(onSelect).toHaveBeenCalledWith("t002");
  });
});
