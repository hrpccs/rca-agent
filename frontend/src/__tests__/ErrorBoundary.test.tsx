import type { ReactElement } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "../components/ErrorBoundary";

/** A component that throws on render when its `boom` prop is true. */
function Boom({ boom, label }: { boom: boolean; label: string }) {
  if (boom) {
    throw new Error(`boom-${label}`);
  }
  return <div data-testid="ok">{label}</div>;
}

describe("ErrorBoundary", () => {
  it("renders children when nothing throws", () => {
    render(
      <ErrorBoundary resetKey="a">
        <Boom boom={false} label="healthy" />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("ok")).toHaveTextContent("healthy");
  });

  it("shows the fallback when a child throws", () => {
    render(
      <ErrorBoundary resetKey="a">
        <Boom boom={true} label="bad" />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
    expect(screen.getByText(/boom-bad/i)).toBeInTheDocument();
  });

  it("recovers when resetKey changes (does not stick the fallback until reload)", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { rerender } = render(
      <ErrorBoundary resetKey="case-1">
        <Boom boom={true} label="bad" />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/boom-bad/i)).toBeInTheDocument();

    // Simulate the user selecting a different case -> resetKey changes ->
    // boundary must clear the error and retry children.
    rerender(
      <ErrorBoundary resetKey="case-2">
        <Boom boom={false} label="recovered" />
      </ErrorBoundary>,
    );
    expect(screen.queryByText(/Something went wrong/i)).not.toBeInTheDocument();
    expect(screen.getByTestId("ok")).toHaveTextContent("recovered");
    consoleSpy.mockRestore();
  });

  it("offers a manual retry button that clears the error", () => {
    let shouldBoom = true;
    const Stateful = () => <Boom boom={shouldBoom} label="x" />;
    const { rerender } = render(
      <ErrorBoundary resetKey="k">
        <Stateful />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();

    // Fix the underlying fault, then click retry.
    shouldBoom = false;
    rerender(
      <ErrorBoundary resetKey="k">
        <Stateful />
      </ErrorBoundary>,
    );
    fireEvent.click(screen.getByRole("button", { name: /Retry/i }));
    expect(screen.queryByText(/Something went wrong/i)).not.toBeInTheDocument();
  });

  it("supports a custom fallback renderer", () => {
    render(
      <ErrorBoundary
        resetKey="k"
        fallback={(err) => <div data-testid="custom">{err.message}</div>}
      >
        <Boom boom={true} label="custom" />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("custom")).toHaveTextContent("boom-custom");
  });

  it("shows the fallback when a child throws a falsy non-Error value", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    function ThrowFalsy(): ReactElement {
      // React permits throwing any value; some libs throw strings or null.
      // The boundary must still render its fallback (truthy `if (error)`
      // would wrongly re-render the throwing child).
      throw null;
    }
    render(
      <ErrorBoundary resetKey="k">
        <ThrowFalsy />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
    // The falsy throw is normalized into an Error whose message is non-empty.
    expect(screen.getByText(/Thrown: null/i)).toBeInTheDocument();
    consoleSpy.mockRestore();
  });
});

