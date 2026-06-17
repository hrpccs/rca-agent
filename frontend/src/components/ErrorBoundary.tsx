import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  /**
   * When this value changes, the boundary recovers: any captured error is
   * cleared and the children are re-rendered. Pass something that changes
   * between distinct user contexts — e.g. the selected case id — so a
   * transient render fault in one context does not permanently wedge the UI
   * until a full page reload.
   */
  resetKey?: string | number | null;
  /** Optional renderer for the fallback UI. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
  /**
   * Mirror of the `resetKey` last seen by the boundary. Used by
   * getDerivedStateFromProps to detect a prop *change* (vs. the initial
   * render) and clear a captured error only when the key has actually moved
   * on. This is the canonical React pattern for "reset derived state when a
   * prop changes" — see
   * https://react.dev/reference/react/Component#static-getderivedstatefromprops
   */
  prevResetKey: string | number | null | undefined;
}

/**
 * A plain React class error boundary (no extra dependency).
 *
 * Recovery semantics: whenever `resetKey` changes after an error has been
 * captured, the boundary clears the error and re-renders its children. This
 * prevents a single bad render from sticking the fallback until a full reload.
 * Callers can also expose a manual reset button via the `fallback` renderer.
 *
 * Implementation note: recovery is driven entirely by the static
 * getDerivedStateFromProps comparison of `props.resetKey` against the mirrored
 * `state.prevResetKey`. getDerivedStateFromError only sets the error and
 * leaves `prevResetKey` untouched, so the boundary does NOT clear the error on
 * the render immediately following the throw (props.resetKey still equals the
 * mirrored baseline there).
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  override state: ErrorBoundaryState = { error: null, prevResetKey: undefined };

  static getDerivedStateFromProps(
    props: ErrorBoundaryProps,
    state: ErrorBoundaryState,
  ): Partial<ErrorBoundaryState> | null {
    const changed = props.resetKey !== state.prevResetKey;
    if (!changed) return null;
    // The reset key moved on. If we're currently showing an error, clear it so
    // the children get retried under the new context. Either way, re-baseline
    // prevResetKey to the incoming value so we only react to future changes.
    if (state.error != null) {
      return { error: null, prevResetKey: props.resetKey };
    }
    return { prevResetKey: props.resetKey };
  }

  static getDerivedStateFromError(error: unknown): Partial<ErrorBoundaryState> {
    // Set the error WITHOUT touching prevResetKey. At this point
    // state.prevResetKey already mirrors the current props.resetKey (kept fresh
    // by getDerivedStateFromProps on every prior render), so leaving it alone
    // means the next getDerivedStateFromProps pass sees no change and the
    // fallback is shown rather than immediately cleared.
    //
    // Coerce non-Error throws (React permits throwing any value; some libs
    // throw strings, null, or numbers) into a real Error so the fallback can
    // safely read `error.message` and the `error !== null` render guard holds
    // even for falsy throws.
    const normalized =
      error instanceof Error
        ? error
        : new Error(
            typeof error === "string"
              ? error
              : `Thrown: ${String(error)}`,
          );
    return { error: normalized };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surface to the console for dev visibility; a production app would forward
    // this to a telemetry sink. Intentionally not re-thrown.
    console.error("[ErrorBoundary] captured render error:", error, info.componentStack);
  }

  private reset = () => {
    this.setState({ error: null, prevResetKey: this.props.resetKey });
  };

  override render(): ReactNode {
    const { error } = this.state;
    // Use an explicit null check rather than truthiness: React permits throwing
    // any value (some libs throw strings or null), and getDerivedStateFromError
    // would store a falsy non-null value here. `!== null` correctly shows the
    // fallback in that case instead of re-rendering the throwing children.
    if (error !== null) {
      if (this.props.fallback) {
        return this.props.fallback(error, this.reset);
      }
      return (
        <div className="error-boundary" role="alert">
          <h2>Something went wrong · 渲染异常</h2>
          <p className="error-boundary__msg">{error.message}</p>
          <button type="button" className="error-boundary__reset" onClick={this.reset}>
            Retry · 重试
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
