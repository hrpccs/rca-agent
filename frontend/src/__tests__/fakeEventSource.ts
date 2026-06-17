/**
 * A spec-shaped fake of the browser `EventSource` for testing the SSE consumer
 * in api.ts without a backend. jsdom does NOT provide `EventSource`, so this
 * polyfill on `globalThis`/`window` in tests.
 *
 * It supports:
 *  - named-event addEventListener / removeEventListener / onerror
 *  - `readyState` transitions (CONNECTING -> OPEN -> CLOSED) driven by the
 *    test via `simulateOpen()` / `simulateError()`
 *  - `dispatchEvent` to deliver a parsed SSE message to a named listener
 *  - `close()` recording that the stream was closed and readyState -> CLOSED
 *
 * Tests retrieve emitted events and close-state through the returned instance.
 */

export const FAKE_CONNECTING = 0 as const;
export const FAKE_OPEN = 1 as const;
export const FAKE_CLOSED = 2 as const;

export type FakeReadyState = typeof FAKE_CONNECTING | typeof FAKE_OPEN | typeof FAKE_CLOSED;

type Listener = (ev: MessageEvent) => void;

interface EmittedMessage {
  type: string;
  data?: string;
}

/**
 * A minimal EventSource mock. Instances carry the static CLOSED constant so
 * the production api.ts guard `es.readyState === EventSource.CLOSED` works.
 */
export class FakeEventSource {
  static readonly CONNECTING = FAKE_CONNECTING;
  static readonly OPEN = FAKE_OPEN;
  static readonly CLOSED = FAKE_CLOSED;

  readonly url: string;
  readyState: FakeReadyState = FAKE_CONNECTING;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;

  private listeners = new Map<string, Set<Listener>>();
  private closed = false;
  private emitted: EmittedMessage[] = [];

  constructor(url: string) {
    this.url = url;
    // Asynchronously transition to OPEN like a real EventSource would.
    // Tests can also call simulateOpen() to force the transition synchronously
    // (the api consumer doesn't depend on OPEN for dispatch).
    fakeEventSourcesCreated.push(this);
  }

  addEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type)!.add(listener as Listener);
  }

  removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    this.listeners.get(type)?.delete(listener as Listener);
  }

  /** @internal production EventSource field */
  get CLOSED(): FakeReadyState {
    return FakeEventSource.CLOSED;
  }

  /** Test driver: mark the connection OPEN and fire onopen. */
  simulateOpen(): void {
    this.readyState = FAKE_OPEN;
    this.onopen?.(new Event("open"));
  }

  /** Test driver: deliver a named SSE event to registered listeners. */
  dispatchEventMessage(type: string, data?: unknown): void {
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    this.emitted.push({ type, data: payload });
    const set = this.listeners.get(type);
    const ev = new MessageEvent(type, { data: payload });
    set?.forEach((l) => l(ev));
    // `message` listeners also receive events without an explicit event: line.
    if (type !== "message") {
      this.listeners.get("message")?.forEach((l) => l(ev));
    }
  }

  /**
   * Test driver: simulate a native transport error. If `forceClosed` is true
   * (or readyState is already CLOSED), set readyState = CLOSED before firing.
   *
   * Mirrors a real browser EventSource, which on a transport failure fires the
   * `error` event to BOTH the `.onerror` handler AND any registered
   * `addEventListener("error", …)` listeners, with a MessageEvent whose `data`
   * is undefined (no SSE payload). The named-listener dispatch (with
   * data===undefined) is what exercises the production guard
   * `kind === "error" && msg.data == null` in api.ts.
   */
  simulateTransportError(forceClosed = true): void {
    if (forceClosed) this.readyState = FAKE_CLOSED;
    const ev = new Event("error");
    this.onerror?.(ev);
    // Also dispatch an error MessageEvent with no data to named listeners, like
    // a real EventSource does on native transport failure.
    const msgEv = new MessageEvent("error", { data: undefined });
    this.listeners.get("error")?.forEach((l) => l(msgEv));
  }

  close(): void {
    this.closed = true;
    this.readyState = FAKE_CLOSED;
  }

  // --- test-only inspectors ---
  isClosed(): boolean {
    return this.closed;
  }
  emittedMessages(): EmittedMessage[] {
    return [...this.emitted];
  }
}

/** Tracks every FakeEventSource constructed during a test, in order. */
export const fakeEventSourcesCreated: FakeEventSource[] = [];

/**
 * Install (or restore) the FakeEventSource as `globalThis.EventSource` and
 * `window.EventSource`. Returns a cleanup function. The created-instances log
 * is reset on install.
 */
export function installFakeEventSource(): {
  FakeEventSource: typeof FakeEventSource;
  cleanup: () => void;
} {
  fakeEventSourcesCreated.length = 0;
  const previous = {
    global: (globalThis as { EventSource?: unknown }).EventSource,
    window: (window as { EventSource?: unknown }).EventSource,
  };
  (globalThis as { EventSource: unknown }).EventSource = FakeEventSource;
  (window as { EventSource: unknown }).EventSource = FakeEventSource;
  return {
    FakeEventSource,
    cleanup: () => {
      (globalThis as { EventSource: unknown }).EventSource = previous.global;
      (window as { EventSource: unknown }).EventSource = previous.window;
    },
  };
}
