import { useCallback, useEffect, useRef, useState } from "react";
import { fetchCases, openRcaStream, startRca, type StreamHandlers } from "./api";
import type { Backend, RcaReport, RcaStep } from "./types";
import { CasePicker } from "./components/CasePicker";
import { ReportCard } from "./components/ReportCard";
import { RunPanel } from "./components/RunPanel";
import { TraceTimeline } from "./components/TraceTimeline";
import "./App.css";

type RunStatus = "idle" | "starting" | "running" | "done" | "error";

export default function App() {
  const [cases, setCases] = useState<string[]>([]);
  const [casesError, setCasesError] = useState<string | null>(null);
  const [selectedCase, setSelectedCase] = useState<string | null>(null);
  const [backend, setBackend] = useState<Backend>("parquet");

  const [steps, setSteps] = useState<RcaStep[]>([]);
  const [report, setReport] = useState<RcaReport | null>(null);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const [elapsedMs, setElapsedMs] = useState(0);

  const eventSourceRef = useRef<EventSource | null>(null);
  const tickerRef = useRef<number | null>(null);
  // Ref mirror of the run start timestamp so stream callbacks read the latest
  // value without needing to re-bind the handlers.
  const startedAtRef = useRef<number | null>(null);

  // Load case list on mount.
  useEffect(() => {
    let cancelled = false;
    setCasesError(null);
    fetchCases()
      .then((c) => {
        if (!cancelled) setCases(c);
      })
      .catch((e: Error) => {
        if (!cancelled) setCasesError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const closeStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (tickerRef.current != null) {
      window.clearInterval(tickerRef.current);
      tickerRef.current = null;
    }
  }, []);

  // Clean up on unmount.
  useEffect(() => closeStream, [closeStream]);

  // Live elapsed-time ticker while running.
  useEffect(() => {
    if (status === "running" && startedAtRef.current != null) {
      tickerRef.current = window.setInterval(() => {
        setElapsedMs(Date.now() - (startedAtRef.current ?? Date.now()));
      }, 200);
    }
    return () => {
      if (tickerRef.current != null) {
        window.clearInterval(tickerRef.current);
        tickerRef.current = null;
      }
    };
  }, [status]);

  const resetRun = useCallback(() => {
    setSteps([]);
    setReport(null);
    setError(null);
    setElapsedMs(0);
    startedAtRef.current = null;
  }, []);

  const handleRun = useCallback(async () => {
    if (!selectedCase) return;
    closeStream();
    resetRun();
    setStatus("starting");
    startedAtRef.current = Date.now();
    setElapsedMs(0);

    const handlers: StreamHandlers = {
      onStep: (step) => {
        setStatus((s) => (s === "starting" ? "running" : s));
        setSteps((prev) => [...prev, step]);
      },
      onReport: (rep) => {
        setReport(rep);
      },
      onDone: () => {
        if (startedAtRef.current != null) {
          setElapsedMs(Date.now() - startedAtRef.current);
        }
        setStatus("done");
        closeStream();
      },
      onError: (msg) => {
        setError(msg);
        setStatus("error");
        closeStream();
      },
    };

    try {
      await startRca(selectedCase, backend);
    } catch {
      // POST may fail in some setups; the SSE stream is what matters. Continue.
    }
    setStatus("running");
    eventSourceRef.current = openRcaStream(selectedCase, backend, handlers);
  }, [selectedCase, backend, closeStream, resetRun]);

  const handleStop = useCallback(() => {
    closeStream();
    setStatus((s) => (s === "running" || s === "starting" ? "done" : s));
  }, [closeStream]);

  const handleSelectCase = useCallback(
    (id: string) => {
      closeStream();
      setSelectedCase(id);
      resetRun();
      setStatus("idle");
    },
    [closeStream, resetRun],
  );

  return (
    <div className="app">
      <header className="app__header">
        <h1>RCA Ops Console · 根因分析控制台</h1>
        <span className={`app__status app__status--${status}`}>
          {status === "idle" && "就绪 / idle"}
          {status === "starting" && "启动中 / starting…"}
          {status === "running" && "运行中 / running"}
          {status === "done" && "完成 / done"}
          {status === "error" && "错误 / error"}
        </span>
      </header>

      <div className="app__body">
        <aside className="app__sidebar">
          <CasePicker
            cases={cases}
            selected={selectedCase}
            onSelect={handleSelectCase}
            loading={cases.length === 0 && casesError == null}
            error={casesError}
          />
        </aside>

        <main className="app__main">
          <RunPanel
            caseId={selectedCase}
            backend={backend}
            onBackendChange={setBackend}
            status={status}
            onRun={handleRun}
            onStop={handleStop}
            stepCount={steps.length}
            elapsedMs={elapsedMs}
          />

          {error && <div className="app__error">⚠ {error}</div>}

          <TraceTimeline steps={steps} running={status === "running"} />

          {report && <ReportCard report={report} />}
        </main>
      </div>
    </div>
  );
}
