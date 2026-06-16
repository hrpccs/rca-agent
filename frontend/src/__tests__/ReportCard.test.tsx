import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReportCard } from "../components/ReportCard";
import type { RcaReport } from "../types";

const fixture: RcaReport = {
  case_id: "t001",
  task_id: "task-1",
  alert_title: "Pod CPU 使用率过高",
  status: "completed",
  root_cause: {
    summary: "The checkout-svc pod entered a CrashLoopBackOff due to an OOM kill.",
    fault_type: "k8s.pod_crashloop",
    entity_refs: [
      { entity_name: "checkout-svc-7b9", entity_type: "pod", entity_domain: "k8s" },
      { entity_name: "prod", entity_type: "cluster", entity_domain: "k8s" },
    ],
    evidence: ["memory usage >95% for 5m", "OOMKilled exit code 137"],
    confidence: 0.92,
    contributing_factors: ["memory limit set too low"],
    recommended_actions: ["raise memory limit", "add HPA on memory"],
  },
  steps: [],
  token_usage: { prompt_tokens: 1200, completion_tokens: 340, total_tokens: 1540 },
};

describe("ReportCard", () => {
  it("renders summary, confidence, entities, and token usage", () => {
    render(<ReportCard report={fixture} />);

    expect(screen.getByText(/CrashLoopBackOff/i)).toBeInTheDocument();
    expect(screen.getByText("k8s.pod_crashloop")).toBeInTheDocument();
    expect(screen.getByText("checkout-svc-7b9")).toBeInTheDocument();
    expect(screen.getByText(/raise memory limit/i)).toBeInTheDocument();
    expect(screen.getByText("1540")).toBeInTheDocument();
    // 92% confidence text rendered.
    expect(screen.getByText("92%")).toBeInTheDocument();
  });

  it("renders an error status badge when status is error", () => {
    render(<ReportCard report={{ ...fixture, status: "error" }} />);
    expect(screen.getByText(/error/i)).toBeInTheDocument();
  });
});
