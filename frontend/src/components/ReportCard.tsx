import type { RcaReport } from "../types";

interface ReportCardProps {
  report: RcaReport;
}

const STATUS_LABEL: Record<string, string> = {
  completed: "completed · 完成",
  truncated: "truncated · 截断",
  error: "error · 错误",
};

function statusClass(status: string): string {
  if (status === "completed") return "badge--ok";
  if (status === "error") return "badge--err";
  return "badge--warn";
}

/** Renders the final root-cause report when the `report` SSE event arrives. */
export function ReportCard({ report }: ReportCardProps) {
  const rc = report.root_cause;
  const conf = Math.max(0, Math.min(1, rc.confidence ?? 0));
  const tokens = report.token_usage ?? {};

  return (
    <section className="report">
      <div className="report__head">
        <h2>Root Cause Report · 根因报告</h2>
        <span className={`badge ${statusClass(report.status)}`}>
          {STATUS_LABEL[report.status] ?? report.status}
        </span>
      </div>

      <div className="report__alert">
        <span className="report__alert-label">Alert · 告警</span>
        <span className="report__alert-title">{report.alert_title}</span>
        <span className="report__meta">
          case <code>{report.case_id}</code> · task <code>{report.task_id}</code>
          {report.model ? ` · ${report.model}` : ""}
        </span>
      </div>

      <div className="report__summary">
        <h3>Summary · 摘要</h3>
        <p>{rc.summary}</p>
      </div>

      <div className="report__grid">
        <div className="report__field">
          <span className="report__field-label">Fault type · 故障类型</span>
          {rc.fault_type ? (
            <span className="badge badge--fault">{rc.fault_type}</span>
          ) : (
            <span className="report__muted">—</span>
          )}
        </div>

        <div className="report__field">
          <span className="report__field-label">Confidence · 置信度</span>
          <div className="report__conf">
            <div className="report__conf-bar">
              <div className="report__conf-fill" style={{ width: `${conf * 100}%` }} />
            </div>
            <span className="report__conf-pct">{(conf * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {rc.entity_refs && rc.entity_refs.length > 0 && (
        <div className="report__block">
          <h3>Entities · 关联实体</h3>
          <div className="report__chips">
            {rc.entity_refs.map((e, i) => (
              <span key={`${e.entity_name}-${i}`} className="chip chip--entity" title={e.entity_type}>
                <span className="chip__type">{e.entity_type ?? e.entity_domain ?? "entity"}</span>
                <span className="chip__name">{e.entity_name}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {rc.evidence && rc.evidence.length > 0 && (
        <div className="report__block">
          <h3>Evidence · 证据</h3>
          <ul className="report__list">
            {rc.evidence.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {rc.contributing_factors && rc.contributing_factors.length > 0 && (
        <div className="report__block">
          <h3>Contributing factors · 促成因素</h3>
          <ul className="report__list">
            {rc.contributing_factors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {rc.recommended_actions && rc.recommended_actions.length > 0 && (
        <div className="report__block">
          <h3>Recommended actions · 建议措施</h3>
          <ol className="report__list report__list--ordered">
            {rc.recommended_actions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ol>
        </div>
      )}

      <div className="report__footer">
        <span className="report__footer-label">Token usage · Token 用量</span>
        <span className="report__tokens">
          prompt <b>{tokens.prompt_tokens ?? "—"}</b> · completion{" "}
          <b>{tokens.completion_tokens ?? "—"}</b> · total{" "}
          <b>{tokens.total_tokens ?? "—"}</b>
        </span>
      </div>
    </section>
  );
}
