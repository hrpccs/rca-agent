# 诊断报告输出模板

> 用途: 定义 RCA 诊断报告的标准输出格式，包括 JSON 格式和 Markdown 格式。
> 所有诊断完成后，按本模板生成报告。

---

## 1. JSON 报告格式 (report.json)

### 1.1 Schema 定义

```json
{
  "task_id": "t001",
  "alert_title": "checkout 服务错误次数超限",
  "alert_entity": {
    "name": "checkout",
    "entity_id": "abc123",
    "entity_type": "apm.service"
  },
  "alert_window": {
    "start": "2026-05-14T10:54:44+08:00",
    "end": "2026-05-14T11:09:44+08:00"
  },
  "diagnosis": {
    "symptom_category": "error_rate_spike",
    "root_cause_type": "downstream_service_failure",
    "root_cause_entity": {
      "name": "cart",
      "entity_id": "def456",
      "entity_type": "apm.service"
    },
    "causal_chain": [
      {
        "entity": "cart",
        "entity_type": "apm.service",
        "role": "root_cause",
        "symptom": "Pod 被 Kill 导致服务不可达",
        "evidence_ref": "ev_1"
      },
      {
        "entity": "checkout",
        "entity_type": "apm.service",
        "role": "propagation",
        "symptom": "调用 cart 失败，error_rate 升至 30%",
        "evidence_ref": "ev_2"
      },
      {
        "entity": "frontend",
        "entity_type": "apm.service",
        "role": "alert_trigger",
        "symptom": "上游 checkout 超时，用户请求失败",
        "evidence_ref": "ev_3"
      }
    ],
    "confidence": "high",
    "evidence": [
      {
        "id": "ev_1",
        "type": "event",
        "source": "event_critical",
        "description": "cart Pod Killing 事件，时间 10:55",
        "data": {"event_reason": "Killing", "pod_name": "cart-xxx", "timestamp": "..."}
      },
      {
        "id": "ev_2",
        "type": "metric",
        "source": "metric_compare",
        "description": "checkout error_rate 从 0% 升至 30%",
        "data": {"metric_name": "error_rate", "baseline": 0, "current": 30}
      },
      {
        "id": "ev_3",
        "type": "trace",
        "source": "trace_batch-diagnose",
        "description": "错误 trace 指向 cart 服务 connection refused",
        "data": {"root_span": "cart", "error_type": "connection refused"}
      }
    ]
  },
  "tool_calls": [
    {"tool": "context_build", "step": 1, "summary": "获取告警上下文"},
    {"tool": "entity_search", "step": 2, "summary": "搜索 checkout 实体"},
    {"tool": "metric_compare", "step": 3, "summary": "量化错误率变化"}
  ],
  "analysis_summary": "cart 服务 Pod 被 Kill 导致服务不可达，上游 checkout 调用 cart 时 connection refused，error_rate 升高至 30%。"
}
```

### 1.2 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | 是 | 任务 ID |
| `alert_title` | string | 是 | 告警标题 |
| `alert_entity` | object | 是 | 告警实体信息 |
| `alert_window` | object | 是 | 告警时间窗口 |
| `diagnosis.symptom_category` | string | 是 | 症状分类: `error_rate_spike` / `latency_spike` / `traffic_drop` / `pod_crash` |
| `diagnosis.root_cause_type` | string | 是 | 根因类型: `downstream_service_failure` / `application_bug` / `resource_saturation` / `pod_crash` / `network_issue` / `external_dependency` |
| `diagnosis.root_cause_entity` | object | 是 | 根因实体 |
| `diagnosis.causal_chain` | array | 是 | 因果链，从根因到告警实体的传播路径，每个节点含 role: `root_cause` / `propagation` / `alert_trigger` |
| `diagnosis.confidence` | string | 是 | 置信度: `high` / `medium` / `low` |
| `diagnosis.evidence` | array | 是 | 证据列表，每项含 id / type / source / description / data |
| `tool_calls` | array | 是 | 工具调用记录 |
| `analysis_summary` | string | 是 | 一句话分析结论 |

---

## 2. Markdown 报告格式 (report.md)

### 2.1 模板

```markdown
# 诊断报告

## 告警概述

| 项目 | 内容 |
|------|------|
| **告警标题** | <alert_title> |
| **告警实体** | <alert_entity_name> |
| **告警时间** | <alert_window_start> ~ <alert_window_end> |
| **告警级别** | <severity> |

## 诊断结论

| 项目 | 内容 |
|------|------|
| **症状分类** | <symptom_category> |
| **根因类型** | <root_cause_type> |
| **根因实体** | <root_cause_entity_name> |
| **置信度** | <confidence: 高/中/低> |

### 根因说明

<一段话描述根因，量化表达，包含具体数字>

### 因果链

<root_cause_entity> (根因) -> <propagation_entity> (传播) -> <alert_entity> (告警)

## 排障过程

### Stage 1: 告警上下文确认
<描述对告警的理解和实体解析结果>

### Stage 2: 实体拓扑发现
<描述发现的上下游依赖关系>

### Stage 3: 指标量化确认
<描述异常指标的量化结果，包含具体数字>

### Stage 4: 链路追踪定位
<描述 trace 诊断结果，指出根因 Span>

### Stage 5: 基础设施验证
<描述 K8s 事件检查结果>

### Stage 6: 日志佐证
<描述日志中发现的关键错误消息>

## 证据

| # | 类型 | 来源 | 描述 |
|---|------|------|------|
| 1 | metric | metric_compare | <具体指标变化> |
| 2 | trace | trace_batch-diagnose | <具体 trace 发现> |
| 3 | event | event_critical | <具体事件> |
| 4 | log | log_search | <具体日志消息> |

## 建议措施

1. <立即可执行的处置步骤>
2. <中期优化建议>

<!-- REPORT_END -->
```

### 2.2 关键规则

| 规则 | 说明 |
|------|------|
| **量化表达** | 所有结论必须附带具体数字（错误率从 X% 升至 Y%，延迟从 Xms 升至 Yms） |
| **单一根因** | 不罗列多个"可能原因"，直接给出置信度最高的根因 |
| **因果链完整** | 从根因实体到告警实体的完整传播路径 |
| **证据可追溯** | 每项证据标注来源工具和数据类型 |
| **禁止幻觉** | 数据缺失时标注 N/A，不编造 |
| **终止标记** | 报告以 `<!-- REPORT_END -->` 结尾，之后不输出任何内容 |

---

## 3. 置信度评估标准

| 等级 | 条件 |
|------|------|
| **高 (high)** | 三角证据齐全（metric + trace + event/log），因果链完整，时间对齐 |
| **中 (medium)** | 两项证据支持（如 metric + trace），事件或日志缺失 |
| **低 (low)** | 仅一项证据，或多项证据指向不同方向，因果链不完整 |

---

## 4. 报告生成规则

1. **先生成 JSON 再生成 Markdown**: JSON 是结构化数据，Markdown 是人类可读版本
2. **两个文件内容一致**: Markdown 中的数据必须与 JSON 完全一致
3. **分析总结不超过 3 句话**: 一句话说清楚根因，一句话说清楚传播路径，一句话说清楚影响
4. **排障过程按时间顺序**: 描述工具调用的顺序和每步的发现
5. **建议措施可执行**: 每条建议必须是具体的操作步骤，不是"建议检查"
6. **报告完成后立即停止**: 输出 `<!-- REPORT_END -->` 后不输出任何额外内容
