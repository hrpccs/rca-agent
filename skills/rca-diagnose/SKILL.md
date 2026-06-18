---
name: rca-diagnose
description: >
  Start RCA diagnosis for a specific task. Use when the user provides a task ID
  and wants root cause analysis (e.g. "diagnose t001", "analyze t045", "/rca-diagnose t001").
  Takes a task_id (e.g. t001) as argument.
allowed-tools: Bash, Read, Write
---

# RCA Diagnosis Skill

你将对任务 **{{args}}** 执行完整的根因分析诊断。严格遵循以下流程。

---

## Step 0: 提取 Task ID

从参数 `{{args}}` 中提取 task_id（格式 `tNNN`）。如果参数不含有效 task_id，向用户询问。

---

## Step 1: 构建诊断上下文

**必须先执行此命令**，它预计算了告警实体、时间窗口、关键指标和异常信号。

```bash
python3 cli/query.py context build --task <task_id>
```

从返回结果中提取：
- `alert_title` — 告警标题
- `alert_entity` — 告警实体名称和 entity_id
- `alert_window` — 时间窗口（起止时间）
- `severity` — 告警级别
- `topology_summary` — 拓扑概览
- `anomaly_signals` — 异常信号列表

如果 context build 返回 `no_data`，执行 `alert query` 和 `entity types` 作为 fallback。

---

## Step 1.5: SCM 因果预过滤

context build 之后，运行 SCM 结构化因果分析，获取**预计算的怀疑实体排名**：

```bash
python3 cli/query.py scm analyze --task <task_id>
```

从返回结果中提取：
- `suspects` — Top-3 嫌疑实体（含故障概率、置信度、证据包）
- `dependency_graph` — 服务级依赖图（谁调用谁）
- `observations` — 各请求类型的成功/失败统计
- `metadata.has_failed_traces` — 是否有失败 trace（如果是 false，SCM 置信度低）

### 如何使用 SCM 结果指导后续排查

**SCM 已经完成了"谁有问题"的嫌疑筛选**，后续 Steps 应聚焦于"为什么有问题"：

| 后续 Step | SCM 指导 |
|-----------|---------|
| Step 2 信号路由 | 仍需执行（确定症状类别），但**跳过** entity search / topo neighbors |
| Step 3 SOP 执行 | 仅对 SCM Top-3 嫌疑实体查询指标（metric compare / metric query） |
| Step 4 Trace 分析 | 重点搜索涉及 Top-3 嫌疑实体的 trace（trace search --service <suspect>） |
| Step 5 日志/事件 | 重点查询 Top-3 嫌疑实体的日志和 K8s 事件 |

如果 SCM `has_failed_traces == false`（无失败 trace，如纯基础设施故障），则**忽略 SCM 结果**，回退到标准流程。

---

## Step 2: 六维信号分析 & 路由决策

在发出更多查询之前，先完成六维分析：

| 维度 | 含义 | 从哪获取 |
|------|------|---------|
| **intent** | 诊断意图 | alert_title 中的关键词 |
| **signal** | 信号域（metric/trace/log/event） | alert_title + anomaly_signals |
| **scope** | 调查范围 | alert_entity |
| **shape** | 期望输出形态 | 时序/排名/证据包 |
| **anchors** | 可复用精确值 | entity_id, alert_window |
| **blockers** | 缺失信息 | 无 entity_id → 需先 entity search |

### 症状 → SOP 路由决策树

根据 alert_title 和 anomaly_signals 中的关键词，路由到对应 SOP：

```
告警输入
  |
  +-- 关键词含 "错误次数" / "异常" / "error_count" / "error_rate" / "5xx"
  |     → 读取 references/error-rate-spike.md，按其流程执行
  |
  +-- 关键词含 "超时" / "响应时间" / "latency" / "timeout" / "P99" / "RT"
  |     → 读取 references/latency-spike.md，按其流程执行
  |
  +-- 关键词含 "流量下跌" / "traffic" / "drop" / "request_count 下降"
  |     → 读取 references/traffic-drop.md，按其流程执行
  |
  +-- 关键词含 "Pod" / "OOM" / "CrashLoop" / "重启" / "Killing"
  |     → 读取 references/pod-crash.md，按其流程执行
  |
  +-- 无法明确分类
        → 读取 references/rca-framework.md，执行完整 6 阶段诊断
```

### 快速分流（辅助判断）

```
error_rate 高 + 有下游依赖异常     → 错误率路径 (error-rate-spike)
延迟高 + 无明显错误               → 资源瓶颈路径 (latency-spike)
流量骤降 + 多服务同时退化         → 基础设施故障路径 (traffic-drop)
error_rate ~100% + HTTP 500       → 应用 Bug 路径 (error-rate-spike)
Pod 重启/CrashLoop                → Pod 崩溃路径 (pod-crash)
```

---

## Step 3: 读取 SOP Reference 并执行诊断

根据 Step 2 的路由结果，使用 Read 工具读取对应的参考文件：

| 路由目标 | 参考文件 |
|---------|---------|
| 错误率升高 | `references/error-rate-spike.md` |
| 延迟升高 | `references/latency-spike.md` |
| 流量骤降 | `references/traffic-drop.md` |
| Pod 崩溃 | `references/pod-crash.md` |
| 通用/未分类 | `references/rca-framework.md` |

**同时读取**以下辅助参考（用于 trace 解读和日志关键词选择）：
- `references/trace-analysis.md`
- `references/log-analysis.md`

**严格按 SOP 中的工具调用序列执行查询**。每个工具调用必须使用实际数据（从上一步结果中提取的 entity_id、metric_name 等）。

### 反幻觉约束

| 规则 | 说明 |
|------|------|
| 禁止编造指标名 | 指标名必须从 `metric list` 或 `metric query` 返回结果中获取 |
| 禁止编造 entity_id | entity_id 必须从 `entity search` 返回结果中获取 |
| 禁止编造 trace_id | trace_id 必须从 `trace search` 返回结果中获取 |
| 数据缺失标注 N/A | 查询返回空结果时标注 N/A，不猜测 |

---

## Step 4: 证据收敛

完成所有查询后，进行证据收敛：

1. **三项证据必须一致**：指标（量化） + Trace（定位） + 事件/日志（佐证）必须指向同一个根因实体
2. **时间对齐**：所有异常信号必须在 alert_window 内对齐
3. **因果方向**：根因在下游，症状在上游，不反向归因
4. **置信度评估**：
   - **high**：三角证据齐全，因果链完整，时间对齐
   - **medium**：两项证据支持，一项缺失或模糊
   - **low**：仅一项证据，或多项证据指向不同方向

---

## Step 5: 生成报告

读取 `references/report-template.md` 获取报告格式定义。

生成两个文件：

### JSON 报告 → `reports/<task_id>/report.json`

必须包含的字段：
- `task_id`, `alert_title`, `alert_entity`, `alert_window`
- `diagnosis.symptom_category`, `diagnosis.root_cause_type`, `diagnosis.root_cause_entity`
- `diagnosis.causal_chain`（从 root_cause → propagation → alert_trigger 的完整路径）
- `diagnosis.confidence`（high/medium/low）
- `diagnosis.evidence`（每项含 id, type, source, description, data）
- `tool_calls`（工具调用记录）
- `analysis_summary`（一句话结论）

### Markdown 报告 → `reports/<task_id>/report.md`

按模板输出：告警概述 → 诊断结论 → 根因说明 → 因果链 → 排障过程 → 证据 → 建议措施

**报告以 `<!-- REPORT_END -->` 结尾，之后不输出任何内容。**

---

## 深度知识（按需查阅）

如果诊断过程中遇到以下场景，使用 Read 工具查阅对应的领域知识：

| 场景 | 参考文件 |
|------|---------|
| JVM 相关问题（GC/OOM/线程池/连接池） | `references/` 目录下的 `sop-knowledge.md` |
| 观测工具选择（PromQL vs 封装工具） | `references/` 目录下的 `sop-observability.md` |
| APM 深度排查（火焰图/Trace 模式） | `references/` 目录下的 `sop-advanced.md` |
| 需要参考案例研究 | `references/` 目录下的 `case-study-01.md` ~ `04.md` |

<!-- ILLUSTRATIVE ONLY: the deep-knowledge files (sop-knowledge.md, sop-observability.md, sop-advanced.md, case-study-*.md) are NOT bundled in this skill. The absolute path below is illustrative prose, not a runtime dependency — the bundled skill ships only the 9 SOPs in references/. -->
> 注意：这些文件路径是相对于 **Skill 的 references/ 子目录** 的父级 `references/` 目录，位于项目根目录下。
> 完整路径示例：`/Users/hrpccs/Desktop/workspace/aiops/claude-cli-agent/references/sop-knowledge.md`
