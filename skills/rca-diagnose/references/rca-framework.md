# 6 阶段 RCA 诊断框架

> 通用根因分析框架，适用于所有告警类型。专项 SOP 在此框架基础上展开。

---

## 1. 核心推理原则

| 原则 | 说明 |
|------|------|
| **先拓扑后指标** | 通过 topo 发现调查目标，避免盲目搜索指标 |
| **告警实体 != 根因实体** | 根因几乎总在依赖链的下游（被调用方） |
| **证据三角验证** | 量化证据(指标) + 结构证据(trace) + 因果证据(事件/日志)，缺一不可 |
| **最多向下追踪 3 层** | 告警服务 -> 直接依赖 -> 依赖的依赖，不无限深入 |
| **数据驱动，禁止幻觉** | 所有结论必须基于查询返回的真实数据，数据缺失标注 N/A |

---

## 2. 六阶段诊断流程

```
Stage 1: 告警上下文确认
  |
  v
Stage 2: 实体拓扑发现
  |
  v
Stage 3: 指标量化确认
  |
  v
Stage 4: 链路追踪定位
  |
  v
Stage 5: 基础设施验证
  |
  v
Stage 6: 日志佐证
```

---

### Stage 1: 告警上下文确认

**目标**: 理解告警内容，解析告警实体，确定调查时间窗口。

**工具调用**:

```bash
# 构建完整诊断上下文
python cli/query.py context build --task <task_id>
# 工具名: context_build

# 查询告警详情
python cli/query.py alert query --task <task_id>
# 工具名: alert_query
```

**关键输出**:
- `alert_title`: 告警标题
- `alert_entity`: 告警实体名称和 ID
- `alert_window`: 告警时间窗口（起止时间）
- `severity`: 告警级别

**推理检查**: 告警实体是什么类型？服务？Pod？Node？这将决定后续拓扑发现的方向。

---

### Stage 2: 实体拓扑发现

**目标**: 找到告警实体的上下游依赖，确定调查范围。

**工具调用**:

```bash
# 搜索告警实体
python cli/query.py entity search --task <task_id> --keyword <alert_entity>
# 工具名: entity_search

# 获取拓扑邻居（上下游依赖）
python cli/query.py topo neighbors --task <task_id> --entity-id <entity_id> --depth 2
# 工具名: topo_neighbors

# 获取调用图（可选，更精确的调用关系）
python cli/query.py topo callgraph --task <task_id> --entity-id <entity_id> --depth 2
# 工具名: topo_callgraph
```

**关键输出**:
- `upstream`: 调用告警服务的上游服务列表
- `downstream`: 告警服务依赖的下游服务列表
- `edges`: 调用方向和关系类型

**推理检查**: 依赖链中哪个下游最可能是根因？构建 `A -> B -> C` 的假设。

---

### Stage 3: 指标量化确认

**目标**: 用指标数据量化异常幅度，确认症状真实存在。

**工具调用**:

```bash
# 查询告警实体的指标（窗口 vs 基线对比）
python cli/query.py metric compare --task <task_id> --entity-id <entity_id> --name <metric_name>
# 工具名: metric_compare

# 查询具体指标时序数据
python cli/query.py metric query --task <task_id> --entity-id <entity_id> --name error_count
# 工具名: metric_query

# 聚合查询（跨服务对比）
python cli/query.py metric aggregate --task <task_id> --name error_rate --group-by service --agg sum
# 工具名: metric_aggregate
```

**关键指标对照表**:

| 症状类型 | 首查指标 | 说明 |
|---------|---------|------|
| 错误率升高 | `error_rate`, `error_count` | 确认错误量级和趋势 |
| 延迟升高 | `avg_request_latency_seconds`, `p99_request_latency_seconds` | 确认延迟幅度 |
| 流量变化 | `request_count` | 确认请求量变化方向 |
| 资源异常 | `cpu_usage`, `memory_usage` | 排除资源瓶颈 |

**推理检查**: 异常幅度多大？是否仅影响告警实体自身？下游是否也有异常？

---

### Stage 4: 链路追踪定位

**目标**: 通过 trace 数据找到错误的源头或延迟的瓶颈。

**工具调用**:

```bash
# 搜索错误/慢调用 trace
python cli/query.py trace search --task <task_id> --condition error --service <service_name> --limit 5
# 工具名: trace_search

# 批量诊断（自动归因多条 trace）
python cli/query.py trace batch-diagnose --task <task_id> --condition error --service <service_name> --limit 5
# 工具名: trace_batch-diagnose

# 单条 trace 诊断
python cli/query.py trace diagnose --task <task_id> --trace-id <trace_id>
# 工具名: trace_diagnose
```

**关键输出**:
- `root_error_span`: 根因 Span（最深层错误）
- `error_propagation_path`: 错误传播路径
- `primary_bottleneck_span`: 延迟瓶颈 Span
- `diagnosis_type`: `error_trace` / `slow_trace` / `not_found`

**推理检查**: 错误从哪里开始？是告警服务自身还是下游？延迟瓶颈在哪个 Span？

---

### Stage 5: 基础设施验证

**目标**: 检查 K8s 事件、Pod 状态、节点资源，排除基础设施层面的问题。

**工具调用**:

```bash
# 查询关键 K8s 事件（Pod Killing, OOMKilled, CrashLoop 等）
python cli/query.py event critical --task <task_id>
# 工具名: event_critical

# 查询特定 Pod 的 K8s 事件
python cli/query.py event k8s --task <task_id> --pod-name <pod_name> --namespace <namespace>
# 工具名: event_k8s
```

**事件类型解读**:

| 事件关键词 | 含义 | 指向 |
|-----------|------|------|
| `Killing` + `Created` + `Started` | Pod 被杀后重建 | Pod Kill 故障 |
| `BackOff` | CrashLoopBackOff 重启循环 | 应用崩溃 |
| `OOMKilled` | 内存溢出被杀 | 内存问题 |
| `Evicted` | 节点资源压力驱逐 | 节点资源不足 |
| `Unhealthy` (Readiness probe failed) | 就绪探针失败 | LB 无法路由流量 |

**推理检查**: 是否有 Pod 重启/OOM/驱逐事件？事件时间是否与告警时间对齐？

---

### Stage 6: 日志佐证

**目标**: 用日志确认根因假设，找到具体的错误消息或异常堆栈。

**工具调用**:

```bash
# 搜索服务日志
python cli/query.py log search --task <task_id> --service <service_name> --keyword "error" --limit 30
# 工具名: log_search

# 获取错误日志摘要（自动聚合错误类型）
python cli/query.py log error-summary --task <task_id> --service <service_name>
# 工具名: log_error-summary
```

**关键日志模式**:

| 日志关键词 | 指向的根因类型 |
|-----------|-------------|
| `connection refused` / `ECONNREFUSED` | 下游服务不可达 |
| `timeout` / `timed out` | 下游响应慢或网络问题 |
| `OOM` / `OutOfMemoryError` | 内存溢出 |
| `NullPointerException` | 应用代码 Bug |
| `CrashLoopBackOff` | 容器启动失败 |
| `redis` / `cache miss` | 缓存问题 |
| `slow query` / `SQL` | 数据库慢查询 |

**推理检查**: 日志中的错误消息是否能解释 trace 中看到的异常？时间是否对齐？

---

## 3. 证据收敛规则

1. **三项证据必须一致**: 指标（量化） + Trace（定位） + 事件/日志（佐证）必须指向同一个根因实体
2. **时间对齐**: 所有异常信号必须在时间窗口内对齐，否则可能是不相关事件
3. **因果方向**: 根因在下游（被依赖方），症状在上游（调用方），不要反向归因
4. **置信度评估**:
   - **高**: 三角证据齐全，因果链完整
   - **中**: 两项证据支持，一项缺失或模糊
   - **低**: 仅一项证据，其他缺失

---

## 4. 反幻觉约束

| 规则 | 说明 |
|------|------|
| 禁止编造指标名 | 指标名必须从 `metric list` 或 `metric query` 的返回结果中获取 |
| 禁止编造 entity_id | entity_id 必须从 `entity search` 的返回结果中获取 |
| 禁止编造 trace_id | trace_id 必须从 `trace search` 的返回结果中获取 |
| 数据缺失标注 N/A | 查询返回空结果时，在报告中标注 N/A，不猜测 |
| 不罗列多个可能原因 | 直接给出置信度最高的根因，不列"可能原因"列表 |
