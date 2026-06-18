# 错误率/错误次数飙升诊断 SOP

> 触发条件: 告警标题含 "错误次数" / "异常" / "error_count" / "error_rate" / "5xx" 等关键词。

---

## 1. 诊断决策树

```
告警: 某服务错误率/错误次数升高
  |
  +-- 1. metric_compare 量化错误率幅度
  |     |
  |     +-- error_rate ~100% + HTTP 500
  |     |     -> 应用代码级 Bug（NPE、配置错误等）
  |     |     -> 走分支 A: 代码 Bug 定位
  |     |
  |     +-- error_rate ~50%（非 100%，条件性失败）
  |     |     -> 特定请求类型失败，可能下游部分实例异常
  |     |     -> 走分支 B: 下游定位
  |     |
  |     +-- error_rate 较低（1%-10%）
  |           -> 偶发错误，可能是间歇性下游超时或网络抖动
  |           -> 走分支 B: 下游定位
  |
  +-- 2. trace_batch-diagnose 定位错误来源
  |     |
  |     +-- 下游服务也有错误
  |     |     -> 根因在下游，递归查下游服务
  |     |
  |     +-- 仅自身有错误，下游正常
  |           -> 根因在自身应用代码
  |
  +-- 3. event_critical 检查 Pod 事件
  |     |
  |     +-- 有 Pod Killing/CrashLoop 事件
  |     |     -> 转入 05_pod_crash SOP
  |     |
  |     +-- 无基础设施事件
  |           -> 继续日志佐证
  |
  +-- 4. log_search 确认具体错误消息
```

---

## 2. 工具调用序列

### 标准序列

```
context_build -> metric_compare -> trace_batch-diagnose -> event_critical -> log_search
```

### Step 1: 量化错误率

```bash
# 窗口 vs 基线对比
python cli/query.py metric compare --task <task_id> --entity-id <entity_id> --name error_rate
# 工具名: metric_compare

# 查询错误计数时序
python cli/query.py metric query --task <task_id> --entity-id <entity_id> --name error_count
# 工具名: metric_query

# 跨服务对比（找出哪些服务同时异常）
python cli/query.py metric aggregate --task <task_id> --name error_rate --group-by service --agg avg --limit 10
# 工具名: metric_aggregate
```

**判断规则**:
- `error_rate ~100%` -> 应用级 Bug（所有请求都失败）
- `error_rate 30%-60%` -> 条件性失败（部分请求类型失败或部分实例异常）
- `error_rate < 10%` -> 偶发错误（间歇性下游问题）

### Step 2: Trace 定位错误来源

```bash
# 批量诊断错误 trace（自动归因）
python cli/query.py trace batch-diagnose --task <task_id> --condition error --service <service_name> --limit 5
# 工具名: trace_batch-diagnose

# 如果 batch-diagnose 指向下游，递归查下游
python cli/query.py trace batch-diagnose --task <task_id> --condition error --service <downstream_service> --limit 5

# 深入单条 trace 确认根因 Span
python cli/query.py trace diagnose --task <task_id> --trace-id <trace_id>
# 工具名: trace_diagnose
```

**关键信号**:
- `span duration=0ms + error status` -> 连接从未建立（下游不可达）
- `gRPC status=UNAVAILABLE (code=2)` -> 下游服务完全不可用
- `HTTP 500 + TypeError/NullPointerException` -> 应用代码级 Bug
- `HTTP 403/429` -> 限流/拒绝

### Step 3: 检查 K8s 事件

```bash
# 查询关键事件
python cli/query.py event critical --task <task_id>
# 工具名: event_critical
```

**如果发现**:
- `Killing` 事件 -> Pod 被杀，转入 05_pod_crash SOP
- `OOMKilled` -> 内存溢出，转入 05_pod_crash SOP
- 无事件 -> 排除基础设施问题，继续日志分析

### Step 4: 日志佐证

```bash
# 搜索错误日志
python cli/query.py log search --task <task_id> --service <service_name> --keyword "error" --limit 30
# 工具名: log_search

# 获取错误摘要（自动聚合错误类型）
python cli/query.py log error-summary --task <task_id> --service <service_name>
# 工具名: log_error-summary
```

---

## 3. 典型故障模式速查

| 错误特征 | 根因类型 | 关键证据 |
|---------|---------|---------|
| error_rate ~100%, 所有 Pod 受影响 | 应用代码 Bug（NPE、配置错误） | trace: HTTP 500 一致返回; log: Exception 堆栈 |
| error_rate ~50%, 特定接口失败 | 条件性 Bug（特定参数/路径触发） | trace: 仅特定 span 报错; log: 特定错误消息 |
| error_rate 升高 + 下游也报错 | 下游服务故障级联 | trace: 错误指向下游; 下游 metric 也异常 |
| error_rate 升高 + Pod 重启 | Pod 不稳定导致部分请求失败 | event: BackOff/Killing; trace: connection refused |
| error_rate 升高 + connection refused | 下游服务完全不可达 | trace: span duration=0ms; event: Killing |
| error_rate 升高 + HTTP 403/429 | 下游限流/拒绝 | trace: HTTP 403/429; 下游 error_rate > 90% |

---

## 4. 根因收敛规则

1. **根因在依赖链最深层的故障实体**: 如果 A 调用 B，B 调用 C，C 报错 -> 根因是 C
2. **最多追踪 3 层**: 告警服务 -> 直接依赖 -> 依赖的依赖
3. **不反向归因**: 下游故障导致上游报错，不能说"上游调用过多导致下游故障"（除非有流量突增证据）
4. **三角验证**: trace 中的错误 Span + metric 的错误率曲线 + log 的错误消息必须时间对齐、指向同一实体
