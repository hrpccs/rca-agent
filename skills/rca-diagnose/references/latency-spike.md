# 延迟/超时飙升诊断 SOP

> 触发条件: 告警标题含 "超时" / "响应时间" / "latency" / "timeout" / "P99" / "RT" / "慢请求" 等关键词。

---

## 1. 诊断决策树

```
告警: 某服务延迟/超时升高
  |
  +-- 1. metric_compare 量化延迟幅度
  |     |
  |     +-- avg 延迟正常但 P99 极高
  |     |     -> 间歇性慢请求（长尾延迟）
  |     |     -> 走分支 A: 慢请求定位
  |     |
  |     +-- avg 延迟也升高（全量变慢）
  |     |     -> 下游系统性变慢
  |     |     -> 走分支 B: 下游瓶颈定位
  |     |
  |     +-- 延迟 100x+ 飙升（如 0.3s -> 200s）
  |           -> 外部依赖完全不可用（缓存/DB）
  |           -> 走分支 C: 外部依赖故障
  |
  +-- 2. trace_batch-diagnose 定位延迟瓶颈
  |     |
  |     +-- 下游 Span 占总耗时 > 80%
  |     |     -> 根因在下游，递归查下游延迟
  |     |
  |     +-- 自身 Span 占主导（exclusive_duration 高）
  |           -> 根因在自身：CPU 饱和 / GC / 代码问题
  |
  +-- 3. metric_aggregate 检查资源饱和
  |     |
  |     +-- CPU/内存饱和 -> 资源瓶颈
  |     +-- 资源正常 -> 检查下游依赖指标
  |
  +-- 4. 日志佐证确认具体瓶颈
```

---

## 2. 工具调用序列

### 标准序列

```
context_build -> metric_compare -> trace_batch-diagnose -> metric_aggregate -> log_search
```

### Step 1: 量化延迟幅度

```bash
# 窗口 vs 基线对比（延迟）
python cli/query.py metric compare --task <task_id> --entity-id <entity_id> --name avg_request_latency_seconds
# 工具名: metric_compare

# 查询延迟时序
python cli/query.py metric query --task <task_id> --entity-id <entity_id> --name avg_request_latency_seconds
# 工具名: metric_query

# 查询慢请求数量
python cli/query.py metric query --task <task_id> --entity-id <entity_id> --name slow_count
# 工具名: metric_query
```

**判断规则**:
- 延迟 100x+ 飙升（如 0.3s -> 200s）-> 外部依赖不可用（Redis/DB）
- 延迟 5-10x 升高 -> 下游系统性变慢
- avg 正常但 P99 极高 -> 间歇性慢请求（长尾）

### Step 2: Trace 定位延迟瓶颈

```bash
# 搜索高延迟 trace
python cli/query.py trace search --task <task_id> --condition high_latency --service <service_name> --limit 5
# 工具名: trace_search

# 批量诊断（自动归因延迟瓶颈）
python cli/query.py trace batch-diagnose --task <task_id> --condition high_latency --service <service_name> --limit 5
# 工具名: trace_batch-diagnose

# 深入单条 trace 查看瓶颈 Span
python cli/query.py trace diagnose --task <task_id> --trace-id <trace_id>
# 工具名: trace_diagnose
```

**关键信号**:
- `single downstream span 占总延迟 99%+` -> 该下游是瓶颈（Redis/DB 不可用典型特征）
- `DB span 占总耗时 > 80%` -> 慢 SQL
- `多个下游 Span 同时变慢` -> 网络层/节点层问题
- `自身 Span 的 exclusive_duration 高` -> CPU/GC/代码瓶颈

### Step 3: 检查下游依赖指标

```bash
# 聚合查询各服务延迟对比
python cli/query.py metric aggregate --task <task_id> --name avg_request_latency_seconds --group-by service --agg max --limit 10
# 工具名: metric_aggregate

# 查询特定下游服务的延迟
python cli/query.py metric compare --task <task_id> --entity-id <downstream_entity_id> --name avg_request_latency_seconds
# 工具名: metric_compare
```

### Step 4: 资源饱和检查（可选）

```bash
# 检查 Pod 资源使用率
python cli/query.py metric query --task <task_id> --entity-id <pod_entity_id> --name cpu_usage
python cli/query.py metric query --task <task_id> --entity-id <pod_entity_id> --name memory_usage
# 工具名: metric_query
```

### Step 5: 日志佐证

```bash
# 搜索超时/慢请求相关日志
python cli/query.py log search --task <task_id> --service <service_name> --keyword "timeout" --limit 20
# 工具名: log_search

# 搜索数据库相关错误
python cli/query.py log search --task <task_id> --service <service_name> --keyword "slow query" --limit 20
# 工具名: log_search
```

---

## 3. 典型故障模式速查

| 延迟特征 | 根因类型 | 关键证据 |
|---------|---------|---------|
| 延迟 100x+，单 span 占 99% | 外部依赖不可用（Redis/DB） | trace: 单 span 主导; log: connection refused/timeout |
| 延迟 5-10x，DB span > 80% | 慢 SQL | trace: DB span 耗时主导; metric: DB 服务延迟同步升高 |
| P99 高但 avg 正常 | 间歇性慢请求 | trace: 仅部分 trace 慢; slow_count 尖峰 |
| 所有请求同时变慢 + 多服务 | 节点资源瓶颈 | metric: 节点 CPU/IO 高; 多服务同步退化 |
| 延迟升高 + error_rate 也升高 | 下游服务部分故障 | trace: 错误 span + 慢 span 混合; 下游 metric 异常 |
| 延迟抖动（忽高忽低） | GC 停顿 / 资源竞争 | metric: GC 相关指标; 延迟曲线呈锯齿状 |

---

## 4. 根因收敛规则

1. **延迟瓶颈在调用链的最深端**: 如果 A 调用 B 延迟高，B 调用 C 延迟高，C 无异常下游 -> C 是根因
2. **区分自身延迟 vs 下游延迟**: `exclusive_duration = 总延迟 - 子调用延迟`，自身延迟高才是自身问题
3. **资源饱和是间接证据**: CPU/内存高可以解释延迟升高，但需要同时存在 trace 证据证明是资源导致的延迟（而非流量增加导致的正常排队）
4. **外部依赖延迟传递**: Redis 不可用导致依赖 Redis 的服务延迟升高，再级联到上游，根因在 Redis
