# 流量下跌诊断 SOP

> 触发条件: 告警标题含 "流量下跌" / "traffic" / "drop" / "request_count 下降" / "QPS 下降" 等关键词。

---

## 1. 诊断决策树

```
告警: 服务流量下跌
  |
  +-- 1. 确认下跌范围
  |     |
  |     +-- 仅单个服务流量下跌
  |     |     -> 该服务自身问题（readiness 失败 / Pod 异常）
  |     |     -> 走分支 A: 单服务排查
  |     |
  |     +-- 多个服务同时流量下跌
  |     |     -> 共性依赖故障（节点 / 网络 / DNS）
  |     |     -> 走分支 B: 基础设施排查
  |     |
  +-- 2. 检查 Pod 状态和 readiness 探针
  |     |
  |     +-- Pod Running 但流量 = 0
  |     |     -> readiness probe 失败，LB 无法路由
  |     |
  |     +-- Pod 非 Running（Pending / CrashLoop）
  |     |     -> 转入 05_pod_crash SOP
  |     |
  |     +-- Pod 状态正常
  |           -> 检查上游调用方流量和节点级问题
  |
  +-- 3. 检查节点级问题
  |     |
  |     +-- 多服务共享同一节点 + 节点异常
  |     |     -> 节点资源瓶颈（磁盘 IO / CPU / 网络）
  |     |
  |     +-- 无节点级问题
  |           -> 检查上游服务是否停止调用
  |
  +-- 4. 日志佐证
```

---

## 2. 工具调用序列

### 标准序列

```
context_build -> metric_query -> metric_aggregate -> event_k8s -> log_search
```

### Step 1: 确认流量下跌范围

```bash
# 查询告警服务的请求量
python cli/query.py metric query --task <task_id> --entity-id <entity_id> --name request_count
# 工具名: metric_query

# 窗口 vs 基线对比
python cli/query.py metric compare --task <task_id> --entity-id <entity_id> --name request_count
# 工具名: metric_compare

# 聚合查询所有服务流量（判断是否多服务同时下跌）
python cli/query.py metric aggregate --task <task_id> --name request_count --group-by service --agg sum --limit 20
# 工具名: metric_aggregate
```

**判断规则**:
- 仅单个服务下跌 -> 该服务自身问题
- 多服务同时下跌 -> 共性基础设施问题
- 下跌同时 error_rate 升高 -> 流量因为错误而减少（错误率问题，非流量问题）

### Step 2: 检查 K8s 事件

```bash
# 查询关键事件
python cli/query.py event critical --task <task_id>
# 工具名: event_critical

# 查询特定 Pod 事件
python cli/query.py event k8s --task <task_id> --namespace <namespace>
# 工具名: event_k8s
```

**重点关注**:
- `Unhealthy` (Readiness probe failed) -> Pod 无法接收流量
- `Killing` / `OOMKilled` -> Pod 被杀
- `FailedScheduling` -> 调度失败
- `disk pressure` / `memory pressure` -> 节点资源压力

### Step 3: 检查 Pod 状态

```bash
# 搜索 Pod 实体
python cli/query.py entity search --task <task_id> --keyword <service_name> --type k8s.pod
# 工具名: entity_search

# 查询 Pod 资源指标
python cli/query.py metric query --task <task_id> --entity-id <pod_entity_id> --name cpu_usage
python cli/query.py metric query --task <task_id> --entity-id <pod_entity_id> --name memory_usage
# 工具名: metric_query
```

### Step 4: 检查上游调用方

```bash
# 获取拓扑（上游调用方）
python cli/query.py topo neighbors --task <task_id> --entity-id <entity_id> --direction upstream --depth 1
# 工具名: topo_neighbors

# 查询上游服务流量
python cli/query.py metric query --task <task_id> --entity-id <upstream_entity_id> --name request_count
# 工具名: metric_query
```

### Step 5: 日志佐证

```bash
# 搜索相关日志
python cli/query.py log search --task <task_id> --service <service_name> --keyword "refused" --limit 20
# 工具名: log_search

# 搜索 readiness/健康检查相关日志
python cli/query.py log search --task <task_id> --service <service_name> --keyword "health" --limit 20
# 工具名: log_search
```

---

## 3. 典型故障模式速查

| 流量下跌特征 | 根因类型 | 关键证据 |
|------------|---------|---------|
| 单服务流量跌至 0，Pod Running | readiness probe 失败，LB 无法路由 | event: Unhealthy/Readiness failed; metric: 流量=0 |
| 单服务流量下跌，Pod 非 Running | Pod CrashLoop / OOMKilled | event: BackOff/OOMKilled; 转入 05_pod_crash |
| 多服务同时下跌，无 error_rate 升高 | 节点资源瓶颈（磁盘 IO / 网络） | metric: 多服务同步退化; event: disk pressure |
| 流量下跌 + 上游也下跌 | 上游服务停止调用（级联下跌） | topo: 上游依赖关系; metric: 上游流量也下跌 |
| 流量下跌 + error_rate 升高 | 错误导致请求失败重试（实际是错误率问题） | metric: error_rate 同步升高; 转入 02_error_rate_spike |
| 流量下跌 + Deployment 副本不足 | HPA 异常 / Deployment 配置错误 | metric: 可用副本 < 期望副本; event: FailedScheduling |

---

## 4. 根因收敛规则

1. **先判断范围**: 单服务 vs 多服务，决定了排查方向是应用层还是基础设施层
2. **readiness 失败 != 服务挂了**: Pod 可能还在运行，只是健康检查失败导致流量无法路由
3. **流量下跌可能是果不是因**: 先排除 error_rate 升高导致流量下跌的情况，优先处理错误率问题
4. **级联下跌追踪**: 如果上游也下跌，继续往上游追踪，找到第一个流量正常但下游下跌的实体
