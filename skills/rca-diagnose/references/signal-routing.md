# 信号路由决策 SOP (Signal Routing)

> 元技能：根据告警症状将诊断路由到正确的 SOP 路径。
> 在发出任何查询命令之前，先完成本路由决策。

---

## 1. 角色定位

**角色**: RCA 信号路由引擎

**核心职责**: 解析告警标题和内容，提取六维信号模型，决策进入哪条诊断 SOP。

---

## 2. 六维信号分析模型

在发出第一个查询命令之前，必须确定以下六个维度：

| 维度 | 含义 | 示例 |
|------|------|------|
| **intent** | 查询意图：查找 / 计算 / 诊断 / 证据 | "错误率为什么升高" -> 诊断 |
| **signal** | 信号域：metric / trace / log / event / topology | "错误次数" -> metric + trace |
| **scope** | 调查范围：精确实体 / 实体族 / 关系范围 | "checkout 服务" -> 精确实体 |
| **shape** | 期望输出形态：标量 / 时序 / 排名 / 证据包 | "趋势" -> 时序 |
| **anchors** | 从告警上下文中复用的精确值 | task_id, entity_id, alert_window |
| **blockers** | 阻塞当前路由的缺失信息 | 不知道 entity_id -> 需先 entity search |

---

## 3. 症状 -> SOP 路由决策树

根据告警标题/内容中的关键词，路由到对应的诊断 SOP：

```
告警输入
  |
  +-- 关键词含 "错误次数" / "异常" / "error_count" / "error_rate" / "5xx"
  |     -> 01_rca_framework 阶段 1-2 后进入 02_error_rate_spike
  |
  +-- 关键词含 "超时" / "响应时间" / "latency" / "timeout" / "P99" / "RT"
  |     -> 01_rca_framework 阶段 1-2 后进入 03_latency_spike
  |
  +-- 关键词含 "流量下跌" / "traffic" / "drop" / "request_count 下降"
  |     -> 01_rca_framework 阶段 1-2 后进入 04_traffic_drop
  |
  +-- 关键词含 "Pod" / "OOM" / "CrashLoop" / "重启" / "Killing"
  |     -> 01_rca_framework 阶段 1-2 后进入 05_pod_crash
  |
  +-- 无法明确分类
        -> 进入 01_rca_framework 全阶段通用诊断
```

### 路由规则

1. **先走通用框架的阶段 1-2**（告警确认 + 拓扑发现），再进入专项 SOP
2. 每条路径的工具调用序列不同，但都遵循"量化 -> 定位 -> 验证 -> 佐证"的三角验证原则
3. 如果中途发现信号类型判断错误（如以为是错误率问题实则是延迟问题），允许回退重新路由

---

## 4. 快速分流决策（前 3 条命令内）

```
error_rate 高 + 有下游依赖异常 -> 走 "下游定位" 路径 (02_error_rate_spike)
延迟高 + 无明显错误           -> 走 "资源瓶颈" 路径 (03_latency_spike)
流量骤降 + 多服务同时退化     -> 走 "基础设施故障" 路径 (04_traffic_drop)
error_rate ~100% + HTTP 500   -> 走 "应用 Bug" 路径 (02_error_rate_spike)
Pod 重启/CrashLoop            -> 走 "Pod 崩溃" 路径 (05_pod_crash)
```

---

## 5. 标准入口命令序列

无论进入哪条路径，前两步是固定的：

```bash
# Step 1: 构建诊断上下文（获取告警、实体、时间窗口）
python cli/query.py context build --task <task_id>
# 对应工具名: context_build

# Step 2: 拓扑发现（找到上下游依赖）
python cli/query.py entity search --task <task_id> --keyword <alert_entity>
# 对应工具名: entity_search

python cli/query.py topo neighbors --task <task_id> --entity-id <entity_id> --depth 2
# 对应工具名: topo_neighbors
```

---

## 6. 路由速查表

| RCA 场景 | 路由序列 |
|---------|---------|
| 服务错误率升高 | `context_build` -> `entity_search` -> `metric_compare` -> `trace_batch-diagnose` -> `event_critical` -> `log_search` |
| 服务延迟升高 | `context_build` -> `entity_search` -> `metric_compare` -> `trace_diagnose` -> `metric_aggregate` |
| 流量骤降 | `context_build` -> `entity_search` -> `metric_query` -> `event_k8s` -> `log_search` |
| Pod 重启/OOM | `context_build` -> `entity_search` -> `event_critical` -> `metric_compare` -> `log_search` |
| 级联故障 | `context_build` -> `topo_neighbors` -> `trace_batch-diagnose` -> `event_critical` -> `log_search` |
