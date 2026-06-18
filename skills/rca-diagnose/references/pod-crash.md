# Pod 崩溃/被杀/OOM 诊断 SOP

> 触发条件: 告警标题含 "Pod" / "OOM" / "CrashLoop" / "重启" / "Killing" / "BackOff" 等关键词。

---

## 1. 诊断决策树

```
告警: Pod 崩溃/重启/OOM
  |
  +-- 1. event_critical 确认事件类型
  |     |
  |     +-- OOMKilled
  |     |     -> 内存溢出被杀
  |     |     -> 走分支 A: 内存问题排查
  |     |
  |     +-- Killing + Created + Started（非 OOM）
  |     |     -> Pod 被外部杀掉（人为/调度器）
  |     |     -> 走分支 B: Pod Kill 排查
  |     |
  |     +-- BackOff / CrashLoopBackOff
  |     |     -> 应用启动失败，反复重启
  |     |     -> 走分支 C: 启动失败排查
  |     |
  |     +-- Evicted
  |           -> 节点资源压力驱逐
  |           -> 走分支 D: 节点资源排查
  |
  +-- 2. metric_compare 检查资源趋势
  |     |
  |     +-- 内存使用率持续升高直至被杀
  |     |     -> 确认内存泄漏/OOM
  |     |
  |     +-- CPU 使用率飙升后 Pod 重启
  |     |     -> 可能 CPU 饱和导致健康检查失败
  |     |
  |     +-- 资源指标正常但 Pod 重启
  |           -> 应用层错误（代码 Bug/配置错误）
  |
  +-- 3. log_search 查找崩溃原因
        |
        +-- OOM / OutOfMemoryError
        |     -> 内存溢出（确认堆内/堆外/容器级别）
        +-- NullPointerException / StackOverflow
        |     -> 应用代码级 Bug
        +-- 配置错误 / 启动失败
              -> 配置问题导致无法启动
```

---

## 2. 工具调用序列

### 标准序列

```
context_build -> event_critical -> metric_compare -> log_search
```

### Step 1: 确认 K8s 事件类型

```bash
# 查询关键事件
python cli/query.py event critical --task <task_id>
# 工具名: event_critical

# 查询特定 Pod 的 K8s 事件
python cli/query.py event k8s --task <task_id> --pod-name <pod_name> --namespace <namespace>
# 工具名: event_k8s

# 查询特定原因的事件
python cli/query.py event k8s --task <task_id> --reason OOMKilled
python cli/query.py event k8s --task <task_id> --reason Killing
# 工具名: event_k8s
```

**事件类型解读**:

| 事件组合 | 故障类型 | 后续路径 |
|---------|---------|---------|
| `OOMKilled` | 内存溢出被杀 | 分支 A: 内存排查 |
| `Killing` + `Created` + `Started` | Pod 被杀后重建 | 分支 B: Pod Kill |
| `BackOff` (反复出现) | CrashLoopBackOff | 分支 C: 启动失败 |
| `Evicted` + `MemoryPressure` / `DiskPressure` | 节点资源压力 | 分支 D: 节点资源 |
| `FailedScheduling` | 调度失败 | 检查资源请求 |

### Step 2: 检查资源趋势

```bash
# 查询 Pod 内存使用率
python cli/query.py metric query --task <task_id> --entity-id <pod_entity_id> --name memory_usage
# 工具名: metric_query

# 查询 Pod CPU 使用率
python cli/query.py metric query --task <task_id> --entity-id <pod_entity_id> --name cpu_usage
# 工具名: metric_query

# 窗口 vs 基线对比
python cli/query.py metric compare --task <task_id> --entity-id <pod_entity_id> --name memory_usage
# 工具名: metric_compare
```

### Step 3: 检查上游影响

```bash
# 查询上游服务的错误率（Pod 崩溃会导致上游 connection refused）
python cli/query.py metric compare --task <task_id> --entity-id <upstream_entity_id> --name error_rate
# 工具名: metric_compare

# 搜索上游的错误 trace
python cli/query.py trace batch-diagnose --task <task_id> --condition error --service <upstream_service> --limit 5
# 工具名: trace_batch-diagnose
```

**关键信号**:
- 上游 trace 出现 `connection refused` / `unavailable` + `span duration=0ms` -> 证实 Pod 不可达
- 上游 error_rate 同步升高 -> Pod 崩溃已级联影响上游

### Step 4: 日志佐证

```bash
# 搜索崩溃相关日志
python cli/query.py log search --task <task_id> --pod-name <pod_name> --keyword "error" --limit 30
# 工具名: log_search

# 搜索 OOM 相关日志
python cli/query.py log search --task <task_id> --pod-name <pod_name> --keyword "OutOfMemory" --limit 20
# 工具名: log_search

# 搜索崩溃前最后的日志（不限定关键词）
python cli/query.py log search --task <task_id> --pod-name <pod_name> --limit 50
# 工具名: log_search
```

---

## 3. 典型故障模式速查

| 故障类型 | 事件特征 | 关键证据 |
|---------|---------|---------|
| **内存泄漏 -> OOMKilled** | `OOMKilled` 事件; 内存曲线单调上升 | event: OOMKilled; metric: memory 持续上升; log: OutOfMemoryError |
| **Pod 被杀（非 OOM）** | `Killing` 事件; 突然开始非渐进 | event: Killing; trace: 上游 gRPC UNAVAILABLE + duration=0ms |
| **CrashLoopBackOff** | `BackOff` 反复出现 | event: BackOff; log: 启动错误堆栈; 应用无法完成初始化 |
| **配置错误导致启动失败** | `BackOff` + 启动日志报错 | log: 配置加载失败/端口冲突/依赖不可达 |
| **节点驱逐** | `Evicted` + `MemoryPressure`/`DiskPressure` | event: Evicted; 同节点其他 Pod 也被驱逐 |
| **资源超限被杀** | `OOMKilled` + memory_usage 接近 limit | metric: memory 接近 limit; 资源配额不合理 |

---

## 4. 根因收敛规则

1. **事件类型决定排查方向**: OOMKilled -> 内存; BackOff -> 启动失败; Killing -> 外部杀进程
2. **Pod 崩溃是上游报错的根因**: 如果上游服务出现 connection refused 错误，检查下游 Pod 是否崩溃
3. **区分容器 OOM 和应用 OOM**: 容器 OOMKilled（Pod 被杀）vs 应用 OutOfMemoryError（进程内异常），两者可能不同时出现
4. **时间对齐**: Pod 崩溃时间（事件时间戳）必须与上游报错时间对齐，才能建立因果关系
5. **不反向归因**: "Pod 崩溃是因为上游流量大" 除非有明确的证据（如 OOM 因为流量导致内存暴涨）
