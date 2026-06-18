# Trace 链路深度分析 SOP

> 用途: 当需要通过链路追踪数据定位错误源头或延迟瓶颈时使用本 SOP。
> 通常在 02_error_rate_spike 或 03_latency_spike 的 Step 2 中调用。

---

## 1. 四阶段分析流程

```
Phase 1: 搜索 Trace
  |  -> trace_search / trace_batch-diagnose
  v
Phase 2: 获取 Span 详情
  |  -> trace_get / trace_tree
  v
Phase 3: 构建 Span 树
  |  -> trace_tree / trace_diagnose
  v
Phase 4: 诊断归因
     -> trace_diagnose / trace_batch-diagnose
```

---

## 2. Phase 1: 搜索 Trace

### 按症状类型选择搜索条件

| 症状 | 搜索条件 | 示例命令 |
|------|---------|---------|
| 错误率升高 | `--condition error` | `trace search --condition error --service checkout --limit 5` |
| 延迟升高 | `--condition high_latency` | `trace search --condition high_latency --service cart --limit 5` |
| 不确定 | `--condition any` | `trace search --condition any --service frontend --limit 10` |

```bash
# 搜索错误 trace
python cli/query.py trace search --task <task_id> --condition error --service <service_name> --limit 5
# 工具名: trace_search

# 批量诊断（推荐：自动归因多条 trace）
python cli/query.py trace batch-diagnose --task <task_id> --condition error --service <service_name> --limit 5
# 工具名: trace_batch-diagnose
```

---

## 3. Phase 2: 获取 Span 详情

```bash
# 获取 trace 的所有 span
python cli/query.py trace get --task <task_id> --trace-id <trace_id>
# 工具名: trace_get
```

**输出字段关注**:

| 字段 | 用途 |
|------|------|
| `span_id` | Span 唯一标识 |
| `parent_span_id` | 父 Span（构建调用树） |
| `service_name` | 所属服务 |
| `operation_name` | 操作名称（接口/RPC） |
| `duration_ns` | 持续时间（纳秒） |
| `status_code` | 状态码（2=ERROR） |
| `attributes` | 附加属性（HTTP 状态码、异常信息等） |

---

## 4. Phase 3: 构建 Span 树

```bash
# 构建 span 树（自动计算 exclusive_duration 等）
python cli/query.py trace tree --task <task_id> --trace-id <trace_id>
# 工具名: trace_tree
```

**树结构分析**:

- **关键路径**: 从根 Span 沿最大子 Span 递归到底
- **exclusive_duration**: `总耗时 - 所有子调用耗时`（自身真正消耗的时间）
- **瓶颈 Span**: `exclusive_duration` 最大的 Span

---

## 5. Phase 4: 诊断归因

```bash
# 单条 trace 诊断（自动归因）
python cli/query.py trace diagnose --task <task_id> --trace-id <trace_id>
# 工具名: trace_diagnose
```

### 5.1 错误传播分析

```
错误 trace 诊断:
  1. 找到 status_code=2 或有 exception 属性的 Span
  2. 从最深层错误 Span 回溯到根 Span -> 错误传播路径
  3. 最深层错误 Span = 根因 Span
```

**常见错误模式**:

| 错误模式 | 判定条件 | 说明 |
|---------|---------|------|
| **下游不可达** | span duration=0ms + error status + connection refused | 连接从未建立 |
| **应用 Bug** | HTTP 500 + TypeError/NullPointerException | 代码级错误 |
| **下游超时** | 上游 span duration 下游 span duration，下游无子调用 | leaf 阻塞级联 |
| **限流拒绝** | HTTP 403/429 + 下游 error_rate > 90% | 下游限流 |
| **条件性失败** | 仅特定接口/参数的 Span 报错 | 非所有请求失败 |

### 5.2 延迟瓶颈分析

```
延迟 trace 诊断:
  1. 计算 exclusive_duration = duration - sum(child.duration)
  2. 找 exclusive_duration 最大的 Span -> 主要瓶颈
  3. 检查关键路径上的 Span 耗时占比
```

**常见延迟模式**:

| 延迟模式 | 判定条件 | 说明 |
|---------|---------|------|
| **单点瓶颈** | 单个 Span 占总 duration > 80% | 如 DB 慢查询 |
| **外部依赖不可用** | 单下游 Span 占总延迟 99%+ | 如 Redis connection refused 导致 200s+ |
| **串行瓶颈** | 关键路径上多个 Span 累计耗时高 | 多次串行调用 |
| **间歇性慢请求** | 仅部分 trace 慢，大部分正常 | 长尾延迟 |

### 5.3 异常模式识别

| 模式 | 判定条件 |
|------|---------|
| **重试风暴** | 同一服务/端点的多个 Span 具有相似参数 + 短时间间隔 |
| **级联超时** | 上游 Span duration 约等于 下游 Span duration，下游无子调用 |
| **串行瓶颈** | 关键路径上单个 Span 占总 duration > 80% |

---

## 6. 使用规则

1. **先 batch-diagnose 再单条深入**: batch-diagnose 能自动归因多条 trace，找到共性根因后再用 diagnose 深入确认
2. **查 2-3 条 trace 确认模式**: 不要只看一条 trace 就下结论，至少确认 2-3 条 trace 的根因指向同一个实体
3. **错误和延迟可能共存**: 先处理错误（error trace），延迟问题可能因错误导致
4. **trace 无数据时不要反复重试**: 如果 trace_search 返回空，检查时间和条件是否正确，不要扩大范围盲目搜索
