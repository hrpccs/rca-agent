# 日志分析 SOP

> 用途: 通过日志数据确认根因假设，提取具体错误消息和异常堆栈。
> 通常在诊断流程的最后阶段（Stage 6）使用，作为三角验证的佐证。

---

## 1. 日志查询策略

### 1.1 查询路径选择

```
需要日志证据
  |
  +-- 知道具体服务名
  |     -> log_search --service <service_name> --keyword <keyword>
  |
  +-- 知道具体 Pod 名
  |     -> log_search --pod-name <pod_name> --keyword <keyword>
  |
  +-- 不知道查什么
        -> log_error-summary --service <service_name>
           （自动聚合错误类型，发现最常见的错误）
```

---

## 2. 工具调用

### 2.1 关键词搜索

```bash
# 按服务 + 关键词搜索
python cli/query.py log search --task <task_id> --service <service_name> --keyword "error" --limit 30
# 工具名: log_search

# 按 Pod 名搜索（Pod 崩溃场景）
python cli/query.py log search --task <task_id> --pod-name <pod_name> --keyword "exception" --limit 30
# 工具名: log_search

# 搜索超时相关日志
python cli/query.py log search --task <task_id> --service <service_name> --keyword "timeout" --limit 20
# 工具名: log_search

# 搜索连接拒绝相关日志
python cli/query.py log search --task <task_id> --service <service_name> --keyword "connection refused" --limit 20
# 工具名: log_search
```

### 2.2 错误摘要（推荐优先使用）

```bash
# 自动聚合错误类型（不需要猜关键词）
python cli/query.py log error-summary --task <task_id> --service <service_name>
# 工具名: log_error-summary
```

---

## 3. 关键词模式速查

根据不同故障类型使用不同的搜索关键词：

| 故障类型 | 搜索关键词 | 期望看到的日志内容 |
|---------|-----------|----------------|
| **连接失败** | `connection refused` / `ECONNREFUSED` / `unavailable` | 下游服务不可达 |
| **超时** | `timeout` / `timed out` / `Timeout` | 请求处理超时 |
| **OOM** | `OutOfMemoryError` / `OOM` / `heap space` | 内存溢出 |
| **应用 Bug** | `NullPointerException` / `TypeError` / `StackOverflow` | 代码级异常 |
| **启动失败** | `failed to start` / `CrashLoop` / `exit code` | 容器启动错误 |
| **数据库问题** | `slow query` / `SQL` / `HikariPool` / `getConnection` | 数据库慢查询/连接池 |
| **Redis/缓存** | `redis` / `cache miss` / `connection reset` | 缓存不可用 |
| **限流** | `rate limit` / `429` / `forbidden` / `quota` | 被限流 |
| **通用错误** | `error` / `exception` / `fail` | 广泛匹配 |

---

## 4. 日志解读规则

### 4.1 错误消息提取

从日志中提取以下关键信息：

1. **异常类型**: 如 `NullPointerException`, `TypeError`, `TimeoutException`
2. **错误消息**: 如 `fetch failed`, `connection refused`, `heap space`
3. **调用栈**: 前几行堆栈（能定位到具体的代码路径）
4. **时间戳**: 与告警窗口对齐
5. **关联信息**: trace_id, request_id, span_id 等

### 4.2 日志与 Trace 交叉验证

```
trace 诊断结果: 根因在 checkout 服务，HTTP 500 错误
  |
  +-- 日志验证: log_search --service checkout --keyword "error"
  |     |
  |     +-- 找到 "NullPointerException: cart is null"
  |     |     -> 确认: 应用代码 Bug，NPE 导致 500
  |     |
  |     +-- 找到 "connection refused: redis:6379"
  |     |     -> 确认: Redis 不可用导致 checkout 失败
  |     |
  |     +-- 找到 "timeout: cart service did not respond within 5000ms"
  |           -> 确认: cart 服务超时导致 checkout 失败
```

### 4.3 错误摘要使用策略

`log_error-summary` 会自动聚合错误类型，返回出现次数最多的错误模式。使用场景：

- 不知道该搜什么关键词时，先用 error-summary 发现主要错误类型
- 确认错误集中度（是单一错误类型还是多种错误混合）
- 与 trace 诊断结果交叉验证

---

## 5. 使用规则

1. **日志是佐证不是起点**: 先通过 metric 和 trace 形成假设，再用日志确认。不要一开始就海量搜索日志
2. **限定服务/Pod 范围**: 优先按服务名或 Pod 名搜索，避免全量扫描
3. **优先用 error-summary**: 不知道搜什么时，先看错误摘要，再有针对性地搜索
4. **时间对齐**: 日志中的错误时间必须在告警窗口内，窗口外的错误不相关
5. **日志无数据不等于无故障**: 可能是日志采样导致未捕获到相关日志，此时依赖 metric 和 trace 的证据
