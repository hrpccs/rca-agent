---
kind: runbook
---

# Runbook: Latency spike / regression

A service's latency (p50/p95/p99) has jumped above baseline. Unlike a hard
error, latency regressions degrade gradually and cascade into timeouts/errors
if left alone. The fastest path to root cause is to **trace the slowest span**,
then determine whether the cost is **local** (this service) or
**downstream** (a dependency), and finally whether it is **resource
saturation** or a **change**.

## Triage order

### 1. Trace the slowest span

Find a representative slow trace through the alerted operation and locate the
most expensive span:

- The **slowest span** tells you where wall-clock time is spent.
  - If the slowest span is **inside this service** (a local computation, a
    lock, GC, serialization), the regression is local → go to step 3.
  - If the slowest span is a **downstream call** (an RPC/HTTP/db client span
    whose duration dominates), the regression is in that dependency → recurse:
    treat the dependency as the new "this service" and repeat.
- Note the span's `service` and `name` (e.g. `/oteldemo.CartService/GetCart`,
  a SQL query, an HTTP GET). The span name often names the culprit directly.

### 2. Walk upstream and downstream

- **Downstream:** for each dependency the alerted service calls, is that
  dependency's latency also up? A downstream regression propagates upward;
  confirm the dependency is independently slow before blaming the caller.
- **Upstream:** are callers retrying or sending more traffic? A retry storm or
  a load spike from an upstream client amplifies latency through queueing (see
  `domain-network-latency.md` for the queueing pattern). Check call-rate /
  retry metrics for the alerted operation.

### 3. Check resource saturation on the suspect entity

Once you have a suspect service/span, check its resources in the alert window:

- **CPU:** usage / throttling. CPU saturation delays processing → latency
  rises with CPU% (contention pattern).
- **Memory/GC:** high heap or frequent/long GC pauses stall the request loop.
- **DB connections / pools:** pool at 100% means requests queue waiting for a
  connection (queueing pattern) — p99 rises faster than p50.
- **Network:** retransmits/loss only if the path itself is the problem
  (blame the network last).
- **I/O / disk:** slow disk (logs, local cache) or iowait on the node.

A latency rise that tracks a resource metric is saturation; name the resource.

### 4. Look for the triggering change

Latency regressions usually have a trigger in the last minutes-to-hours:

- **Deploy:** a new version with a slower code path, an N+1 query, a missing
  index after a schema/migration change, a heavier serialization.
- **Data shape:** a query that was fast on small data now scans a large table
  (missing index, data growth, a full-table scan). Check the DB slow-query log
  and `EXPLAIN` the suspect query.
- **Load:** a traffic spike, a new high-volume caller, a changed retry/timeout
  policy that now hammers the service.
- **Dependency:** an upstream dependency slowed down (recurse to step 2).

## Confirming the root cause

State the root cause as **entity (service/operation) + fault type
(e.g. `db.slow_query`, `resource.cpu_saturation`, `dependency.regression`) +
the span/resource evidence + the triggering change**, e.g.:

> cart p99 rose to 2.4s (baseline 180ms) because the cart→inventory HTTP call
> dominates; inventory DB connections are saturated by a full-table scan on
> `inventory.orders` introduced by the 05:18 deploy that removed an index.

Assign confidence, list contributing factors (e.g. no circuit breaker, retry
amplification) and recommended actions (restore/add index, add a bulkhead,
cap retries) separately from the root cause.
