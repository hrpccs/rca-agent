---
kind: domain_fact
---

# Network latency / RTT rise

A rise in network round-trip time (RTT) or request latency is rarely caused by
the network alone. When you see RTT or end-to-end latency climb, treat it as a
symptom and check the three classic producers in order: **CPU contention**,
**genuine network latency**, and **queueing**.

## Common producers

### 1. CPU contention (most common)
When a node or container is CPU-saturated, the kernel scheduler delays both
interrupt handling and user-space wakeups. The packet leaves the wire on time
but the application is slow to accept/process it, so measured RTT rises even
though the network path is healthy.
- **Signals:** high CPU usage / high run-queue length / high iowait on the
  serving node; latency correlates with CPU%, not with packet loss.
- **Confirm:** check node/pod CPU and throttle metrics (`container_cpu_usage`,
  `node_cpu_seconds_total`, CPU throttling) on the node hosting the slow
  endpoint.

### 2. Genuine network latency
An actual propagation or queuing delay in the network path: cross-AZ/cross-
region hops, a saturated NIC or switch port, TCP retransmits, or a
misconfigured/oversubscribed CNI.
- **Signals:** RTT rises **without** CPU saturation on either endpoint; packet
  loss or TCP retransmits (`tcp_retranssegs`) along the path; the rise affects
  all traffic on the link, not one process.
- **Confirm:** compare RTT from multiple source pods to the same target; a
  uniform rise points at the path, a single-source rise points at that source.

### 3. Queueing / backlog
The server accepts the connection but spends time queued waiting for a worker
(thread, goroutine, connection pool, or kernel accept queue). The client
perceives this as latency; the server CPU may look idle because it is blocked
on a saturated downstream resource (DB, lock, pool).
- **Signals:** rising connection-queue depth / `listen backlog` drops; pool
  utilization at 100%; latency rises while CPU is **low**; p99 rises faster
  than p50 (tail latency from queueing).
- **Confirm:** check pool/queue metrics and downstream saturation (DB
  connections, thread pools, accept queue overflows).

## Triage order

1. Is CPU saturated on the node/pod serving the slow endpoint? → contention.
2. Are downstream pools/queues full or is a DB/lock saturated? → queueing.
3. Is RTT uniformly up across sources with retransmits/loss on the path? →
   genuine network latency.

A latency rise that tracks CPU is contention; one that tracks downstream
saturation is queueing; only a uniform, loss-accompanied rise across all
sources is genuine network latency. SRE rule of thumb: **blame the network
last.**
