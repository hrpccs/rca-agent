---
kind: sop
---

# SOP: Standard RCA investigation order

The canonical order an SRE (and this agent) follows when investigating an
alert. Each step produces evidence; do not skip ahead. The discipline is to
**narrow scope before going deep**, and to **corroborate across modalities**
before declaring a root cause.

## 1. Alerts — read the trigger

Start from the alert that fired. Capture: alert title/rule, the alerted
**entity** (service, operation, pod, node), the **time window**, and the
**direction/magnitude** of the anomaly (error-rate up, latency up, traffic
down). The alert entity is the entry point for every later query. Note that
alert entities can be partially null (e.g. only an `entity_id` with no name) —
resolve the entity against the topology in step 2.

## 2. Topology — establish the blast radius

Resolve the alerted entity and walk its neighborhood (1–2 hops) to see which
upstream callers and downstream dependencies are in scope. This converts
"checkout is failing" into "checkout calls cart, currency, payment, email,
shipping, product-catalog, flagd; it is called by frontend". The topology
tells you where to look next and which entities can plausibly explain the
symptom.

## 3. Traces — localize the failure/latency

Fetch traces through the alerted operation in the window. For a latency alert,
find the **slowest span** (it names where time is spent). For an error alert,
find the **erroring span** (`status_code: ERROR`) and its message. Walk
parent/child spans to see whether the cost/failure is introduced locally or by
a downstream call. Traces give the most specific localization of any modality.

## 4. Metrics — confirm the hypothesis on resources

With a suspect entity/span, confirm on resource metrics: CPU, memory, DB
connections/pools, queue depth, GC, retries, network retransmits. A hypothesis
from traces should be **visible** in metrics — e.g. "the slow DB call" should
correlate with saturated DB connections or a slow-query signature. Compare the
window against baseline (previous hour/day) to confirm it is a real shift, not
noise.

## 5. Logs — find the smoking gun

Pull `ERROR`/`WARN` logs from the suspect entity in the window. Logs give the
explicit failure: a stack trace, an `OOMKilled`, a `connection refused`, a
timeout message, a bad SQL statement. This is usually the most direct evidence
for the mechanism.

## 6. Events — establish the change

Check Kubernetes and change events around the alert start: `Warning` events
(BackOff, OOMKilled, FailedScheduling, Evicted), deploys/rollouts, config /
feature-flag changes, migrations. Most incidents have a **change** in the
window; finding it answers the "what triggered this" half of the root cause.

## 7. Hypothesis — synthesize

Combine the evidence into a root-cause hypothesis: **entity** + **fault type**
+ **mechanism** + **triggering change**. Corroborate across at least two
modalities (e.g. a slow span + a saturated resource metric; an erroring span
+ a log stack trace; a k8s event + a deploy marker). Assign a confidence
(0..1) reflecting how well the evidence supports the hypothesis.

## 8. Root cause — conclude

State the final root cause with: a 1–3 sentence summary, the entity
references, the fault type (e.g. `k8s.oom`, `db.slow_query`,
`deploy.regression`, `config.error`), the evidence (pointers to the
steps/observations that support it), the confidence, contributing factors, and
recommended actions. Separate the root cause from contributing factors and from
actions — they answer different questions (why / what helped / what to do).

## Principles

- **Narrow before you go deep.** Steps 1–3 localize; steps 4–6 confirm. Going
  straight to logs without localizing wastes the alert window.
- **Corroborate across modalities.** A single signal can mislead; two
  independent signals agreeing is strong evidence.
- **Always ask "what changed."** The triggering change is half the root cause.
- **Separate cause from symptom.** The first failing component is often a
  victim; follow the evidence to the originating entity.
