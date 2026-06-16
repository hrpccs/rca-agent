---
kind: domain_fact
---

# Error-rate / latency anomaly triage

General principles for triaging an alert that fires on an error-rate or latency
SLO/budget burn. The goal of the first 5 minutes is to **localize** the failure
(scope, blast radius, change) before diving into a single component.

## Step 1 — Confirm the alert is real

- Is the signal a single noisy datapoint or a sustained shift? Compare the
  alert window against the baseline (previous hour/day). A step change that
  persists is real; a single spike is often a scrape/ingest blip.
- Corroborate with a second modality: an error-rate alert should also show
  raised error counts in logs and/or `ERROR` span status codes in traces.
  An alert with no corroborating signal is suspect.

## Step 2 — Localize the scope

Narrow from "something is slow/failing" to a specific entity:

- **Which service/operation?** The alert's `entity` (service, operation,
  endpoint) is the starting point. Walk the topology one hop out to see which
  upstream/downstream services are in the blast radius.
- **Which dimension changed?** Error-rate up vs. latency up vs. both:
  - Error-rate up with flat latency → a hard failure (crash, bad deploy, bad
    config, dependency down).
  - Latency up with flat error-rate → saturation/resource contention; errors
    often follow once timeouts kick in.
  - Both up → saturation cascading into timeouts/errors, or a bad deploy.

## Step 3 — Look for a recent change (the "what changed" question)

Most incidents are caused by a change in the last minutes-to-hours:

- Recent **deploy** of the alerted service or a direct dependency.
- **Config** change (feature flag, limit, env var) — check `flagd`/config
  flags and recent configmap changes.
- **Load** change (traffic spike, new client, retry storm).
- **Dependency** change (DB migration, schema, upstream API change).

Time-box this: if a deploy/config change coincides with the alert start, it is
the leading hypothesis.

## Step 4 — Trace, then metrics, then logs

1. **Traces:** find the slowest/failing trace through the alerted operation.
  The slowest span localizes the latency; the erroring span localizes the
  failure. Walk parent/child spans to see where time/failure is introduced.
2. **Metrics:** confirm the localized hypothesis on resource metrics — CPU,
  memory, DB connections, pool utilization, queue depth, GC, retries — on the
  suspect entity and its node.
3. **Logs:** pull `ERROR`/`WARN` logs from the suspect entity in the alert
  window for the smoking gun (stack trace, OOM kill, connection refused,
  timeout message).

## Step 5 — Events / deploy markers

Check k8s events (`Warning` BackOff, OOMKilled, FailedScheduling, Evicted) and
deploy/change markers around the alert start time. An `OOMKilled` or a
`CrashLoopBackOff` is often the whole answer.

## Step 6 — Form the root-cause hypothesis

State the root cause as: **entity** + **fault type** + **mechanism** + **the
change that triggered it**, backed by 2-3 pieces of evidence (a metric
anomaly, a slow/error span, a log line, a k8s event). Assign a confidence
(0..1) and list contributing factors and recommended actions separately from
the root cause.

## Pitfalls

- **Don't chase the first error you see.** Corroborate; the first failing
  component is often a victim of an upstream/downstream problem.
- **Don't conflate correlation and causation.** A metric moving at the same
  time as the alert is a clue, not a cause — tie it to a mechanism.
- **Mind retries.** A retry storm from an upstream client can look like a load
  spike and amplify the original failure. Check retry/backoff behavior.
