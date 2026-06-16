---
kind: runbook
---

# Runbook: Kubernetes CrashLoopBackOff

A pod's container is repeatedly crashing and being restarted by kubelet. The
`CrashLoopBackOff` state means kubelet has backed off exponentially after
several failed restarts. This is a **symptom**, not a root cause — the
container exits for a reason, and the reason is in one of four places: the
**events**, an **OOM kill**, a **recent deploy**, or a **config** problem.

## Triage order

### 1. Check the pod events

```
kubectl describe pod <pod> -n <namespace>
kubectl get events -n <namespace> --sort-by=.lastTimestamp | tail -30
```

Look at `Last State` and the exit code/message:
- **Exit 137** (128+9 = SIGKILL) → almost always an **OOM kill** (see step 2)
  or an explicit `kubectl delete`/eviction.
- **Exit 1 / non-zero app code** → the application crashed (uncaught
  exception, failed healthcheck, bad config). The reason is in the logs (step 3).
- **Exit 0** → the container ran to completion but was expected to stay up
  (wrong command/entrypoint, a Job-style binary in a Deployment, or a
  misconfigured liveness probe killing a healthy pod).

Also check for `FailedScheduling`, `Evicted`, `Liveness probe failed`, and
`Readiness probe failed` events.

### 2. Rule out OOM

An OOM kill is the single most common CrashLoopBackOff cause:

```
kubectl describe pod <pod> -n <namespace> | grep -A5 "Last State"
#   Reason:   OOMKilled / Exit Code: 137
```

- Confirm with node memory pressure: is the node overcommitted? Did the pod's
  memory limit get lowered?
- Check memory metrics for the pod in the alert window — a steady climb to the
  limit is a leak; a sudden jump is a load spike or large allocation.
- **Fix:** raise the memory request/limit, or fix the leak / reduce working
  set. Do not just raise the limit if there is a leak.

### 3. Check recent deploys

Was the crashing image/version deployed minutes before the loop started?

```
kubectl rollout history deployment/<dep> -n <namespace>
kubectl logs <pod> -n <namespace> --previous      # logs from the crashed container
```

- A bad image (broken binary, missing env, bad migration) will crash every pod
  in the rollout. Roll back to the previous working revision to confirm:
  `kubectl rollout undo deployment/<dep> -n <namespace>`.
- A crash that starts immediately after a deploy, across all replicas, is the
  deploy. Prioritize rollback over debugging in place.

### 4. Check config / env / dependencies

The container starts but exits because it cannot function:

- **Missing/invalid config:** a required env var is unset, a configmap/secret
  is missing or was renamed, a mounted file has bad syntax.
- **Dependency unavailable at startup:** the app fails its startup check
  because a DB, cache, or peer service is unreachable (connection refused,
  DNS error, TLS handshake failure).
- **Bad command/args:** a changed entrypoint, a flag that no longer exists, a
  wrong port/path.

```
kubectl logs <pod> -n <namespace> --previous | head -50
```

The crash log almost always names the missing config, the bad connection, or
the offending flag.

## Confirming the root cause

Tie the loop to one of the four causes with evidence:
- **OOM** → `OOMKilled` in Last State + memory metric at the limit.
- **Deploy** → crash start == rollout time; rollback stops it.
- **Config** → the crash log names the missing/invalid input.
- **Probe** → `Liveness/Readiness probe failed` events precede the restart,
  and the container's own logs are healthy.

State the root cause as **entity (pod/workload) + fault type
(`k8s.pod_crashloop` / `k8s.oom` / `deploy.regression` / `config.error`) +
triggering change**, with the corroborating evidence. List the immediate fix
(raise limit / rollback / fix config) and the preventive action (memory leak
fix, deploy guard, config validation) separately.
