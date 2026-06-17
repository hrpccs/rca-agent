"""Pure scoring + structural-metric helpers for the eval runner.

The rca100 benchmark is currently blind (no published ground truth), so these
helpers score what can be measured objectively against either an *optional*
ground-truth entity set (entity-set P/R/F1, fault_type match) or purely over a
:class:`RootCause` (richness / structural metrics). Everything here is pure —
no I/O, no agent, no settings — so :mod:`runner` can call them and tests can
exercise edge cases without touching the LLM.

When the benchmark's prediction_schema / taxonomy is published these functions
are the plug-in point; the runner's output schema does not need to change.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..contracts import RootCause, StepKind

# Any set-like / iterable of predicted + ground-truth entity identifiers.
EntitySet = Iterable[Any]


def _to_set(xs: EntitySet) -> set[Any]:
    """Normalize an entity iterable into a hashable set, skipping empty values.

    The RootCause stores entity_refs as ``list[dict]``; ground truth may be a
    list of ids, names, or dicts. We coerce dict entries to their best
    identifier (entity_name then entity_id) so two refs referring to the same
    entity collapse to one regardless of representation. ``None`` / empty
    string / empty dict entries are dropped — they carry no entity identity.
    """
    out: set[Any] = set()
    for x in xs or ():
        if isinstance(x, dict):
            ident = x.get("entity_name") or x.get("entity_id")
            if ident:
                out.add(ident)
        elif x:  # truthy scalar (non-empty id/name); skips None and ""
            out.add(x)
    return out


def entity_precision(predicted: EntitySet, truth: EntitySet) -> float:
    """Precision of predicted entities vs ground truth (both empty -> 0.0).

    Of the entities the agent named, what fraction were correct. Returns 0.0
    when nothing was predicted (avoids ZeroDivisionError) rather than raising.
    """
    p = _to_set(predicted)
    if not p:
        return 0.0
    t = _to_set(truth)
    tp = len(p & t)
    return tp / len(p)


def entity_recall(predicted: EntitySet, truth: EntitySet) -> float:
    """Recall of predicted entities vs ground truth (either empty -> 0.0).

    Of the true entities, what fraction the agent named. Returns 0.0 when the
    truth set is empty (avoids ZeroDivisionError) rather than raising.
    """
    t = _to_set(truth)
    if not t:
        return 0.0
    p = _to_set(predicted)
    tp = len(p & t)
    return tp / len(t)


def entity_f1(predicted: EntitySet, truth: EntitySet) -> float:
    """Harmonic mean of :func:`entity_precision` / :func:`entity_recall`.

    Returns 0.0 when either set is empty (so precision *or* recall is 0),
    avoiding the divide-by-zero that a naive ``2PR/(P+R)`` hits at P=R=0.
    """
    p = entity_precision(predicted, truth)
    r = entity_recall(predicted, truth)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def fault_type_match(predicted: str | None, truth: str | None) -> bool:
    """Exact, case-insensitive match of a predicted fault_type against truth.

    ``None``/empty never matches (even if both are empty) — a missing
    prediction is not a correct one. Comparison is trimmed + lowercased so
    ``"K8s.PodCrashLoop"`` and ``"k8s.podcrashloop"`` agree.
    """
    pp = (predicted or "").strip().lower()
    tt = (truth or "").strip().lower()
    return bool(pp) and bool(tt) and pp == tt


# --------------------------------------------------------------------------- #
# Structural helpers over a RootCause (no ground truth needed)
#
# These preserve the raw truthiness / len() semantics the runner has always
# recorded, so the per-case metrics dict and the historical eval_summary.{json,
# csv} shape do not shift when scoring is introduced. The richer normalization
# (dedup, whitespace-strip, identifier collapse) lives in the entity-set
# P/R/F1 helpers above, which are the plug-in point for *ground-truth*
# comparison once the benchmark publishes a truth set — not for the richness
# counts the runner reports today.
# --------------------------------------------------------------------------- #
def has_fault_type(rc: RootCause) -> bool:
    """True if the root cause names any fault_type (raw truthiness).

    Matches the runner's historical ``bool(rc.fault_type)`` so existing
    ``pct_has_fault_type`` aggregates are unchanged.
    """
    return bool(rc.fault_type)


def n_entities(rc: RootCause) -> int:
    """Number of entity refs the root cause points at (raw count).

    Matches the runner's historical ``len(rc.entity_refs)`` so existing
    ``avg_entities`` aggregates are unchanged. Use :func:`entity_precision` /
    :func:`entity_recall` for de-duplicated ground-truth comparison.
    """
    return len(rc.entity_refs)


def n_evidence(rc: RootCause) -> int:
    """Number of evidence pointers cited by the root cause."""
    return len(rc.evidence or [])


def is_tool_call_step(s: Any) -> bool:
    """True if an RcaStep-like object is a TOOL_CALL step.

    Tolerates both a :class:`StepKind` enum and its raw ``"tool_call"`` string
    value, and plain dicts, so it works over real steps and over dicts in tests.
    Single source of truth for the tool-call predicate used by both
    :func:`n_tool_calls` and the runner's per-name Counter.
    """
    kind = getattr(s, "step_kind", None)
    if kind is None and isinstance(s, dict):
        kind = s.get("step_kind")
    return kind == StepKind.TOOL_CALL or getattr(kind, "value", None) == "tool_call"


def n_tool_calls(steps: Iterable[Any]) -> int:
    """Count TOOL_CALL steps in an iterable of RcaStep-like objects.

    Derived from :func:`is_tool_call_step` so the predicate lives in one place.
    """
    return sum(1 for s in steps or () if is_tool_call_step(s))


__all__ = [
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "fault_type_match",
    "has_fault_type",
    "n_entities",
    "n_evidence",
    "is_tool_call_step",
    "n_tool_calls",
]
