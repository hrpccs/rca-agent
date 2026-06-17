"""Pure-function tests for the scoring helpers (no LLM / no I/O).

Covers the entity-set precision/recall/F1 edge cases the rca100 benchmark
exposes: empty predicted or truth, perfect overlap, partial overlap, and a
predicted superset; plus exact fault_type matching and the structural richness
helpers used by the runner's metrics dict.
"""
from __future__ import annotations

import pytest

from rca_agent.contracts import RcaStep, RootCause, StepKind
from rca_agent.eval import scoring


# --------------------------------------------------------------------------- #
# Entity-set precision / recall / F1
# --------------------------------------------------------------------------- #
class TestEntityPrecisionRecallF1:
    def test_perfect_overlap_is_one(self):
        ents = ["svc-a", "pod-1", "db-2"]
        assert scoring.entity_precision(ents, ents) == 1.0
        assert scoring.entity_recall(ents, ents) == 1.0
        assert scoring.entity_f1(ents, ents) == 1.0

    def test_empty_predicted_returns_zero_not_raises(self):
        # No predictions against a non-empty truth -> recall 0, no ZeroDivision
        truth = ["svc-a"]
        assert scoring.entity_precision([], truth) == 0.0
        assert scoring.entity_recall([], truth) == 0.0
        assert scoring.entity_f1([], truth) == 0.0

    def test_empty_truth_returns_zero_not_raises(self):
        # Non-empty prediction against empty truth -> precision 0, no ZeroDivision
        pred = ["svc-a"]
        assert scoring.entity_precision(pred, []) == 0.0
        assert scoring.entity_recall(pred, []) == 0.0
        assert scoring.entity_f1(pred, []) == 0.0

    def test_both_empty_returns_zero_not_raises(self):
        assert scoring.entity_precision([], []) == 0.0
        assert scoring.entity_recall([], []) == 0.0
        assert scoring.entity_f1([], []) == 0.0

    def test_partial_overlap(self):
        # 2 of 3 predicted are correct, 2 of 4 truth covered
        pred = ["svc-a", "pod-1", "nope"]
        truth = ["svc-a", "pod-1", "db-2", "queue-x"]
        assert scoring.entity_precision(pred, truth) == pytest.approx(2 / 3)
        assert scoring.entity_recall(pred, truth) == pytest.approx(2 / 4)
        p, r = 2 / 3, 2 / 4
        assert scoring.entity_f1(pred, truth) == pytest.approx(2 * p * r / (p + r))

    def test_predicted_superset(self):
        # Every truth entity is predicted, plus extras -> recall 1, precision < 1
        truth = ["a", "b"]
        pred = ["a", "b", "c", "d"]
        assert scoring.entity_recall(pred, truth) == 1.0
        assert scoring.entity_precision(pred, truth) == pytest.approx(2 / 4)
        assert 0.0 < scoring.entity_f1(pred, truth) < 1.0

    def test_predicted_subset(self):
        # Fewer predictions than truth, all correct -> precision 1, recall < 1
        truth = ["a", "b", "c"]
        pred = ["a", "b"]
        assert scoring.entity_precision(pred, truth) == 1.0
        assert scoring.entity_recall(pred, truth) == pytest.approx(2 / 3)

    def test_no_overlap_is_zero(self):
        assert scoring.entity_precision(["x"], ["y"]) == 0.0
        assert scoring.entity_recall(["x"], ["y"]) == 0.0
        assert scoring.entity_f1(["x"], ["y"]) == 0.0

    def test_dicts_normalized_by_best_identifier(self):
        # entity_refs are list[dict]; name wins over id when collapsing
        pred = [{"entity_id": "i1", "entity_name": "svc-a"}, {"entity_id": "i2"}]
        truth = ["svc-a", "i2"]
        assert scoring.entity_precision(pred, truth) == 1.0
        assert scoring.entity_recall(pred, truth) == 1.0

    def test_none_and_empty_skipped(self):
        # None entries and empty dicts must not be counted as entities
        pred = [None, "", {}, "svc-a"]
        truth = ["svc-a"]
        assert scoring.entity_precision(pred, truth) == 1.0
        assert scoring.entity_recall(pred, truth) == 1.0


# --------------------------------------------------------------------------- #
# fault_type exact match
# --------------------------------------------------------------------------- #
class TestFaultTypeMatch:
    def test_exact_match(self):
        assert scoring.fault_type_match("k8s.pod_crashloop", "k8s.pod_crashloop") is True

    def test_case_insensitive(self):
        assert scoring.fault_type_match("K8s.PodCrashLoop", "k8s.podcrashloop") is True

    def test_whitespace_trimmed(self):
        assert scoring.fault_type_match("  k8s.pod_crashloop ", "k8s.pod_crashloop") is True

    def test_mismatch(self):
        assert scoring.fault_type_match("k8s.pod_crashloop", "net.dns_error") is False

    def test_predicted_none_is_false(self):
        assert scoring.fault_type_match(None, "k8s.pod_crashloop") is False

    def test_truth_none_is_false(self):
        assert scoring.fault_type_match("k8s.pod_crashloop", None) is False

    def test_both_none_is_false(self):
        # A missing prediction is not a correct one, even if truth is also missing
        assert scoring.fault_type_match(None, None) is False

    def test_empty_strings_are_false(self):
        assert scoring.fault_type_match("", "") is False
        assert scoring.fault_type_match("   ", "") is False


# --------------------------------------------------------------------------- #
# Structural helpers over RootCause
# --------------------------------------------------------------------------- #
class TestStructuralHelpers:
    def test_has_fault_type_true_when_set(self):
        rc = RootCause(summary="x", fault_type="k8s.pod_crashloop")
        assert scoring.has_fault_type(rc) is True

    def test_has_fault_type_false_when_none(self):
        rc = RootCause(summary="x")
        assert scoring.has_fault_type(rc) is False

    def test_has_fault_type_preserves_raw_truthiness(self):
        # Structural helper must match the runner's historical bool(rc.fault_type)
        # so existing pct_has_fault_type aggregates don't shift. A whitespace-only
        # fault_type is truthy under bool(); the richer strip-aware check lives
        # in fault_type_match (the ground-truth comparison helper).
        rc = RootCause(summary="x", fault_type="   ")
        assert scoring.has_fault_type(rc) is True
        assert scoring.has_fault_type(rc) == bool(rc.fault_type)

    def test_n_entities_is_raw_len_preserving_runner_semantics(self):
        # Structural helper must match the runner's historical len(rc.entity_refs)
        # so avg_entities aggregates don't shift. Dedup/collapse lives in the
        # entity-set P/R/F1 helpers (the ground-truth comparison plug-in point),
        # not in this richness count.
        rc = RootCause(
            summary="x",
            entity_refs=[
                {"entity_id": "i1", "entity_name": "svc-a"},
                {"entity_id": "i1", "entity_name": "svc-a"},  # dup -> NOT collapsed here
                {"entity_id": "i2"},
                {},
            ],
        )
        assert scoring.n_entities(rc) == 4
        assert scoring.n_entities(rc) == len(rc.entity_refs)

    def test_n_entities_dedup_lives_in_entity_set_helpers(self):
        # The dedup-by-identifier behavior is available via the P/R/F1 helpers
        # for ground-truth comparison, where collapsing duplicates is correct.
        refs = [
            {"entity_id": "i1", "entity_name": "svc-a"},
            {"entity_id": "i1", "entity_name": "svc-a"},
        ]
        truth = ["svc-a"]
        assert scoring.entity_recall(refs, truth) == 1.0  # two refs collapse to one

    def test_n_entities_zero_when_empty(self):
        rc = RootCause(summary="x")
        assert scoring.n_entities(rc) == 0

    def test_n_evidence_counts_list(self):
        rc = RootCause(summary="x", evidence=["step-1", "step-2", "step-3"])
        assert scoring.n_evidence(rc) == 3

    def test_n_evidence_zero_when_empty(self):
        rc = RootCause(summary="x")
        assert scoring.n_evidence(rc) == 0


# --------------------------------------------------------------------------- #
# n_tool_calls over a step iterable
# --------------------------------------------------------------------------- #
class TestNToolCalls:
    def _step(self, kind: StepKind | str, name: str | None = "t") -> RcaStep:
        return RcaStep(step_id="s", case_id="c", step_kind=kind, tool_name=name)

    def test_counts_only_tool_call_steps(self):
        steps = [
            self._step(StepKind.REASONING),
            self._step(StepKind.TOOL_CALL, "query_logs"),
            self._step(StepKind.TOOL_RESULT),
            self._step(StepKind.TOOL_CALL, "query_metrics"),
        ]
        assert scoring.n_tool_calls(steps) == 2

    def test_accepts_string_step_kind(self):
        steps = [self._step("tool_call"), self._step("reasoning")]
        assert scoring.n_tool_calls(steps) == 1

    def test_empty_iterable_is_zero(self):
        assert scoring.n_tool_calls([]) == 0

    def test_none_iterable_is_zero(self):
        assert scoring.n_tool_calls(None) == 0

    def test_accepts_dict_steps(self):
        steps = [
            {"step_kind": "tool_call", "tool_name": "x"},
            {"step_kind": "reasoning"},
        ]
        assert scoring.n_tool_calls(steps) == 1


# --------------------------------------------------------------------------- #
# is_tool_call_step — the single predicate shared by n_tool_calls + the runner
# --------------------------------------------------------------------------- #
class TestIsToolCallStep:
    def _step(self, kind: StepKind | str) -> RcaStep:
        return RcaStep(step_id="s", case_id="c", step_kind=kind)

    def test_true_for_tool_call_enum(self):
        assert scoring.is_tool_call_step(self._step(StepKind.TOOL_CALL)) is True

    def test_true_for_tool_call_string(self):
        assert scoring.is_tool_call_step(self._step("tool_call")) is True

    def test_false_for_other_kinds(self):
        for kind in [StepKind.REASONING, StepKind.TOOL_RESULT, StepKind.CONCLUDE]:
            assert scoring.is_tool_call_step(self._step(kind)) is False

    def test_accepts_dict(self):
        assert scoring.is_tool_call_step({"step_kind": "tool_call"}) is True
        assert scoring.is_tool_call_step({"step_kind": "reasoning"}) is False

    def test_predicate_matches_n_tool_calls_total(self):
        # The runner's per-name Counter + total and scoring.n_tool_calls must
        # agree because they share this one predicate.
        from collections import Counter

        steps = [
            self._step(StepKind.TOOL_CALL),
            self._step(StepKind.TOOL_CALL),
            self._step(StepKind.REASONING),
        ]
        breakdown = Counter(s.tool_name for s in steps if scoring.is_tool_call_step(s))
        assert sum(breakdown.values()) == scoring.n_tool_calls(steps) == 2
