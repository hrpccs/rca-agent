"""Offline tests for the ``runs`` / ``trace`` CLI subcommands (``rca_agent.cli``).

No real DB / network is constructed: the CLI's store factory
(:func:`rca_agent.cli._new_store`) is monkeypatched to a fake store so the tests
exercise only argument routing, table/trace rendering, and error handling.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import pytest

from rca_agent import cli
from rca_agent.contracts import RcaReport, RcaStep, RootCause, StepKind


def _step(
    step_kind: StepKind,
    *,
    step_id: str = "s1",
    thought: str | None = None,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    tool_result_text: str | None = None,
    hypothesis: str | None = None,
    confidence: float | None = None,
) -> RcaStep:
    return RcaStep(
        step_id=step_id,
        case_id="t001",
        step_kind=step_kind,
        thought=thought,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result_text=tool_result_text,
        hypothesis=hypothesis,
        confidence=confidence,
    )


def _report(steps: list[RcaStep] | None = None) -> RcaReport:
    return RcaReport(
        case_id="t001",
        task_id="t001",
        alert_title="checkout alert",
        status="completed",
        model="deepseek-reasoner",
        token_usage={"total": 42},
        root_cause=RootCause(
            confidence=0.82,
            fault_type="db.slow_query",
            summary="cart latency regression from slow inventory DB queries",
        ),
        steps=steps or [],
    )


class _FakeStore:
    """Minimal stand-in for MysqlStore (only the methods the CLI uses)."""

    def __init__(
        self,
        *,
        reports: list[RcaReport] | None = None,
        by_id: dict[str, RcaReport] | None = None,
        raise_on_list: bool = False,
        raise_on_get: bool = False,
    ) -> None:
        self._reports = reports if reports is not None else []
        self._by_id = by_id or {}
        self._raise_on_list = raise_on_list
        self._raise_on_get = raise_on_get

    def list_reports(self, case_id: str | None = None, limit: int = 50) -> list[RcaReport]:
        if self._raise_on_list:
            from rca_agent.store.mysql_store import StoreError

            raise StoreError("list_reports boom")
        rows = self._reports
        if case_id is not None:
            rows = [r for r in rows if r.case_id == case_id]
        return rows[:limit]

    def get_report(self, report_id: str) -> RcaReport | None:
        if self._raise_on_get:
            from rca_agent.store.mysql_store import StoreError

            raise StoreError("get_report boom")
        return self._by_id.get(report_id)


def _patch_store(monkeypatch: pytest.MonkeyPatch, store: _FakeStore) -> None:
    """Redirect ``rca_agent.cli._new_store`` to return ``store``."""
    monkeypatch.setattr(cli, "_new_store", lambda: store)


# --------------------------------------------------------------------------- #
# parser routing
# --------------------------------------------------------------------------- #
class TestRunsTraceRouting:
    @pytest.mark.parametrize(
        ("argv", "func_name"),
        [
            (["runs"], "_cmd_runs"),
            (["runs", "--case", "t001"], "_cmd_runs"),
            (["runs", "--limit", "5"], "_cmd_runs"),
            (["trace", "0192f8c1a4b748e29a8f1c2d3b4e5f60"], "_cmd_trace"),
        ],
    )
    def test_routing(self, argv: list[str], func_name: str) -> None:
        args = cli.build_parser().parse_args(argv)
        assert args.func.__name__ == func_name

    def test_runs_defaults(self) -> None:
        args = cli.build_parser().parse_args(["runs"])
        assert args.case is None
        assert args.limit == 50

    def test_runs_limit_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCA_RUNS_LIMIT", "7")
        args = cli.build_parser().parse_args(["runs"])
        assert args.limit == 7


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
class TestRunsCmd:
    def test_list_prints_table(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Use 2 steps (not 1) so the count token "steps=2" can't be a substring
        # of a future "steps=12"/"steps=20" — guards against fragile matching.
        _patch_store(
            monkeypatch,
            _FakeStore(
                reports=[
                    _report([_step(StepKind.REASONING, thought="x"), _step(StepKind.OBSERVE)]),
                    _report(),
                ]
            ),
        )
        args = argparse.Namespace(case=None, limit=50)
        rc = cli._cmd_runs(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 run(s):" in out
        assert "case=t001" in out
        assert "status=completed" in out
        # step_count is rendered (len(report.steps)); tokens are unambiguous.
        assert "steps=2" in out
        assert "steps=0" in out

    def test_case_filter_passed_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        seen: dict[str, Any] = {}
        store = _FakeStore(reports=[_report()])

        def list_reports(case_id=None, limit=50):  # noqa: ANN001
            seen["case_id"] = case_id
            seen["limit"] = limit
            return store._reports  # type: ignore[attr-defined]

        store.list_reports = list_reports  # type: ignore[method-assign]
        _patch_store(monkeypatch, store)
        args = argparse.Namespace(case="t001", limit=10)
        rc = cli._cmd_runs(args)
        assert rc == 0
        assert seen == {"case_id": "t001", "limit": 10}

    def test_empty_list_friendly_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _patch_store(monkeypatch, _FakeStore(reports=[]))
        rc = cli._cmd_runs(argparse.Namespace(case=None, limit=50))
        assert rc == 0
        out = capsys.readouterr().out
        assert "no runs found" in out

    def test_empty_list_with_case_filter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _patch_store(monkeypatch, _FakeStore(reports=[]))
        rc = cli._cmd_runs(argparse.Namespace(case="t999", limit=50))
        assert rc == 0
        out = capsys.readouterr().out
        assert "no runs found" in out and "t999" in out

    def test_store_error_stderr_and_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _patch_store(monkeypatch, _FakeStore(raise_on_list=True))
        rc = cli._cmd_runs(argparse.Namespace(case=None, limit=50))
        assert rc == 1
        err = capsys.readouterr().err
        assert "error" in err and "list_runs" not in err  # message surfaces, no traceback

    def test_unexpected_error_stderr_and_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def boom() -> None:  # noqa: ANN202
            raise RuntimeError("kaboom")

        monkeypatch.setattr(cli, "_new_store", boom)
        rc = cli._cmd_runs(argparse.Namespace(case=None, limit=50))
        assert rc == 1
        err = capsys.readouterr().err
        assert "kaboom" in err


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #
class TestTraceCmd:
    _GOOD_ID = "0192f8c1a4b748e29a8f1c2d3b4e5f60"

    def _args(self, report_id: str = _GOOD_ID) -> argparse.Namespace:
        return argparse.Namespace(report_id=report_id)

    def test_prints_ordered_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        steps = [
            _step(StepKind.REASONING, step_id="s1", thought="spike in checkout errors"),
            _step(
                StepKind.TOOL_CALL,
                step_id="s2",
                tool_name="query_metrics",
                tool_args={"entity_names": ["cart"]},
            ),
            _step(
                StepKind.TOOL_RESULT,
                step_id="s3",
                tool_name="query_metrics",
                tool_result_text="cart p99 2.4s",
            ),
            _step(StepKind.CONCLUDE, step_id="s4", hypothesis="slow inventory DB", confidence=0.82),
        ]
        store = _FakeStore(by_id={self._GOOD_ID: _report(steps)})
        _patch_store(monkeypatch, store)
        rc = cli._cmd_trace(self._args())
        assert rc == 0
        out = capsys.readouterr().out
        # Header
        assert "=== trace" in out and "case=t001" in out
        assert "steps=4" in out and "confidence=0.82" in out
        # Steps rendered with their 1-based index AND in chronological order:
        # verify the index markers appear in increasing order in the output.
        markers = [out.index(f"#{i}   ") for i in (1, 2, 3, 4)]
        assert markers == sorted(markers), "step indices must be in ascending order"
        assert "#1   reasoning" in out and "spike in checkout errors" in out
        # tool_call renders tool_name AND its args.
        assert "#2   tool_call" in out and "query_metrics" in out
        assert '"entity_names"' in out and '"cart"' in out
        assert "#3   tool_result" in out and "cart p99 2.4s" in out
        # conclude renders confidence AND hypothesis.
        assert "#4   conclude" in out and "conf=0.82" in out and "slow inventory DB" in out
        # Root cause footer
        assert "ROOT CAUSE:" in out and "FAULT TYPE: db.slow_query" in out
        assert "TOKENS:" in out

    def test_unknown_run_stderr_and_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _patch_store(monkeypatch, _FakeStore(by_id={}))  # no such id
        rc = cli._cmd_trace(self._args())
        assert rc == 1
        err = capsys.readouterr().err
        assert "no such run" in err and self._GOOD_ID in err

    def test_store_error_stderr_and_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _patch_store(monkeypatch, _FakeStore(raise_on_get=True))
        rc = cli._cmd_trace(self._args())
        assert rc == 1
        err = capsys.readouterr().err
        assert "error" in err and self._GOOD_ID in err

    def test_unexpected_error_stderr_and_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def boom() -> None:  # noqa: ANN202
            raise RuntimeError("trace boom")

        monkeypatch.setattr(cli, "_new_store", boom)
        rc = cli._cmd_trace(self._args())
        assert rc == 1
        err = capsys.readouterr().err
        assert "trace boom" in err

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            "short",
            "0192f8c1a4b748e29a8f1c2d3b4e5f6",  # 31 chars
            "0192f8c1a4b748e29a8f1c2d3b4e5f601",  # 33 chars
            "g192f8c1a4b748e29a8f1c2d3b4e5f60",  # non-hex
            "0192f8c1-a4b7-48e2-9a8f-1c2d3b4e5f60",  # dashed uuid (not accepted)
        ],
    )
    def test_bad_id_format_usage_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        bad_id: str,
    ) -> None:
        # Even with a store that would raise, validation must short-circuit first.
        _patch_store(monkeypatch, _FakeStore(raise_on_get=True))
        rc = cli._cmd_trace(self._args(bad_id))
        assert rc == 2
        err = capsys.readouterr().err
        assert "valid report id" in err


# --------------------------------------------------------------------------- #
# _print_trace_step defensive rendering
# --------------------------------------------------------------------------- #
class TestPrintTraceStep:
    def test_renders_every_kind(self, capsys: pytest.CaptureFixture[str]) -> None:
        # observe/hypothesize/investigate must NOT be silently dropped.
        for i, kind in enumerate(
            [
                StepKind.OBSERVE,
                StepKind.HYPOTHESIZE,
                StepKind.INVESTIGATE,
                StepKind.REASONING,
                StepKind.TOOL_CALL,
                StepKind.TOOL_RESULT,
                StepKind.CONCLUDE,
                StepKind.ERROR,
            ],
            start=1,
        ):
            cli._print_trace_step(i, _step(kind, thought=f"t-{kind.value}"))
        out = capsys.readouterr().out
        for kind in [
            StepKind.OBSERVE,
            StepKind.HYPOTHESIZE,
            StepKind.INVESTIGATE,
            StepKind.REASONING,
            StepKind.TOOL_CALL,
            StepKind.TOOL_RESULT,
            StepKind.CONCLUDE,
            StepKind.ERROR,
        ]:
            assert kind.value in out, f"{kind.value} should be rendered"

    def test_malformed_step_does_not_raise(
        self,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class Broken:
            @property
            def step_kind(self) -> Any:
                raise RuntimeError("bad step")

        with caplog.at_level(logging.WARNING, logger="rca_agent.cli"):
            cli._print_trace_step(1, Broken())
        out = capsys.readouterr().out
        assert out == ""
        msgs = [r.getMessage() for r in caplog.records if r.name == "rca_agent.cli"]
        assert any("failed to render trace step" in m for m in msgs), msgs

    def test_conclude_renders_conf_and_hypothesis(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli._print_trace_step(
            1, _step(StepKind.CONCLUDE, hypothesis="slow inventory DB", confidence=0.82)
        )
        out = capsys.readouterr().out
        assert "conclude" in out and "conf=0.82" in out and "slow inventory DB" in out

    def test_conclude_without_confidence_omits_conf_not_None(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # confidence is Optional; when absent it must NOT leak the literal "None".
        cli._print_trace_step(1, _step(StepKind.CONCLUDE, hypothesis="rc", confidence=None))
        out = capsys.readouterr().out
        assert "conclude" in out and "rc" in out
        assert "conf=None" not in out
        assert "conf=" not in out

    def test_conclude_confidence_zero_is_not_dropped(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # confidence=0.0 is falsy but valid; must still render (not be dropped).
        cli._print_trace_step(1, _step(StepKind.CONCLUDE, hypothesis="unsure", confidence=0.0))
        out = capsys.readouterr().out
        assert "conf=0.0" in out

    def test_conclude_empty_hypothesis_still_shows_confidence(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # hypothesis empty/None but confidence set: confidence must still render.
        cli._print_trace_step(1, _step(StepKind.CONCLUDE, hypothesis=None, confidence=0.9))
        out = capsys.readouterr().out
        assert "conf=0.9" in out

    def test_tool_call_renders_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli._print_trace_step(
            1,
            _step(
                StepKind.TOOL_CALL,
                tool_name="query_metrics",
                tool_args={"entity_names": ["cart"], "metrics": ["error_rate"]},
            ),
        )
        out = capsys.readouterr().out
        assert "tool_call" in out and "query_metrics" in out
        assert '"entity_names"' in out and '"cart"' in out


# --------------------------------------------------------------------------- #
# _env_int (RCA_RUNS_LIMIT non-numeric guard)
# --------------------------------------------------------------------------- #
class TestEnvInt:
    def test_missing_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RCA_RUNS_LIMIT", raising=False)
        assert cli._env_int("RCA_RUNS_LIMIT", 50) == 50

    def test_numeric_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCA_RUNS_LIMIT", "7")
        assert cli._env_int("RCA_RUNS_LIMIT", 50) == 7

    def test_non_numeric_falls_back_and_warns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("RCA_RUNS_LIMIT", "unlimited")
        with caplog.at_level(logging.WARNING, logger="rca_agent.cli"):
            assert cli._env_int("RCA_RUNS_LIMIT", 50) == 50
        msgs = [r.getMessage() for r in caplog.records if r.name == "rca_agent.cli"]
        assert any("RCA_RUNS_LIMIT" in m and "not an int" in m for m in msgs), msgs

    def test_runs_help_builds_even_with_bad_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # build_parser() must NOT raise on a non-numeric RCA_RUNS_LIMIT.
        monkeypatch.setenv("RCA_RUNS_LIMIT", "oops")
        args = cli.build_parser().parse_args(["runs"])
        assert args.limit == 50
