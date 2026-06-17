"""Offline tests for the RCA agent CLI (``rca_agent.cli``).

No real agent, LLM, DB, or server is constructed. The heavy async entrypoints
(``build_agent_for_case``, ``run_eval``, ``import_case``, ``default_client``,
``uvicorn.run``) are monkeypatched to stubs/no-ops, and case discovery is
redirected at a temp ``RCA_CASES_DIR`` via the settings cache.

Every assertion is on the CLI dispatch contract — argument routing, proxy
cleanup, and that the right downstream entrypoint is invoked with the expected
arguments — never on real RCA behavior.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from rca_agent import cli
from rca_agent.config import get_settings

# Single source of truth: the canonical proxy-var list lives in rca_agent.cli.
_PROXY_VARS = cli._PROXY_VARS


def _set_all_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate every proxy env var with a sentinel value."""
    for v in _PROXY_VARS:
        monkeypatch.setenv(v, "http://sentinel-proxy.example:3128")


def _make_case_dir(
    cases_root: Path,
    case_id: str = "t_offline",
) -> Path:
    """Build a minimal on-disk benchmark case (task.json + topology.json).

    Only metadata needed by the CLI's ``cases`` and ``data <case> topology``
    paths is written; no parquet files are required for those subcommands.
    """
    case_dir = cases_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": case_id,
                "alert_title": "offline test alert",
                "alert_window": {
                    "start": "2026-04-25T05:18:12Z",
                    "end": "2026-04-25T05:28:12Z",
                },
                "prompt_text": "rca please",
                "available_modalities": ["metrics", "logs"],
            }
        )
    )
    (case_dir / "topology.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "window": {
                    "start_iso": "2026-04-25T05:18:12Z",
                    "end_iso": "2026-04-25T05:28:12Z",
                },
                "entities": [
                    {
                        "id": "pod-1",
                        "type": "pod",
                        "name": "checkout-pod",
                        "first_observed": 1777094292,
                        "last_observed": 1777094892,
                        "props": {"image": "checkout:v1"},
                    }
                ],
                "edges": [],
            }
        )
    )
    return case_dir


@pytest.fixture
def cases_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point ``Settings.cases_dir`` at a temp dir and reset the settings cache.

    ``get_settings`` is ``@lru_cache``d and a module-level ``settings`` is also
    exported, so both must be reset after the env var is set for the new value
    to take effect.
    """
    root = tmp_path / "cases"
    root.mkdir()
    monkeypatch.setenv("RCA_CASES_DIR", str(root))
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# _clear_proxy_env
# --------------------------------------------------------------------------- #
class TestClearProxyEnv:
    def test_removes_all_proxy_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_all_proxy_env(monkeypatch)
        # Sanity: they are really set.
        for v in _PROXY_VARS:
            assert os.environ.get(v) is not None

        cli._clear_proxy_env()

        for v in _PROXY_VARS:
            assert os.environ.get(v) is None, f"{v} should have been removed"

    def test_idempotent_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for v in _PROXY_VARS:
            monkeypatch.delenv(v, raising=False)
        # Must not raise even when nothing is set.
        cli._clear_proxy_env()
        for v in _PROXY_VARS:
            assert os.environ.get(v) is None

    def test_preserves_unrelated_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_all_proxy_env(monkeypatch)
        monkeypatch.setenv("RCA_DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setenv("NO_PROXY", "*")
        cli._clear_proxy_env()
        assert os.environ["RCA_DEEPSEEK_API_KEY"] == "sk-test"
        assert os.environ["NO_PROXY"] == "*"

    def test_logs_removed_vars_at_debug(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _set_all_proxy_env(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger="rca_agent.cli"):
            cli._clear_proxy_env()
        msgs = [r.getMessage() for r in caplog.records if r.name == "rca_agent.cli"]
        assert any("cleared proxy env vars" in m for m in msgs), msgs


# --------------------------------------------------------------------------- #
# parser / dispatch
# --------------------------------------------------------------------------- #
class TestParser:
    def test_help_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "rca-agent" in out
        assert "cases" in out and "run" in out and "serve" in out

    def test_no_subcommand_exits_nonzero(self, capsys: pytest.CaptureFixture[str]) -> None:
        # required=True on the subparsers -> argparse exits 2.
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args([])
        assert exc.value.code == 2

    @pytest.mark.parametrize(
        ("argv", "func_name"),
        [
            (["cases"], "_cmd_cases"),
            (["run", "t001"], "_cmd_run"),
            (["run", "t001", "--backend", "clickhouse", "-o", "/tmp/x.json"], "_cmd_run"),
            (["llm", "ping"], "_cmd_llm_ping"),
            (["data", "t001", "alerts"], "_cmd_data"),
            (["data", "t001", "metrics", "--filter", "svc"], "_cmd_data"),
            (["import-case", "t001"], "_cmd_import"),
            (["serve"], "_cmd_serve"),
            (["serve", "--host", "127.0.0.1", "--port", "9000", "--no-reload"], "_cmd_serve"),
            (["eval"], "_cmd_eval"),
            (["eval", "--cases", "a,b", "--backend", "parquet", "--limit", "3"], "_cmd_eval"),
            (
                ["eval", "--out-dir", "out", "--sample", "5", "--concurrency", "2"],
                "_cmd_eval",
            ),
        ],
    )
    def test_routing_selects_right_func(self, argv: list[str], func_name: str) -> None:
        args = cli.build_parser().parse_args(argv)
        assert args.func.__name__ == func_name

    def test_run_backend_defaults_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RCA_DATA_BACKEND", "clickhouse")
        args = cli.build_parser().parse_args(["run", "t001"])
        assert args.backend == "clickhouse"

    def test_data_modality_choices_enforced(self) -> None:
        # An invalid modality is rejected by argparse -> SystemExit code 2.
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args(["data", "t001", "not-a-modality"])
        assert exc.value.code == 2

    def test_eval_flags_default_and_explicit(self) -> None:
        # Defaults: out_dir=runs, concurrency=1, sample=None.
        a = cli.build_parser().parse_args(["eval"])
        assert a.out_dir == "runs"
        assert a.concurrency == 1
        assert a.sample is None
        # Explicit values flow through.
        b = cli.build_parser().parse_args(
            ["eval", "--out-dir", "eval_out", "--concurrency", "4", "--sample", "7"]
        )
        assert b.out_dir == "eval_out"
        assert b.concurrency == 4
        assert b.sample == 7


# --------------------------------------------------------------------------- #
# cases / data (real on-disk case dir, no network)
# --------------------------------------------------------------------------- #
class TestCasesAndData:
    def test_cases_lists_on_disk_cases(
        self,
        cases_dir: Path,
        capsys: pytest.LogCaptureFixture,
    ) -> None:
        _make_case_dir(cases_dir, "t_alpha")
        _make_case_dir(cases_dir, "t_beta")
        # An empty dir without task.json must be ignored.
        (cases_dir / "not-a-case").mkdir()

        rc = cli._cmd_cases(argparse.Namespace())

        assert rc == 0
        out = capsys.readouterr().out
        assert "2 cases:" in out
        assert "t_alpha" in out and "t_beta" in out
        assert "not-a-case" not in out

    def test_cases_empty_dir(self, cases_dir: Path, capsys: pytest.LogCaptureFixture) -> None:
        rc = cli._cmd_cases(argparse.Namespace())
        assert rc == 0
        out = capsys.readouterr().out
        assert "0 cases:" in out

    def test_data_topology_dump(
        self,
        cases_dir: Path,
        capsys: pytest.LogCaptureFixture,
    ) -> None:
        _make_case_dir(cases_dir, "t_offline")
        args = argparse.Namespace(case="t_offline", modality="topology", filter=None)

        rc = cli._cmd_data(args)

        assert rc == 0
        out = capsys.readouterr().out
        # Topology has exactly the one entity written by _make_case_dir.
        assert "1 entities, 0 edges" in out
        assert "checkout-pod" in out

    def test_data_unknown_modality_returns_2(
        self,
        cases_dir: Path,
        capsys: pytest.LogCaptureFixture,
    ) -> None:
        # Bypass argparse choices to hit the in-function else-branch directly.
        _make_case_dir(cases_dir, "t_offline")
        args = argparse.Namespace(
            case="t_offline", modality="definitely-not-a-modality", filter=None
        )
        rc = cli._cmd_data(args)
        assert rc == 2
        out = capsys.readouterr().out
        assert "unknown modality" in out


# --------------------------------------------------------------------------- #
# run / serve / eval / llm ping — monkeypatch the heavy entrypoints
# --------------------------------------------------------------------------- #
class TestHeavySubcommands:
    """Assert each subcommand invokes its downstream async entrypoint with the
    expected arguments, without actually running the agent/server/DeepSeek."""

    def test_run_invokes_build_agent_and_drives_agent(
        self,
        cases_dir: Path,
        capsys: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _make_case_dir(cases_dir, "t_run")

        # A fake agent whose async run() yields a minimal report.
        from rca_agent.contracts import (
            RcaReport,
            RootCause,
        )

        report = RcaReport(
            case_id="t_run",
            task_id="t_run",
            alert_title="offline test alert",
            status="concluded",
            root_cause=RootCause(
                confidence=0.9,
                fault_type="latency",
                summary="checkout pod CPU saturation",
            ),
            steps=[],
            token_usage={"total": 42},
        )

        class FakeAgent:
            model = "fake-model"
            max_steps = 7

            async def run(self, case):  # noqa: ANN001 — matches agent protocol
                yield report

        seen: dict[str, Any] = {}

        def fake_build(case_id, backend=None, **kw):  # noqa: ANN001,ANN002
            from rca_agent.cases import load_case

            seen["case_id"] = case_id
            seen["backend"] = backend
            case = load_case(case_id)
            return case, FakeAgent()

        monkeypatch.setattr("rca_agent.agent.build_agent_for_case", fake_build)

        out_path = cases_dir.parent / "report.json"
        args = argparse.Namespace(
            case="t_run", backend="parquet", output=str(out_path)
        )
        rc = cli._cmd_run(args)

        assert rc == 0
        assert seen == {"case_id": "t_run", "backend": "parquet"}
        # Report file written.
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["case_id"] == "t_run"
        assert data["root_cause"]["summary"] == "checkout pod CPU saturation"
        # Human-readable summary surfaced.
        out = capsys.readouterr().out
        assert "ROOT CAUSE" in out
        assert "TOKENS" in out

    def test_run_no_report_returns_1(
        self,
        cases_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _make_case_dir(cases_dir, "t_empty")

        class EmptyAgent:
            model = "fake"
            max_steps = 1

            async def run(self, case):  # noqa: ANN001
                if False:  # pragma: no cover — never yields
                    yield None

        def fake_build(case_id, backend=None, **kw):  # noqa: ANN001,ANN002
            from rca_agent.cases import load_case

            return load_case(case_id), EmptyAgent()

        monkeypatch.setattr("rca_agent.agent.build_agent_for_case", fake_build)

        args = argparse.Namespace(
            case="t_empty", backend="parquet", output=None
        )
        rc = cli._cmd_run(args)
        assert rc == 1

    def test_llm_ping_invokes_default_client_complete(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.LogCaptureFixture,
    ) -> None:
        seen: dict[str, Any] = {}

        class FakeClient:
            async def complete(self, req):  # noqa: ANN001
                seen["max_tokens"] = req.max_tokens
                seen["messages"] = req.messages
                return ("42", "because 7*6=42", None, {"total": 3})

        # default_client is imported lazily inside _cmd_llm_ping, so patch on
        # the deepseek_client module.
        monkeypatch.setattr(
            "rca_agent.llm.deepseek_client.default_client",
            lambda: FakeClient(),
        )

        rc = cli._cmd_llm_ping(argparse.Namespace())
        assert rc == 0
        assert seen["max_tokens"] == 2048
        assert seen["messages"][0]["content"].startswith("What is 7 * 6?")
        out = capsys.readouterr().out
        assert "42" in out
        assert "reasoning present: True" in out

    def test_eval_invokes_run_eval_with_parsed_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, Any] = {}

        async def fake_run_eval(*, cases=None, backend="parquet", limit=None, **kw):  # noqa: ANN002
            seen["cases"] = cases
            seen["backend"] = backend
            seen["limit"] = limit
            seen.update(kw)
            return []

        # run_eval is imported lazily inside _cmd_eval from rca_agent.eval.runner
        monkeypatch.setattr("rca_agent.eval.runner.run_eval", fake_run_eval)

        args = argparse.Namespace(
            cases="a,b", backend="parquet", limit=3,
            out_dir="runs", concurrency=1, sample=None,
        )
        rc = cli._cmd_eval(args)
        assert rc == 0
        assert seen["cases"] == ["a", "b"]
        assert seen["backend"] == "parquet"
        assert seen["limit"] == 3
        assert seen["out_dir"] == "runs"
        assert seen["concurrency"] == 1
        assert seen["sample"] is None

    def test_eval_default_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        async def fake_run_eval(*, cases=None, backend="parquet", limit=None, **kw):  # noqa: ANN002
            seen["cases"] = cases
            seen["backend"] = backend
            seen["limit"] = limit
            seen.update(kw)
            return []

        monkeypatch.setattr("rca_agent.eval.runner.run_eval", fake_run_eval)

        args = argparse.Namespace(
            cases=None, backend="parquet", limit=None,
            out_dir="runs", concurrency=1, sample=None,
        )
        rc = cli._cmd_eval(args)
        assert rc == 0
        assert seen["cases"] is None
        assert seen["backend"] == "parquet"
        assert seen["limit"] is None
        assert seen["out_dir"] == "runs"
        assert seen["concurrency"] == 1
        assert seen["sample"] is None

    def test_serve_invokes_uvicorn_with_expected_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_uvicorn_run(app, **kw):  # noqa: ANN001
            seen["app"] = app
            seen.update(kw)

        import rca_agent.cli as cli_mod  # local alias for clarity

        # uvicorn is imported inside _cmd_serve; patch on the uvicorn module.
        monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
        args = argparse.Namespace(host="127.0.0.1", port=9001, no_reload=True)
        rc = cli_mod._cmd_serve(args)
        assert rc == 0
        assert seen["app"] == "rca_agent.server.app:app"
        assert seen["host"] == "127.0.0.1"
        assert seen["port"] == 9001
        assert seen["reload"] is False

    def test_serve_falls_back_to_settings_when_no_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_uvicorn_run(app, **kw):  # noqa: ANN001
            seen.update(kw)

        monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
        args = argparse.Namespace(host=None, port=None, no_reload=False)
        rc = cli._cmd_serve(args)
        assert rc == 0
        s = get_settings()
        assert seen["host"] == s.server_host
        assert seen["port"] == s.server_port
        assert seen["reload"] is True

    def test_import_case_invokes_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.LogCaptureFixture,
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_import(case_id, **kw):  # noqa: ANN001
            seen["case_id"] = case_id
            return {"logs": 5, "metrics": 3}

        # import_case is imported lazily from rca_agent.providers.loader
        monkeypatch.setattr(
            "rca_agent.providers.loader.import_case", fake_import
        )

        rc = cli._cmd_import(argparse.Namespace(case="t_imp"))
        assert rc == 0
        assert seen["case_id"] == "t_imp"
        out = capsys.readouterr().out
        assert "imported" in out and "{'logs': 5, 'metrics': 3}" in out


# --------------------------------------------------------------------------- #
# eval flags: --out-dir / --sample / --concurrency end-to-end via _cmd_eval
# --------------------------------------------------------------------------- #
class TestEvalFlags:
    """Exercise the new eval flags through ``_cmd_eval`` itself, driving the
    real :func:`run_eval` with an injected fake-agent factory (no LLM/DB)."""

    @staticmethod
    def _make_case(case_id: str):
        from datetime import UTC, datetime

        from rca_agent.contracts import (
            Case,
            Modality,
            Task,
            TimeWindow,
            Topology,
        )

        tw = TimeWindow(
            start=datetime(2026, 4, 25, 5, 18, 12, tzinfo=UTC),
            end=datetime(2026, 4, 25, 5, 28, 12, tzinfo=UTC),
        )
        task = Task(
            task_id=case_id,
            alert_title=f"alert {case_id}",
            alert_window=tw,
            prompt_text="rca",
            available_modalities=[Modality.METRICS],
        )
        return Case(task=task, topology=Topology(case_id=case_id, window=tw), case_dir="/tmp/fake")

    def _fake_factory(self):
        from rca_agent.contracts import RcaReport, RcaStep, RootCause, StepKind

        def factory(case_id: str, backend: str = "parquet"):
            case = self._make_case(case_id)

            class Agent:
                async def run(self, case):  # noqa: ANN001
                    yield RcaStep(
                        step_id=f"{case_id}-c1",
                        case_id=case_id,
                        step_kind=StepKind.TOOL_CALL,
                        tool_name="query_metrics",
                    )
                    yield RcaStep(
                        step_id=f"{case_id}-r1",
                        case_id=case_id,
                        step_kind=StepKind.TOOL_RESULT,
                        tool_name="query_metrics",
                    )
                    yield RcaReport(
                        case_id=case_id,
                        task_id=case_id,
                        alert_title=f"alert {case_id}",
                        status="completed",
                        root_cause=RootCause(
                            confidence=0.5, fault_type="latency", summary="x"
                        ),
                        steps=[],
                        token_usage={"total_tokens": 1},
                    )

            return case, Agent()

        return factory

    def test_out_dir_flag_writes_to_given_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import rca_agent.eval.runner as runner_mod

        monkeypatch.setattr(runner_mod, "_default_factory", self._fake_factory())
        out = tmp_path / "myruns"
        rc = cli._cmd_eval(
            argparse.Namespace(
                cases="t-a",
                backend="parquet",
                limit=None,
                out_dir=str(out),
                concurrency=1,
                sample=None,
            )
        )
        assert rc == 0
        assert (out / "eval_summary.json").exists()
        summary = json.loads((out / "eval_summary.json").read_text())
        assert summary["aggregate"]["n_cases"] == 1

    def test_sample_flag_picks_subset_of_list_cases(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import rca_agent.eval.runner as runner_mod

        all_ids = [f"t-{i}" for i in range(10)]
        monkeypatch.setattr(runner_mod, "list_cases", lambda: list(all_ids))
        monkeypatch.setattr(runner_mod, "_default_factory", self._fake_factory())
        out = tmp_path / "runs"

        rc = cli._cmd_eval(
            argparse.Namespace(
                cases=None,
                backend="parquet",
                limit=None,
                out_dir=str(out),
                concurrency=1,
                sample=3,
            )
        )
        assert rc == 0
        # The real run_eval ran the sampled subset; the summary covers exactly
        # 3 distinct cases drawn from the 10 available.
        summary = json.loads((out / "eval_summary.json").read_text())
        case_ids = {r["case_id"] for r in summary["results"]}
        assert len(case_ids) == 3
        assert case_ids.issubset(set(all_ids))

    def test_concurrency_flag_runs_all_cases_and_one_aggregate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import rca_agent.eval.runner as runner_mod

        monkeypatch.setattr(runner_mod, "_default_factory", self._fake_factory())
        out = tmp_path / "conc"
        rc = cli._cmd_eval(
            argparse.Namespace(
                cases="t-1,t-2,t-3,t-4",
                backend="parquet",
                limit=None,
                out_dir=str(out),
                concurrency=2,
                sample=None,
            )
        )
        assert rc == 0
        summary = json.loads((out / "eval_summary.json").read_text())
        # One aggregate covers all 4 cases (no per-concurrency clobbering).
        assert summary["aggregate"]["n_cases"] == 4
        assert {r["case_id"] for r in summary["results"]} == {"t-1", "t-2", "t-3", "t-4"}


# --------------------------------------------------------------------------- #
# main() top-level error handling
# --------------------------------------------------------------------------- #
class TestMainErrorHandling:
    def test_main_clears_proxy_then_dispatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _set_all_proxy_env(monkeypatch)
        # Point case discovery at an empty temp dir so _cmd_cases returns 0
        # without touching the real dataset path.
        monkeypatch.setenv("RCA_CASES_DIR", str(tmp_path))
        get_settings.cache_clear()
        try:
            rc = cli.main(["cases"])
        finally:
            get_settings.cache_clear()
        assert rc == 0
        # main() must have cleared proxies before dispatch.
        for v in _PROXY_VARS:
            assert os.environ.get(v) is None

    def test_main_wraps_unexpected_exception_to_exit_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.LogCaptureFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def boom(args):  # noqa: ANN001
            raise RuntimeError("kaboom")

        # Replace the func bound to the `cases` subparser by rebuilding a parser
        # whose cases subcommand points at our stub. Easiest: patch _cmd_cases
        # before build_parser runs by swapping the symbol.
        monkeypatch.setattr(cli, "_cmd_cases", boom)
        # build_parser references _cmd_cases by attribute lookup at parser-build
        # time, so swapping the module symbol then calling build_parser works.
        # But main() calls build_parser() fresh each time -> picks up the swap.
        with caplog.at_level(logging.ERROR, logger="rca_agent.cli"):
            rc = cli.main(["cases"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "kaboom" in err
        msgs = [r.getMessage() for r in caplog.records if r.name == "rca_agent.cli"]
        assert any("command 'cases' failed" in m for m in msgs), msgs

    def test_main_propagates_systemexit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def exit_3(args):  # noqa: ANN001
            raise SystemExit(3)

        monkeypatch.setattr(cli, "_cmd_cases", exit_3)
        with pytest.raises(SystemExit) as exc:
            cli.main(["cases"])
        assert exc.value.code == 3

    def test_help_through_main_exits_zero(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--help"])
        assert exc.value.code == 0


# --------------------------------------------------------------------------- #
# _print_step defensive rendering
# --------------------------------------------------------------------------- #
class TestPrintStep:
    def test_malformed_step_kind_does_not_raise(
        self,
        capsys: pytest.LogCaptureFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class Broken:
            @property
            def step_kind(self) -> Any:
                raise RuntimeError("bad step")

        with caplog.at_level(logging.WARNING, logger="rca_agent.cli"):
            cli._print_step(Broken())
        # Nothing printed, warning logged, no exception escaped.
        out = capsys.readouterr().out
        assert out == ""
        msgs = [r.getMessage() for r in caplog.records if r.name == "rca_agent.cli"]
        assert any("failed to render step" in m for m in msgs), msgs

    def test_malformed_payload_attribute_does_not_raise(
        self,
        capsys: pytest.LogCaptureFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # step_kind is valid (reasoning) but the payload attribute raises.
        # The renderer must defend the WHOLE body, not just the kind read.
        class BadThought:
            class _Kind:
                value = "reasoning"

            step_kind = _Kind()

            @property
            def thought(self) -> Any:
                raise RuntimeError("bad thought attr")

        with caplog.at_level(logging.WARNING, logger="rca_agent.cli"):
            cli._print_step(BadThought())
        out = capsys.readouterr().out
        assert out == "", f"nothing should be printed on a raising payload; got {out!r}"
        msgs = [r.getMessage() for r in caplog.records if r.name == "rca_agent.cli"]
        assert any("failed to render step" in m for m in msgs), msgs

    def test_unknown_kind_logged_at_debug(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class Step:
            step_kind = "weird-kind"

        with caplog.at_level(logging.DEBUG, logger="rca_agent.cli"):
            cli._print_step(Step())
        msgs = [r.getMessage() for r in caplog.records if r.name == "rca_agent.cli"]
        assert any("unhandled step_kind" in m for m in msgs), msgs

    def test_well_formed_reasoning_step_renders(
        self,
        capsys: pytest.LogCaptureFixture,
    ) -> None:
        class Step:
            class _Kind:
                value = "reasoning"

            step_kind = _Kind()
            thought = "analyzing cpu spike"

        cli._print_step(Step())
        out = capsys.readouterr().out
        assert "[thought] analyzing cpu spike" in out


# --------------------------------------------------------------------------- #
# _configure_logging
# --------------------------------------------------------------------------- #
class TestConfigureLogging:
    def test_installs_handler_when_none_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import logging as _logging

        root = _logging.getLogger()
        # Simulate a pristine root logger with no handlers.
        monkeypatch.setattr(root, "handlers", [])
        # basicConfig is a no-op if handlers already exist, so with an empty
        # list it must install exactly one.
        cli._configure_logging()
        assert len(root.handlers) >= 1

    def test_does_not_clobber_existing_handlers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import logging as _logging

        root = _logging.getLogger()
        sentinel = _logging.StreamHandler()
        monkeypatch.setattr(root, "handlers", [sentinel])
        cli._configure_logging()
        # Existing setup left untouched (uvicorn configures its own logging).
        assert sentinel in root.handlers
        assert root.handlers == [sentinel]

    def test_level_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import logging as _logging

        root = _logging.getLogger()
        monkeypatch.setattr(root, "handlers", [])
        monkeypatch.setenv("RCA_LOG_LEVEL", "DEBUG")
        try:
            cli._configure_logging()
            assert _logging.getLogger("rca_agent.cli").getEffectiveLevel() <= _logging.DEBUG
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)

    def test_invalid_level_falls_back_to_info(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import logging as _logging

        root = _logging.getLogger()
        monkeypatch.setattr(root, "handlers", [])
        monkeypatch.setenv("RCA_LOG_LEVEL", "not-a-level")
        try:
            cli._configure_logging()
            # An unrecognized level name falls back to INFO (getattr default).
            assert _logging.getLogger("rca_agent.cli").getEffectiveLevel() == _logging.INFO
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)


# The cli module must not perform network I/O at import time (it is imported by
# the server and tests). A clean reload must succeed and expose the entrypoints.
def test_module_import_is_side_effect_free() -> None:
    import importlib

    mod = importlib.reload(cli)
    assert callable(mod.build_parser)
    assert callable(mod.main)
    assert callable(mod._clear_proxy_env)
    assert callable(mod._configure_logging)
