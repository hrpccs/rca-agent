"""Unit tests for the skill recall engine (alert → best troubleshooting SOP)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rca_agent.skills.recall import (
    DEFAULT_MAX_CHARS,
    MIN_SCORE_THRESHOLD,
    SkillRecaller,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
@dataclass
class _FakeSkill:
    """Minimal skill-like object (matches the duck-typed store contract)."""

    name: str
    description: str
    body: str
    base_dir: str = "/tmp/skills"
    references: list[str] = field(default_factory=list)


class _FakeStore:
    """Duck-typed store: catalog() + get() + reference_text()."""

    def __init__(self, skills: list[_FakeSkill], refs: dict[str, dict[str, str]] | None = None):
        # refs: {skill_name: {rel_path: text}}
        self._skills = {s.name: s for s in skills}
        self._refs = refs or {}
        # Track reference_text calls so tests can assert laziness.
        self.reference_text_calls: list[tuple[str, str]] = []

    def catalog(self) -> list[tuple[str, str]]:
        return [(s.name, s.description) for s in self._skills.values()]

    def get(self, name: str) -> _FakeSkill | None:
        return self._skills.get(name)

    def reference_text(self, skill: _FakeSkill, rel_path: str) -> str | None:
        self.reference_text_calls.append((skill.name, rel_path))
        return self._refs.get(skill.name, {}).get(rel_path)


def _build_rca_diagnose_store() -> _FakeStore:
    """A store mirroring the bundled rca-diagnose SOP keyword router."""
    return _FakeStore(
        skills=[
            _FakeSkill(
                name="error-rate-spike",
                description="Troubleshoot error-rate / 5xx spikes and 错误次数告警.",
                body="# error-rate-spike SOP\nGeneric error-rate body.",
                references=["error-rate-spike.md"],
            ),
            _FakeSkill(
                name="latency-spike",
                description="Diagnose latency / RT / 响应时间 / 超时 spikes and P99 regression.",
                body="# latency-spike SOP\nGeneric latency body.",
                references=["latency-spike.md"],
            ),
            _FakeSkill(
                name="traffic-drop",
                description="Investigate traffic drop / 流量下跌 / QPS decline.",
                body="# traffic-drop SOP\nGeneric traffic body.",
                references=["traffic-drop.md"],
            ),
            _FakeSkill(
                name="pod-crash",
                description="Debug pod crash / OOM / OOMKilled / CrashLoop / 重启.",
                body="# pod-crash SOP\nGeneric pod-crash body.",
                references=["pod-crash.md"],
            ),
            _FakeSkill(
                name="log-analysis",
                description="Log analysis / 日志排查 playbook.",
                body="# log-analysis SOP\nGeneric log body.",
                references=["log-analysis.md"],
            ),
            _FakeSkill(
                name="trace-analysis",
                description="Trace / 链路 analysis playbook.",
                body="# trace-analysis SOP\nGeneric trace body.",
                references=["trace-analysis.md"],
            ),
            _FakeSkill(
                name="rca-framework",
                description="Generic root cause analysis framework — fallback when no specific SOP fits.",
                body="# RCA Framework\nGeneric RCA methodology.",
                references=[],
            ),
        ],
        refs={
            "error-rate-spike": {
                "error-rate-spike.md": (
                    "# Error-Rate Spike\n"
                    "Triage 5xx / 错误次数 spikes: check upstream error logs, "
                    "recent deployments, dependency health."
                ),
            },
            "latency-spike": {
                "latency-spike.md": (
                    "# Latency Spike\n"
                    "Triage 响应时间 / latency / P99 regression and 超时: "
                    "check slow queries, saturation, GC pauses."
                ),
            },
            "traffic-drop": {
                "traffic-drop.md": (
                    "# Traffic Drop\n"
                    "Investigate 流量下跌 / QPS decline / traffic drop: "
                    "check upstream ingress, dependencies, config."
                ),
            },
            "pod-crash": {
                "pod-crash.md": (
                    "# Pod Crash\n"
                    "Debug OOMKilled / CrashLoop / 重启: inspect events, "
                    "memory limits, recent image rollout."
                ),
            },
            "log-analysis": {
                "log-analysis.md": "# Log Analysis\nStructured 日志 / log triage steps.",
            },
            "trace-analysis": {
                "trace-analysis.md": "# Trace Analysis\n链路 / trace inspection steps.",
            },
        },
    )


@pytest.fixture
def recaller() -> SkillRecaller:
    return SkillRecaller(_build_rca_diagnose_store())


# --------------------------------------------------------------------------- #
# catalog() delegation
# --------------------------------------------------------------------------- #
def test_catalog_delegates_to_store(recaller: SkillRecaller):
    names = [n for n, _ in recaller.catalog()]
    assert "error-rate-spike" in names
    assert "latency-spike" in names
    assert "rca-framework" in names


def test_catalog_returns_empty_when_store_catalog_raises():
    class _Boom:
        def catalog(self):
            raise RuntimeError("boom")

    assert SkillRecaller(_Boom()).catalog() == []


# --------------------------------------------------------------------------- #
# Routing: alert title → expected SOP
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("title", "signals", "expected"),
    [
        # bilingual: English + Chinese
        ("checkout 错误次数告警", None, "error-rate-spike"),
        ("error_rate 5xx spike on checkout", None, "error-rate-spike"),
        ("cart 响应时间抖动告警", ["latency P99 high"], "latency-spike"),
        ("latency P99 regression / RT timeout", None, "latency-spike"),
        ("checkout service 超时", None, "latency-spike"),
        ("流量下跌 / traffic drop", None, "traffic-drop"),
        ("Pod OOMKilled CrashLoop", None, "pod-crash"),
        ("pod 重启 / restart loop", None, "pod-crash"),
        ("trace 链路异常", None, "trace-analysis"),
        ("日志 / log error flood", None, "log-analysis"),
    ],
)
def test_best_for_routes_to_expected_sop(recaller: SkillRecaller, title, signals, expected):
    result = recaller.best_for(title, signals)
    assert result is not None, f"expected a match for {title!r}, got None"
    name, body = result
    assert name == expected, f"{title!r} → {name!r}, expected {expected!r}"
    assert isinstance(body, str)
    assert body.strip()  # non-empty


def test_best_for_returns_body_text(recaller: SkillRecaller):
    result = recaller.best_for("checkout 错误次数告警")
    assert result is not None
    _, body = result
    # Should return the matched reference text (not the generic skill body).
    assert "Error-Rate Spike" in body or "error-rate" in body.lower()


def test_best_for_signals_augment_query(recaller: SkillRecaller):
    # Title alone is generic; signals carry the routing signal.
    result = recaller.best_for("alert triggered", ["5xx error spike on checkout"])
    assert result is not None
    name, _ = result
    assert name == "error-rate-spike"


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
def test_best_for_truncates_long_body_with_note(monkeypatch):
    # The reference body carries the routing keywords in its heading so it is
    # the matched doc; the description is generic so the reference wins; the
    # padding makes the body exceed the default cap.
    long_body = "# error-rate 5xx spike triage\n" + "X" * (DEFAULT_MAX_CHARS + 5000)
    store = _FakeStore(
        skills=[
            _FakeSkill(
                name="error-rate-spike",
                description="Generic error troubleshooting SOP.",
                body="short fallback body",
                references=["sop.md"],
            ),
        ],
        refs={"error-rate-spike": {"sop.md": long_body}},
    )
    monkeypatch.delenv("RCA_SKILL_MAX_CHARS", raising=False)
    recaller = SkillRecaller(store)
    result = recaller.best_for("5xx error spike")
    assert result is not None
    _, body = result
    assert body.endswith("…[truncated]")
    # body = cap chars + note; cap is the configured default.
    assert len(body) <= DEFAULT_MAX_CHARS + len("\n\n…[truncated]") + 1


def test_best_for_respects_env_cap(monkeypatch):
    long_body = "# error-rate 5xx spike triage\n" + "Y" * 4000
    store = _FakeStore(
        skills=[
            _FakeSkill(
                name="error-rate-spike",
                description="Generic error troubleshooting SOP.",
                body="short fallback body",
                references=["sop.md"],
            ),
        ],
        refs={"error-rate-spike": {"sop.md": long_body}},
    )
    monkeypatch.setenv("RCA_SKILL_MAX_CHARS", "200")
    recaller = SkillRecaller(store)
    result = recaller.best_for("5xx error spike")
    assert result is not None
    _, body = result
    assert body.endswith("…[truncated]")
    assert len(body) <= 200 + len("\n\n…[truncated]") + 1


def test_best_for_no_truncation_when_under_cap(recaller: SkillRecaller):
    result = recaller.best_for("5xx error spike")
    assert result is not None
    _, body = result
    assert "…[truncated]" not in body


# --------------------------------------------------------------------------- #
# Threshold gating — irrelevant alerts return None
# --------------------------------------------------------------------------- #
def test_best_for_returns_none_for_zero_overlap(recaller: SkillRecaller):
    # Pure stopword / punctuation query → no tokens → None.
    assert recaller.best_for("!!! the a an !!!") is None


def test_best_for_returns_none_for_irrelevant_query(recaller: SkillRecaller):
    # No SOP mentions cooking / 烹饪.
    result = recaller.best_for("烹饪 recipe cooking pasta")
    # Either no router keyword hits (score 0 → None) or below threshold.
    if result is not None:
        name, _ = result
        # If something scored, it must be the generic framework fallback, and
        # even then it should be below the threshold (so this branch is a guard).
        assert name == "rca-framework"
        pytest.fail(f"irrelevant query unexpectedly scored above threshold → {name}")


def test_threshold_is_positive():
    assert MIN_SCORE_THRESHOLD > 0.0


# --------------------------------------------------------------------------- #
# Defensive: store that throws → never raise, return None
# --------------------------------------------------------------------------- #
def test_best_for_never_raises_when_store_catalog_throws():
    class _BoomStore:
        def catalog(self):
            raise RuntimeError("catalog boom")

        def get(self, name):
            raise RuntimeError("get boom")

        def reference_text(self, skill, rel_path):
            raise RuntimeError("ref boom")

    recaller = SkillRecaller(_BoomStore())
    assert recaller.best_for("5xx error spike") is None


def test_best_for_never_raises_when_get_throws():
    class _PartialStore:
        def catalog(self):
            return [("error-rate-spike", "desc"), ("latency-spike", "desc")]

        def get(self, name):
            raise RuntimeError("get boom")

        def reference_text(self, skill, rel_path):
            return None

    recaller = SkillRecaller(_PartialStore())
    # Every get() raises; recall should swallow and return None.
    assert recaller.best_for("5xx error spike") is None


def test_best_for_falls_back_when_reference_text_throws():
    """A skill whose reference_text raises should still be scorable via body."""

    class _RefBoomStore:
        def __init__(self):
            self.skill = _FakeSkill(
                name="error-rate-spike",
                description="error-rate 5xx spike triage.",
                body="# Error-Rate Spike\nTriage 5xx error spikes step by step.",
                references=["sop.md"],
            )

        def catalog(self):
            return [(self.skill.name, self.skill.description)]

        def get(self, name):
            return self.skill if name == self.skill.name else None

        def reference_text(self, skill, rel_path):
            raise RuntimeError("ref read failed")

    recaller = SkillRecaller(_RefBoomStore())
    result = recaller.best_for("5xx error spike")
    # Description/body still carries the routing signal → match, body = skill.body.
    assert result is not None
    name, body = result
    assert name == "error-rate-spike"
    assert "Error-Rate Spike" in body


def test_best_for_handles_skill_without_references():
    """A skill with no .references attr should be scored on description/body only."""

    class _BareSkill:
        name = "error-rate-spike"
        description = "error-rate 5xx spike triage."
        body = "# Error-Rate Spike\nTriage 5xx errors."
        base_dir = "/tmp"
        # NOTE: no .references attr at all

    class _BareStore:
        def catalog(self):
            return [(_BareSkill.name, _BareSkill.description)]

        def get(self, name):
            return _BareSkill() if name == _BareSkill.name else None

        def reference_text(self, skill, rel_path):
            return None

    recaller = SkillRecaller(_BareStore())
    result = recaller.best_for("5xx error spike")
    assert result is not None
    name, body = result
    assert name == "error-rate-spike"
    assert "Error-Rate Spike" in body


def test_best_for_empty_store_returns_none():
    empty = _FakeStore(skills=[])
    assert SkillRecaller(empty).best_for("5xx error spike") is None


def test_best_for_empty_title_returns_none(recaller: SkillRecaller):
    assert recaller.best_for("") is None
    assert recaller.best_for("   ") is None


# --------------------------------------------------------------------------- #
# Reference read caching / laziness
# --------------------------------------------------------------------------- #
def test_reference_peek_is_cached_across_queries(recaller: SkillRecaller):
    store = recaller._store  # type: ignore[attr-defined]
    # First call scores all candidates → populates the cache.
    recaller.best_for("5xx error spike")
    first_calls = len(store.reference_text_calls)
    # Second identical query should reuse the cached full text for both scoring
    # AND body resolution → zero new store reads.
    recaller.best_for("5xx error spike")
    second_calls = len(store.reference_text_calls)
    assert second_calls == first_calls


def test_winner_reference_read_only_once():
    """The winning reference must be read from the store exactly once.

    Scoring peeks the reference and body resolution needs the same text; the
    cache must serve both so the store is not hit twice for the winner.
    """
    store = _FakeStore(
        skills=[
            _FakeSkill(
                name="error-rate-spike",
                description="Generic error SOP.",
                body="short fallback",
                references=["sop.md"],
            ),
        ],
        refs={"error-rate-spike": {"sop.md": "# error 5xx spike\nreal triage steps"}},
    )
    recaller = SkillRecaller(store)
    recaller.best_for("5xx error spike")
    # Exactly one store read for the winning reference (scoring + body share it).
    assert len(store.reference_text_calls) == 1
    assert store.reference_text_calls[0] == ("error-rate-spike", "sop.md")


# --------------------------------------------------------------------------- #
# Tie-breaking: a reference that ties the description wins (returns ref body)
# --------------------------------------------------------------------------- #
def test_reference_wins_on_tie_returns_reference_body():
    """When description and reference carry equal routing weight, the more
    specific reference body is returned, not the generic skill body."""
    store = _FakeStore(
        skills=[
            _FakeSkill(
                name="error-rate-spike",
                description="error 5xx",  # same routing tokens as the reference
                body="# GENERIC SKILL BODY",
                references=["sop.md"],
            ),
        ],
        # Reference heading has identical routing tokens → ties description.
        refs={"error-rate-spike": {"sop.md": "error 5xx specific triage"}},
    )
    recaller = SkillRecaller(store)
    result = recaller.best_for("error 5xx spike")
    assert result is not None
    name, body = result
    assert name == "error-rate-spike"
    # Reference body wins on the tie, not the generic skill body.
    assert body == "error 5xx specific triage"
    assert "GENERIC" not in body


# --------------------------------------------------------------------------- #
# Tokenizer / bilingual awareness (indirectly via routing, plus unit checks)
# --------------------------------------------------------------------------- #
def test_tokenizer_handles_bilingual_mix():
    from rca_agent.skills.recall import _tokenize

    toks = _tokenize("checkout 错误次数告警 5xx")
    # CJK is tokenized into individual ideographs (mirrors the memory module's
    # regex) so localized text is searchable char-by-char.
    assert "错" in toks and "误" in toks
    assert "5xx" in toks
    assert "checkout" in toks


def test_tokenizer_drops_stopwords():
    from rca_agent.skills.recall import _tokenize

    toks = _tokenize("the error is a spike")
    assert "the" not in toks
    assert "a" not in toks
    assert "error" in toks
    assert "spike" in toks


def test_keyword_overlap_substring_aware():
    from rca_agent.skills.recall import _keyword_overlap

    # "oom" is a prefix of "oomkilled" → should match (abbreviation).
    score = _keyword_overlap(["oom"], ["oomkilled", "crash"])
    assert score > 0.0
    # No overlap.
    assert _keyword_overlap(["latency"], ["log", "trace"]) == 0.0


def test_keyword_overlap_prefix_only_no_arbitrary_substring():
    """Arbitrary substring containment (log ⊂ catalog) must NOT score."""
    from rca_agent.skills.recall import _keyword_overlap

    # "log" is a substring of "catalog" but NOT a prefix → noise, no score.
    assert _keyword_overlap(["log"], ["catalog", "inventory"]) == 0.0
    # "rt" is a substring of "report" but not a prefix → no score.
    assert _keyword_overlap(["rt"], ["report", "start"]) == 0.0
    # Prefix match still works: "crash" prefixes "crashloop".
    assert _keyword_overlap(["crash"], ["crashloop"]) > 0.0
