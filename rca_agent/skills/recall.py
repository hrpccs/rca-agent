"""Skill recall engine — match an alert to the single best troubleshooting SOP.

This module powers the harness-side "skill activation" step required by the
agentskills.io spec (keyword/relevance matching is explicitly permitted). At the
start of an RCA run the agent calls :meth:`SkillRecaller.best_for` with the alert
title + signals; the recaller returns the one best-matched troubleshooting SOP
body that the harness injects into the LLM context.

Why keyword + substring matching (not embeddings)?
  The skill catalog is small (a handful of SOPs) and the router vocabulary is
  narrow and bilingual (error/错误/5xx, latency/超时/RT, traffic/流量, pod/crash/
  OOM/重启, trace/链路, log/日志). For a corpus this tiny, IDF estimates are
  unreliable and an exact keyword match is both more predictable and cheaper than
  a vector model — and it reproduces the reference rca-diagnose keyword router
  directly from overlap, with no extra dependency. A future embedding backend can
  satisfy the same API by replacing the scorer.

The store is duck-typed (sibling S1 owns the concrete :class:`SkillStore`); we
never import the memory module's tokenizer so this unit stays decoupled and
independently mergeable.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Protocol, runtime_checkable

__all__ = ["SkillRecaller", "DEFAULT_MAX_CHARS", "MIN_SCORE_THRESHOLD"]

logger = logging.getLogger(__name__)

# Body cap (chars). Overridable via env so operators can tune context budget
# without a code change. Default ~6000 keeps a single SOP well within the LLM
# context window while leaving room for the case's own evidence.
DEFAULT_MAX_CHARS = 6000
_MAX_CHARS_ENV = "RCA_SKILL_MAX_CHARS"

# Below this best score, return None instead of force-injecting an irrelevant SOP.
# A score of, say, 0.1 means "barely any query term hit anything" — injecting a
# weak match would add noise to the LLM context, so we gate on a small floor.
MIN_SCORE_THRESHOLD = 0.15

# Heading/first-lines peek when scoring a reference. Reading the whole SOP body
# for *every* candidate reference would be wasteful; the heading + opening lines
# carry the routing signal, so we peek cheaply and only fetch the full body for
# the single winner afterwards.
_REFERENCE_PEEK_CHARS = 500

# Router keyword weights. These mirror the bundled rca-diagnose SOP keyword router
# (error-rate alert → error-rate-spike; latency/RT/超时 → latency-spike; traffic-
# drop/流量下跌 → traffic-drop; pod crash/OOM/CrashLoop → pod-crash; logs → log-
# analysis; traces → trace-analysis; else → rca-framework). Weighting them above
# generic terms makes the intended SOP win even when several skills mention the
# same surface words (e.g. "error" appears in many SOPs, but "5xx"/"错误次数" is
# the discriminative error-rate signal). Bilingual: English + Chinese.
_ROUTER_KEYWORDS: dict[str, float] = {
    # error-rate family
    "error": 2.5,
    "errors": 2.5,
    "错误": 2.5,
    "5xx": 3.0,
    "500": 2.0,
    "error_rate": 3.0,
    "error-rate": 3.0,
    "errorrate": 3.0,
    "spike": 1.5,
    "spiked": 1.5,
    # latency family
    "latency": 3.0,
    "超时": 3.0,
    "timeout": 3.0,
    "timeouts": 3.0,
    "rt": 2.0,
    "响应时间": 3.0,
    "p99": 2.5,
    "p95": 2.0,
    "slow": 1.5,
    "slowness": 1.5,
    "抖动": 2.0,
    "jitter": 2.0,
    # traffic-drop family
    "traffic": 2.5,
    "流量": 2.5,
    "drop": 2.5,
    "dropped": 2.0,
    "下跌": 3.0,
    "decline": 2.0,
    "qps": 2.0,
    "traffic_drop": 3.0,
    # pod-crash family
    "pod": 2.5,
    "crash": 3.0,
    "crashes": 2.5,
    "crashloop": 3.0,
    "oom": 3.0,
    "oomkilled": 3.0,
    "重启": 2.5,
    "restart": 2.0,
    "restarts": 2.0,
    "killed": 1.5,
    "kubernetes": 1.0,
    "k8s": 1.0,
    # trace family
    "trace": 3.0,
    "traces": 2.5,
    "链路": 3.0,
    "tracing": 2.0,
    "span": 1.5,
    # log family
    "log": 3.0,
    "logs": 2.5,
    "日志": 3.0,
    "logging": 2.0,
}

# Latin/alphanumeric runs OR individual CJK ideographs (so localized — incl.
# Chinese, the dominant language of the corpus — is searchable character-by-
# character rather than dropped entirely). Mirrors the memory module's regex so
# the same alert tokenizes identically here.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿㐀-䶿가-힯]")

# Small stoplist shared with the memory module's flavor: function words that
# carry no routing signal and would otherwise dilute the overlap score.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "without",
        "from",
        "into",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "not",
        "no",
        "do",
        "does",
        "did",
        "may",
        "might",
        "can",
        "could",
        "should",
        "would",
        "will",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "them",
        "his",
        "her",
        "their",
        "our",
        "your",
        "my",
        "me",
        "us",
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
    ]
)


def _tokenize(text: str) -> list[str]:
    """Tokenize a string into lowercase alnum runs + individual CJK ideographs.

    Stoplisted. Reimplemented locally (NOT imported from the memory module) so
    this unit stays decoupled and independently mergeable; the regex mirrors the
    memory tokenizer so the same alert produces the same tokens.
    """
    if not text:
        return []
    return [
        tok for tok in (t.lower() for t in _TOKEN_RE.findall(text)) if tok and tok not in _STOPWORDS
    ]


# Prefix-only matches count at this fraction of the term's full weight. A
# whole-token match (``oom`` == ``oom``) means the doc is *about* the term; a
# prefix match (``oom`` ⊂ ``oomkilled``) is a strong abbreviation signal.
# Arbitrary substring matches (``log`` ⊂ ``catalog``) are NOT counted — they
# are noise. Discounting prefix hits below whole-token keeps the discriminative
# SOP ahead when another doc mentions the term only as a word fragment.
_SUBSTRING_DISCOUNT = 0.5


def _keyword_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Overlap: weighted fraction of query tokens present in the doc.

    For each query token: a whole-token match scores full weight; a prefix
    match (doc token starts with the query token, e.g. ``oom`` ⊂
    ``oomkilled``, ``crash`` ⊂ ``crashloop``) scores ``_SUBSTRING_DISCOUNT`` ×
    weight. Arbitrary substring containment (``log`` ⊂ ``catalog``) is NOT
    scored — it is noise, not evidence the doc is about the term. Each query
    token is weighted by the router table: discriminative router terms (5xx,
    latency, 超时, …) outweigh generic words so the intended SOP wins ties.
    """
    if not query_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(doc_tokens)
    if not dset:
        return 0.0
    total_weight = 0.0
    hit_weight = 0.0
    for q in qset:
        w = _ROUTER_KEYWORDS.get(q, 1.0)
        total_weight += w
        if q in dset:
            hit_weight += w
        elif any(d.startswith(q) for d in dset):
            # Prefix containment: the query term is a PREFIX of a doc token
            # (``oom`` ⊂ ``oomkilled``, ``crash`` ⊂ ``crashloop``). Prefix-only
            # matching avoids the noise of arbitrary substring containment
            # (``log`` ⊂ ``catalog``, ``rt`` ⊂ ``report``), which would let a
            # short routing term spuriously match unrelated words.
            hit_weight += w * _SUBSTRING_DISCOUNT
    if total_weight <= 0.0 or hit_weight <= 0.0:
        return 0.0
    # Weighted recall: what fraction of the query's routing mass was matched.
    return hit_weight / total_weight


# --------------------------------------------------------------------------- #
# Duck-typed store / skill protocols (documentation only — no runtime check on
# the skill object's concrete type; getattr is used defensively at call sites).
# --------------------------------------------------------------------------- #
@runtime_checkable
class _SkillLike(Protocol):
    name: str
    description: str
    body: str
    base_dir: str


class SkillRecaller:
    """Recall the single best troubleshooting SOP for an alert.

    Wraps a duck-typed store (sibling S1's :class:`SkillStore`) and exposes the
    API the agent (sibling S4) consumes. The store must provide:

      * ``catalog() -> list[tuple[str, str]]`` — ``(name, description)`` pairs.
      * ``get(name) -> Skill-like`` with ``.name/.description/.body/.base_dir``
        and optionally ``.references: list[str]``.
      * ``reference_text(skill, rel_path) -> str | None`` — body of a reference
        file, or ``None`` if absent.

    All store access is defensive: a store missing ``references`` /
        ``reference_text`` falls back to scoring skill descriptions/bodies, and
    any store error is caught (returning ``None``) so recall never crashes the
    RCA run.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        # Cache of reference FULL text keyed by (skill_name, rel_path). Storing
        # the full text (not a truncated peek) lets one store read serve both
        # the scoring peek (sliced on read-out) and the winner's body — so the
        # winning reference is never re-read from the store.
        self._ref_peek_cache: dict[tuple[str, str], str | None] = {}

    # ------------------------------------------------------------------ #
    # Public API (sibling S4 depends on this EXACT shape)
    # ------------------------------------------------------------------ #
    def catalog(self) -> list[tuple[str, str]]:
        """Delegate to ``store.catalog()`` — list of ``(name, description)``."""
        try:
            return list(self._store.catalog())
        except Exception:  # noqa: BLE001 — recall must never crash the run
            logger.warning("skill recaller: store.catalog() raised", exc_info=True)
            return []

    def best_for(
        self,
        alert_title: str,
        signals: list[str] | None = None,
    ) -> tuple[str, str] | None:
        """Return ``(skill_name, sop_body)`` for the best-matched SOP, or None.

        Builds the query from ``alert_title`` + ``signals``; for each skill
        scores its ``description`` AND each of its ``references`` (filename +
        heading/first-lines peek via ``store.reference_text``); picks the single
        best ``(skill, reference)`` pair. The returned body is the matched
        reference's text (or the skill body when no reference scored better),
        truncated to the char cap with a truncation note. Returns ``None`` when
        the best score is below :data:`MIN_SCORE_THRESHOLD` (so nothing is
        force-injected when the alert is irrelevant to every SOP).
        """
        try:
            query = self._build_query(alert_title, signals)
            if not query:
                return None
            catalog = self._safe_catalog_skills()
            if not catalog:
                return None

            query_tokens = _tokenize(query)
            if not query_tokens:
                return None

            best = self._rank(query_tokens, catalog)
            if best is None:
                return None
            skill, best_ref, score = best
            if score < MIN_SCORE_THRESHOLD:
                # Below threshold: the alert doesn't meaningfully match any SOP.
                # Injecting a weak match would add noise, so return None.
                return None
            body = self._resolve_body(skill, best_ref)
            if body is None:
                return None
            return skill.name, self._truncate(body)
        except Exception:  # noqa: BLE001 — defensive: recall never crashes RCA
            logger.warning("skill recaller: best_for() raised", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_query(alert_title: str, signals: list[str] | None) -> str:
        parts = [alert_title or ""]
        if signals:
            parts.extend(str(s) for s in signals if s)
        return " ".join(p for p in parts if p).strip()

    def _safe_catalog_skills(self) -> list[Any]:
        """Resolve catalog names to skill objects, skipping any that error."""
        skills: list[Any] = []
        try:
            entries = self._store.catalog()
        except Exception:  # noqa: BLE001
            logger.warning("skill recaller: store.catalog() raised", exc_info=True)
            return skills
        for entry in entries:
            # catalog() yields (name, description); tolerate malformed rows.
            try:
                name = entry[0] if isinstance(entry, (tuple, list)) else entry
            except Exception:  # noqa: BLE001
                continue
            if not name:
                continue
            try:
                skill = self._store.get(name)
            except Exception:  # noqa: BLE001
                logger.warning("skill recaller: store.get(%r) raised", name, exc_info=True)
                continue
            if skill is not None:
                skills.append(skill)
        return skills

    def _rank(
        self, query_tokens: list[str], skills: list[Any]
    ) -> tuple[Any, str | None, float] | None:
        """Score every (skill, reference) pair; return the best.

        Returns ``(skill, best_ref_relpath_or_None, score)``. ``best_ref`` is
        the relpath of the winning reference (used to fetch the full body), or
        ``None`` when the skill's own description/body outscored its references.
        """
        best_skill: Any | None = None
        best_ref: str | None = None
        best_score: float = -1.0

        for skill in skills:
            # Score the skill description (and body as a fallback signal).
            desc = self._skill_attr(skill, "description", "") or ""
            body = self._skill_attr(skill, "body", "") or ""
            desc_tokens = _tokenize(f"{desc} {body}")
            desc_score = _keyword_overlap(query_tokens, desc_tokens)
            if desc_score > best_score:
                best_score = desc_score
                best_skill = skill
                best_ref = None  # description won; no specific reference

            # Score each reference by filename + heading/first-lines peek.
            # A reference that TIES the best score wins (>=) so we return the
            # more specific reference body rather than the generic skill body
            # when description and reference carry equal routing weight.
            for ref in self._skill_references(skill):
                ref_text = self._reference_peek(skill, ref)
                # Filename often carries the routing signal (e.g.
                # "error-rate-spike.md"); include it explicitly.
                doc = f"{ref} {ref_text}" if ref_text else str(ref)
                ref_tokens = _tokenize(doc)
                ref_score = _keyword_overlap(query_tokens, ref_tokens)
                if ref_score >= best_score and ref_score > 0.0:
                    best_score = ref_score
                    best_skill = skill
                    best_ref = ref

        if best_skill is None or best_score <= 0.0:
            return None
        return best_skill, best_ref, best_score

    def _reference_full(self, skill: Any, rel_path: str) -> str | None:
        """Cached read of a reference's FULL text (single store read per ref).

        The cache stores the unbounded text so it serves BOTH the scoring peek
        (sliced to ``_REFERENCE_PEEK_CHARS`` by the caller) and the winner's
        body resolution — otherwise the winner would be re-read from the store
        a second time, which is exactly the double-I/O the cache exists to
        prevent. Returns ``None`` (cached) when the store has no such reference
        or the read raised.
        """
        name = self._skill_attr(skill, "name", "")
        cache_key = (str(name), str(rel_path))
        if cache_key in self._ref_peek_cache:
            return self._ref_peek_cache[cache_key]
        text: str | None = None
        try:
            raw = self._store.reference_text(skill, rel_path)
            if isinstance(raw, str):
                text = raw
        except Exception:  # noqa: BLE001 — missing reference must not crash recall
            logger.debug(
                "skill recaller: reference_text(%r, %r) raised",
                name,
                rel_path,
                exc_info=True,
            )
            text = None
        self._ref_peek_cache[cache_key] = text
        return text

    def _reference_peek(self, skill: Any, rel_path: str) -> str:
        """Bounded heading/first-lines slice of a reference for scoring."""
        full = self._reference_full(skill, rel_path)
        return full[:_REFERENCE_PEEK_CHARS] if full else ""

    def _resolve_body(self, skill: Any, best_ref: str | None) -> str | None:
        """Fetch the full body for the winner: reference text, else skill body.

        Uses the cached full reference text (so the winner is NOT re-read from
        the store — the scoring pass already populated the cache).
        """
        if best_ref:
            text = self._reference_full(skill, best_ref)
            if isinstance(text, str) and text.strip():
                return text
        # Fall back to the skill's own body when no reference won (or the
        # reference read failed). Never raise.
        body = self._skill_attr(skill, "body", "")
        return body if isinstance(body, str) and body.strip() else None

    @staticmethod
    def _skill_attr(skill: Any, name: str, default: Any) -> Any:
        """Defensive getattr: a skill-like object may omit optional attrs."""
        try:
            return getattr(skill, name, default)
        except Exception:  # noqa: BLE001 — property could raise; treat as absent
            return default

    @staticmethod
    def _skill_references(skill: Any) -> list[str]:
        """Return the skill's reference relpaths, or [] if absent."""
        refs = SkillRecaller._skill_attr(skill, "references", None)
        if not refs:
            return []
        if isinstance(refs, (list, tuple)):
            return [str(r) for r in refs if r]
        return []

    @staticmethod
    def _truncate(body: str) -> str:
        """Cap body length; append a truncation note when shortened.

        Cap is read once from ``RCA_SKILL_MAX_CHARS`` (default 6000) so operators
        can tune context budget without a code change. Non-positive / unparseable
        values fall back to the default (never "unbounded-zero" truncation).
        """
        cap = SkillRecaller._parse_max_chars(os.environ.get(_MAX_CHARS_ENV, ""))
        if cap <= 0 or len(body) <= cap:
            return body
        # Leave a little room for the note itself so the cut is clearly marked.
        return body[:cap] + "\n\n…[truncated]"

    @staticmethod
    def _parse_max_chars(raw: str) -> int:
        """Parse the char cap env var; default on any bad value.

        Non-positive or unparseable values resolve to :data:`DEFAULT_MAX_CHARS`
        so a misconfigured env var never silently zero-truncates the SOP body.
        """
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            return DEFAULT_MAX_CHARS
        return value if value > 0 else DEFAULT_MAX_CHARS
