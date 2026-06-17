"""In-memory agent memory store with pure-python TF-IDF retrieval.

This is the reference implementation of the :class:`~rca_agent.contracts.MemoryStore`
Protocol. It stores SRE-style knowledge (runbooks, SOPs, domain facts) and
per-run observations in an in-process ``dict[case_id, list[MemoryItem]]``, with a
dedicated ``"__global__"`` bucket for cross-case / domain knowledge.

Retrieval is keyword-aware TF-IDF implemented in the stdlib (no external deps):
candidate pool = items for ``q.case_id`` + ``__global__``; filter by ``kind`` and
``entities``; rank by cosine-like weighted TF-IDF overlap with ``q.text``. The
interface is generic enough that a future embedding/vector backend can satisfy
the same Protocol — only this file changes when swapping storage.

Seed files (markdown) live in ``memory/seed/*.md`` and are loaded lazily; the
loader tolerates the directory being absent (another worker owns creating it).
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

from rca_agent.config import get_settings
from rca_agent.contracts import MemoryItem, MemoryQuery, MemoryStore

__all__ = ["InMemoryStore", "build_memory"]

GLOBAL = "__global__"

# Env var name for the optional per-bucket max-items cap. Default ``"0"`` means
# UNBOUNDED, preserving the original append-only behavior exactly.
MAX_PER_BUCKET_ENV = "RCA_MEMORY_MAX_PER_BUCKET"

# Tiny stoplist to keep TF-IDF term frequencies meaningful for short SRE texts.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with without from into
    is are was were be been being this that these those it its as not no do does
    did may might can could should would will i you he she we they them his her
    their our your my me us what why how when where which who whom whose
    """.split()
)

# Latin/alphanumeric runs OR individual CJK ideographs (so localized — incl.
# Chinese, the dominant language of the rca100 corpus — is searchable
# character-by-character rather than dropped entirely).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿㐀-䶿가-힯]")


def _tokenize(text: str) -> list[str]:
    """Unicode-aware tokenization.

    Latin/alphanumeric runs are lowercased; each CJK/Hangul ideograph is its own
    token. Keeps the content searchable for both English and Chinese queries.
    """
    if not text:
        return []
    return [t for t in (tok.lower() for tok in _TOKEN_RE.findall(text)) if t not in _STOPWORDS]


def _term_freq(tokens: list[str]) -> dict[str, float]:
    """Raw term frequency (count) per token, as a dict."""
    tf: dict[str, float] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0.0) + 1.0
    return tf


def _tfidf_score(query_tf: dict[str, float], doc_tf: dict[str, float],
                 idf: dict[str, float]) -> float:
    """Cosine-style weighted overlap between query and a single doc.

    Uses log-scaled TF * IDF weighting. Returns a non-negative similarity; the
    query "vector" weights are ``1`` for any present term (the query is short,
    so its TF is ignored in favour of pure IDF weighting) and the doc contributes
    its weighted TF. The result is normalized by the document's L2 norm so longer
    documents are not arbitrarily favored.
    """
    if not query_tf or not doc_tf:
        return 0.0
    # Document weights: log(1 + tf) * idf.
    doc_weights = {
        term: (1.0 + math.log(tf)) * idf.get(term, 0.0)
        for term, tf in doc_tf.items()
    }
    doc_norm = math.sqrt(sum(w * w for w in doc_weights.values())) or 1.0
    # Dot product: sum over query terms present in the doc of (idf * doc_weight).
    dot = 0.0
    for term in query_tf:
        if term in doc_weights:
            # query weight is idf (doc already carries tf + idf).
            q_w = idf.get(term, 0.0)
            dot += q_w * doc_weights[term]
    return dot / doc_norm


def _keyword_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Fallback relevance: fraction of query tokens present in the doc, with
    substring matching so a short query term that is a substring of a doc term
    (e.g. ``oom`` vs ``oomkill``) still scores. Used when the pool is tiny or the
    query is very short, where TF-IDF IDF estimates are unreliable."""
    if not query_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(doc_tokens)
    hits = 0
    doc_blob = " ".join(doc_tokens)
    for q in qset:
        # whole-token match, else substring containment (either direction for
        # short CJK chars / abbreviations).
        if q in dset or q in doc_blob or any(q in d or d in q for d in dset):
            hits += 1
    if not hits:
        return 0.0
    # Weight recall (how many query terms matched).
    return hits / len(qset)


class InMemoryStore:
    """Reference :class:`MemoryStore` backed by an in-process dict.

    Layout: ``self._items[case_id] -> list[MemoryItem]``. The special
    ``"__global__"`` case_id holds cross-case / domain knowledge that is always
    part of every retrieval's candidate pool.
    """

    def __init__(self) -> None:
        self._items: dict[str, list[MemoryItem]] = {}
        self._counter: int = 0  # monotonic id source (NOT bucket count)
        # Optional per-bucket FIFO cap. Parsed ONCE at init from the env var;
        # 0 (or any non-positive / unparseable value) means UNBOUNDED, which
        # preserves the original append-only behavior exactly.
        self._max_per_bucket: int = self._parse_max_per_bucket(
            os.environ.get(MAX_PER_BUCKET_ENV, "0")
        )

    @staticmethod
    def _parse_max_per_bucket(raw: str) -> int:
        """Parse ``RCA_MEMORY_MAX_PER_BUCKET`` once at init.

        Non-positive or unparseable values resolve to ``0`` = UNBOUNDED so a
        misconfigured env var never silently truncates memory.
        """
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            return 0
        return value if value > 0 else 0

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index(self, items: list[MemoryItem]) -> None:
        """Append items; assign a stable, unique id to any item missing one.

        Items are appended (never replaced) so per-run evidence accumulates over
        a case's lifetime. ``clear()`` is the intended reset path.

        When ``RCA_MEMORY_MAX_PER_BUCKET`` > 0, each bucket is bounded: after
        appending, the OLDEST items (FIFO — front of the list) are dropped until
        the bucket is at or below the cap. A cap of ``0`` (the default) leaves
        buckets UNBOUNDED, preserving the original behavior exactly.
        """
        for it in items:
            if not it.id:
                self._counter += 1
                it.id = f"mem-{self._counter}"
            bucket = self._items.setdefault(it.case_id, [])
            bucket.append(it)
            cap = self._max_per_bucket
            # FIFO eviction: drop oldest (front) items when the cap is exceeded.
            # ``cap == 0`` means unbounded — the guard short-circuits.
            if cap > 0 and len(bucket) > cap:
                # Slice off the surplus from the FRONT (oldest first).
                del bucket[: len(bucket) - cap]

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(self, q: MemoryQuery) -> list[MemoryItem]:
        """Rank the candidate pool against ``q.text`` and return the top_k.

        Candidate pool = items for ``q.case_id`` plus the ``__global__`` bucket.
        Filters: ``q.kind`` (exact match) and ``q.entities`` (item must contain
        every requested entity). Ranking uses a pure-python TF-IDF cosine-like
        score built over the candidate pool's contents; when the query is short
        or the pool is tiny it falls back to keyword-overlap. Each returned
        item carries its ``.score``.
        """
        # Build the candidate pool, de-duplicating by id so a global item does
        # not appear twice if it was also indexed under the case.
        pool: list[MemoryItem] = []
        seen_ids: set[str] = set()
        for bucket in (q.case_id, GLOBAL):
            for it in self._items.get(bucket, []):
                if it.id in seen_ids:
                    continue
                seen_ids.add(it.id)
                pool.append(it)

        # --- hard filters ---
        if q.kind:
            pool = [i for i in pool if i.kind == q.kind]
        if q.entities:
            wanted = {e for e in q.entities if e}
            if wanted:
                pool = [
                    i for i in pool
                    if wanted.issubset({e for e in i.entities if e})
                ]

        if not pool:
            return []

        top_k = max(0, q.top_k)
        if top_k == 0:
            return []

        # No query text: stable order, no scoring. Return copies so callers'
        # earlier results are not mutated by later retrievals.
        if not q.text or not q.text.strip():
            return [i.model_copy(update={"score": 0.0}) for i in pool[:top_k]]

        query_tokens = _tokenize(q.text)

        # Short queries or tiny pools can't give a reliable IDF estimate; use
        # the keyword-overlap fallback which still ranks keyword hits sensibly.
        use_tfidf = len(query_tokens) >= 2 and len(pool) >= 3
        if use_tfidf:
            scored = self._rank_tfidf(query_tokens, pool)
        else:
            scored = self._rank_keyword(query_tokens, pool)

        # Stable sort: score desc, then insertion order (preserved via Python's
        # stable sort keyed solely on -score).
        scored.sort(key=lambda pair: -pair[1])
        # Return copies with the score set, so stored items are never mutated
        # (a later retrieve() with different text must not retroactively change
        # a caller's held result scores).
        return [
            item.model_copy(update={"score": round(score, 6)})
            for item, score in scored[:top_k]
        ]

    @staticmethod
    def _rank_tfidf(query_tokens: list[str],
                    pool: list[MemoryItem]) -> list[tuple[MemoryItem, float]]:
        """TF-IDF cosine ranking over the candidate pool.

        IDF is computed from the pool itself (document frequency / N), which is
        exactly the classic TF-IDF setting and keeps the implementation
        dependency-free. N = pool size.
        """
        n_docs = len(pool)
        # Document frequency per term across the pool.
        df: dict[str, int] = {}
        doc_tokens: list[list[str]] = []
        for it in pool:
            toks = _tokenize(it.content)
            doc_tokens.append(toks)
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        # IDF with +1 smoothing so a term present in all docs isn't zeroed out.
        idf = {
            term: math.log((n_docs + 1) / (docf + 1)) + 1.0
            for term, docf in df.items()
        }
        query_tf = _term_freq(query_tokens)
        scored: list[tuple[MemoryItem, float]] = []
        for it, toks in zip(pool, doc_tokens):
            doc_tf = _term_freq(toks)
            scored.append((it, _tfidf_score(query_tf, doc_tf, idf)))
        return scored

    @staticmethod
    def _rank_keyword(query_tokens: list[str],
                      pool: list[MemoryItem]) -> list[tuple[MemoryItem, float]]:
        """Keyword-overlap fallback ranker."""
        scored: list[tuple[MemoryItem, float]] = []
        for it in pool:
            toks = _tokenize(it.content)
            scored.append((it, _keyword_overlap(query_tokens, toks)))
        return scored

    def retrieve_for_context(self, case_id: str, query: str,
                             top_k: int = 8) -> list[MemoryItem]:
        """Convenience: build a :class:`MemoryQuery` and delegate to retrieve."""
        return self.retrieve(
            MemoryQuery(case_id=case_id, text=query, top_k=top_k)
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def clear(self, case_id: str | None = None) -> None:
        """Clear one case's bucket, or all memory when ``case_id`` is None."""
        if case_id is None:
            self._items.clear()
        else:
            self._items.pop(case_id, None)

    # ------------------------------------------------------------------ #
    # Seed loading
    # ------------------------------------------------------------------ #
    @classmethod
    def load_seed(cls, path: str | Path,
                  case_id: str = GLOBAL) -> "InMemoryStore":
        """Build a store preloaded from markdown seed files.

        Each ``*.md`` file under ``path`` becomes a :class:`MemoryItem`:
        ``kind`` is read from a ``kind:`` front-matter line (defaulting to
        ``"domain_fact"``), ``case_id`` is the provided bucket (default
        ``__global__``), and ``content`` is the markdown body. A nonexistent
        ``path`` is a no-op (returns an empty store) so callers can point at a
        seed directory that another worker may or may not have created yet.
        """
        store = cls()
        root = Path(path)
        if not root.exists():
            return store
        if root.is_file():
            files = [root] if root.suffix == ".md" else []
        else:
            files = sorted(p for p in root.rglob("*.md") if p.is_file())

        for fp in files:
            try:
                raw = fp.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                # Bad/missing file — skip it, don't take down seed loading.
                continue
            kind, body = cls._parse_seed(raw)
            body = body.strip()
            if not body:
                # Degenerate (front-matter-only) seed: nothing retrievable; skip.
                continue
            store.index([
                MemoryItem(
                    id=fp.stem,
                    case_id=case_id,
                    content=body,
                    kind=kind,
                    source_tool="seed",
                    meta={"source_path": str(fp)},
                )
            ])
        return store

    @staticmethod
    def _parse_seed(raw: str) -> tuple[str, str]:
        """Split optional front-matter from a seed markdown file.

        Recognizes a leading block of ``key: value`` lines (optionally wrapped
        in ``---`` fences). Only ``kind`` is consumed; the rest of the body is
        returned verbatim as the item content.
        """
        kind = "domain_fact"
        text = raw
        # Strip a leading fenced front-matter block (---\n...\n---).
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                front = parts[1]
                text = parts[2]
                for line in front.splitlines():
                    key, _, value = line.partition(":")
                    if key.strip().lower() == "kind":
                        kind = value.strip() or kind
                        break
                return kind, text.lstrip("\n")
        # Otherwise: recognize ONLY a single leading ``kind: value`` line (the
        # first non-blank line). All other ``key: value`` lines are preserved as
        # body, so authored content like "service: checkout" / "alert: 5xx" is
        # never stripped as front-matter.
        lines = text.splitlines()
        start = 0
        while start < len(lines) and not lines[start].strip():
            start += 1
        if start < len(lines):
            stripped = lines[start].strip()
            if ":" in stripped and not stripped.startswith("#"):
                key, _, value = stripped.partition(":")
                if key.strip().lower() == "kind":
                    kind = value.strip() or kind
                    lines = lines[:start] + lines[start + 1:]
                    text = "\n".join(lines).lstrip("\n")
        return kind, text


def build_memory(backend: str | None = None) -> MemoryStore:
    """Construct a :class:`MemoryStore` for the requested backend.

    ``backend`` defaults to :attr:`Settings.memory_backend` (``"inmemory"``).
    Unknown backends raise :class:`NotImplementedError`; the interface is
    reserved so a future vector store can slot in behind the same Protocol.
    """
    name = backend or get_settings().memory_backend or "inmemory"
    name = name.strip().lower()
    if name == "inmemory":
        return InMemoryStore()
    raise NotImplementedError(f"memory backend not implemented: {name!r}")
