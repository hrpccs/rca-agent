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
import re
from pathlib import Path

from rca_agent.config import get_settings
from rca_agent.contracts import MemoryItem, MemoryQuery, MemoryStore

__all__ = ["InMemoryStore", "build_memory"]

GLOBAL = "__global__"

# Tiny stoplist to keep TF-IDF term frequencies meaningful for short SRE texts.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with without from into
    is are was were be been being this that these those it its as not no do does
    did may might can could should would will i you he she we they them his her
    their our your my me us what why how when where which who whom whose
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Lowercase + word-boundary tokenization. Keeps alphanumerics."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


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
    """Fallback relevance: fraction of query tokens present in the doc
    (Jaccard-like). Used when the pool is tiny or the query is very short, where
    TF-IDF IDF estimates are unreliable."""
    if not query_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(doc_tokens)
    hits = qset & dset
    if not hits:
        return 0.0
    # Weight recall (how many query terms matched) plus a small precision term.
    return len(hits) / len(qset)


class InMemoryStore:
    """Reference :class:`MemoryStore` backed by an in-process dict.

    Layout: ``self._items[case_id] -> list[MemoryItem]``. The special
    ``"__global__"`` case_id holds cross-case / domain knowledge that is always
    part of every retrieval's candidate pool.
    """

    def __init__(self) -> None:
        self._items: dict[str, list[MemoryItem]] = {}

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index(self, items: list[MemoryItem]) -> None:
        """Append items; assign a stable id to any item missing one.

        Items are appended (never replaced) so per-run evidence accumulates over
        a case's lifetime. ``clear()`` is the intended reset path.
        """
        for idx, it in enumerate(items):
            if not it.id:
                # Assign a unique-ish id scoped to this index call. Pydantic
                # models are mutable by default, so we can set .id in place.
                it.id = f"mem-{len(self._items) + idx}"
            self._items.setdefault(it.case_id, []).append(it)

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

        # No query text: stable order, no scoring.
        if not q.text or not q.text.strip():
            for i in pool[:top_k]:
                i.score = 0.0
            return pool[:top_k]

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
        results: list[MemoryItem] = []
        for item, score in scored[:top_k]:
            item.score = round(score, 6)
            results.append(item)
        return results

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
            except OSError:
                continue
            kind, body = cls._parse_seed(raw)
            store.index([
                MemoryItem(
                    id=fp.stem,
                    case_id=case_id,
                    content=body.strip(),
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
        # Otherwise, parse leading ``key: value`` lines until a blank line.
        lines = text.splitlines()
        consumed = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # blank line ends any leading meta run
                if consumed:
                    break
                continue
            if ":" in stripped and not stripped.startswith("#"):
                key, _, value = stripped.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "kind":
                    kind = value or kind
                    consumed += 1
                    continue
                # unknown key but still a meta line: consume if at top
                if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_\-]*", key) and value:
                    consumed += 1
                    continue
            break
        if consumed:
            text = "\n".join(lines[consumed:]).lstrip("\n")
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
