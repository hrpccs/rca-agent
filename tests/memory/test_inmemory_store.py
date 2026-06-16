"""Unit tests for the in-memory agent memory store (TF-IDF retrieval)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rca_agent.contracts import MemoryItem, MemoryQuery, MemoryStore
from rca_agent.memory.inmemory_store import (
    GLOBAL,
    InMemoryStore,
    build_memory,
)


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #
def test_satisfies_memory_store_protocol():
    store = InMemoryStore()
    assert isinstance(store, MemoryStore)


def test_build_memory_default_returns_inmemory():
    store = build_memory()
    assert isinstance(store, InMemoryStore)


def test_build_memory_explicit_inmemory():
    assert isinstance(build_memory("inmemory"), InMemoryStore)


def test_build_memory_unknown_backend_raises():
    with pytest.raises(NotImplementedError):
        build_memory("vector-blob")


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def test_index_appends_and_assigns_missing_ids():
    store = InMemoryStore()
    store.index([
        MemoryItem(id="", case_id=GLOBAL, content="alpha", kind="domain_fact"),
        MemoryItem(id="x2", case_id="t001", content="beta", kind="note"),
    ])
    g = store._items[GLOBAL]
    assert g[0].id  # missing id was assigned
    assert store._items["t001"][0].id == "x2"


def test_index_is_append_not_replace():
    store = InMemoryStore()
    store.index([MemoryItem(id="1", case_id="t001", content="a")])
    store.index([MemoryItem(id="2", case_id="t001", content="b")])
    assert [i.id for i in store._items["t001"]] == ["1", "2"]


# --------------------------------------------------------------------------- #
# Candidate pool + filters
# --------------------------------------------------------------------------- #
def _seed_corpus(store: InMemoryStore) -> None:
    store.index([
        MemoryItem(id="g1", case_id=GLOBAL,
                   content="RTT rise may be caused by CPU contention or network latency",
                   kind="domain_fact", entities=["rtt", "cpu"]),
        MemoryItem(id="g2", case_id=GLOBAL,
                   content="checkout pod restart loops indicate a crashloop backoff",
                   kind="runbook", entities=["pod", "crashloop"]),
        MemoryItem(id="g3", case_id=GLOBAL,
                   content="database connection pool exhaustion raises 5xx errors",
                   kind="runbook", entities=["db", "5xx"]),
        MemoryItem(id="c1", case_id="t001",
                   content="observed RTT spike on checkout service at 5am",
                   kind="evidence", entities=["rtt"]),
    ])


def test_retrieve_pool_combines_case_and_global():
    store = InMemoryStore()
    _seed_corpus(store)
    # A query with no meaningful text returns up to top_k items in stable order.
    res = store.retrieve(MemoryQuery(case_id="t001", text="", top_k=10))
    ids = {i.id for i in res}
    assert ids == {"g1", "g2", "g3", "c1"}


def test_retrieve_kind_filter():
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(
        MemoryQuery(case_id="t001", text="anything", kind="runbook", top_k=10)
    )
    assert {i.id for i in res} == {"g2", "g3"}
    assert all(i.kind == "runbook" for i in res)


def test_retrieve_entities_filter_requires_all():
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(
        MemoryQuery(case_id="t001", text="x", entities=["rtt", "cpu"], top_k=10)
    )
    assert {i.id for i in res} == {"g1"}


def test_retrieve_unknown_case_falls_back_to_global_pool():
    # The candidate pool is always q.case_id + __global__; an unknown case_id
    # therefore still surfaces global items.
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(MemoryQuery(case_id="missing", text="x", top_k=5))
    assert {i.id for i in res} == {"g1", "g2", "g3"}


def test_retrieve_truly_empty_when_global_also_empty():
    store = InMemoryStore()
    res = store.retrieve(MemoryQuery(case_id="missing", text="x", top_k=5))
    assert res == []


# --------------------------------------------------------------------------- #
# Ranking: TF-IDF relevance
# --------------------------------------------------------------------------- #
def test_retrieve_ranks_relevant_item_first():
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(
        MemoryQuery(case_id="t001", text="why is network latency high rtt", top_k=2)
    )
    # g1 talks about RTT + network latency -> should rank top.
    assert res[0].id == "g1"
    assert res[0].score is not None and res[0].score > 0


def test_retrieve_sets_score_on_results():
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(
        MemoryQuery(case_id="t001", text="rtt cpu contention latency", top_k=3)
    )
    assert all(i.score is not None for i in res)
    # Scores should be sorted descending.
    scores = [i.score for i in res]  # type: ignore[misc]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_top_k():
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(
        MemoryQuery(case_id="t001", text="latency rtt cpu contention network", top_k=1)
    )
    assert len(res) == 1


def test_retrieve_fallback_keyword_on_short_query():
    # A single-token query is too short for stable TF-IDF IDF; the keyword
    # fallback should still surface items containing that token.
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(MemoryQuery(case_id="t001", text="crashloop", top_k=4))
    assert res and res[0].id == "g2"
    assert res[0].score and res[0].score > 0


def test_retrieve_fallback_keyword_on_tiny_pool():
    store = InMemoryStore()
    store.index([
        MemoryItem(id="only", case_id=GLOBAL, content="memory pressure oom kill",
                   kind="domain_fact"),
    ])
    res = store.retrieve(MemoryQuery(case_id="t001", text="oom memory", top_k=1))
    assert res and res[0].id == "only"
    assert res[0].score and res[0].score > 0


def test_retrieve_no_text_returns_unscored_top_k():
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(MemoryQuery(case_id="t001", text=None, top_k=2))
    assert len(res) == 2
    assert all(i.score == 0.0 for i in res)


def test_global_items_not_duplicated_when_also_case_scoped():
    store = InMemoryStore()
    # Same id indexed under both global and case_id -> should appear once.
    item = MemoryItem(id="dup", case_id=GLOBAL, content="rtt latency cpu", kind="domain_fact")
    store.index([item])
    store.index([MemoryItem(id="dup", case_id="t001", content="rtt latency cpu",
                            kind="domain_fact")])
    res = store.retrieve(MemoryQuery(case_id="t001", text="rtt", top_k=5))
    ids = [i.id for i in res]
    assert ids.count("dup") == 1


# --------------------------------------------------------------------------- #
# Convenience + lifecycle
# --------------------------------------------------------------------------- #
def test_retrieve_for_context_builds_query():
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve_for_context("t001", "rtt latency cpu contention", top_k=2)
    assert res and res[0].id == "g1"


def test_clear_single_case():
    store = InMemoryStore()
    _seed_corpus(store)
    store.clear("t001")
    assert "t001" not in store._items
    assert GLOBAL in store._items  # global untouched


def test_clear_all():
    store = InMemoryStore()
    _seed_corpus(store)
    store.clear()
    assert store._items == {}


# --------------------------------------------------------------------------- #
# Seed loading
# --------------------------------------------------------------------------- #
def test_load_seed_handles_missing_dir(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    store = InMemoryStore.load_seed(missing)
    assert isinstance(store, InMemoryStore)
    assert store._items == {}


def test_load_seed_reads_markdown_files(tmp_path: Path):
    (tmp_path / "rtt.md").write_text(
        "---\nkind: domain_fact\n---\nRTT rise may be CPU contention or network latency.\n",
        encoding="utf-8",
    )
    (tmp_path / "crashloop.md").write_text(
        "kind: runbook\n\nPod restart loops indicate a crashloop backoff.\n",
        encoding="utf-8",
    )
    store = InMemoryStore.load_seed(tmp_path)
    by_id = {i.id: i for i in store._items[GLOBAL]}
    assert by_id["rtt"].kind == "domain_fact"
    assert "RTT rise" in by_id["rtt"].content
    assert by_id["crashloop"].kind == "runbook"
    assert "crashloop backoff" in by_id["crashloop"].content
    assert all(i.case_id == GLOBAL for i in by_id.values())


def test_load_seed_defaults_kind_when_absent(tmp_path: Path):
    (tmp_path / "bare.md").write_text("Just a body with no front matter.\n",
                                      encoding="utf-8")
    store = InMemoryStore.load_seed(tmp_path)
    item = store._items[GLOBAL][0]
    assert item.kind == "domain_fact"
    assert "Just a body" in item.content


def test_load_seed_single_file(tmp_path: Path):
    fp = tmp_path / "one.md"
    fp.write_text("---\nkind: sop\n---\nStep one: check CPU.\n", encoding="utf-8")
    store = InMemoryStore.load_seed(fp)
    assert len(store._items[GLOBAL]) == 1
    assert store._items[GLOBAL][0].kind == "sop"


def test_loaded_seed_is_retrievable(tmp_path: Path):
    (tmp_path / "rtt.md").write_text(
        "---\nkind: domain_fact\n---\nRTT rise may be CPU contention or network latency.\n",
        encoding="utf-8",
    )
    store = InMemoryStore.load_seed(tmp_path)
    res = store.retrieve(
        MemoryQuery(case_id="t001", text="rtt cpu contention latency", top_k=3)
    )
    assert res and res[0].id == "rtt"
