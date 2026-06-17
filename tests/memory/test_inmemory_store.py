"""Unit tests for the in-memory agent memory store (TF-IDF retrieval)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rca_agent.contracts import MemoryItem, MemoryQuery, MemoryStore
from rca_agent.memory.inmemory_store import (
    GLOBAL,
    MAX_PER_BUCKET_ENV,
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


# --------------------------------------------------------------------------- #
# Bounded eviction (RCA_MEMORY_MAX_PER_BUCKET)
# --------------------------------------------------------------------------- #
@pytest.fixture
def _clear_cap_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure the cap env var is unset/clean for any test in this module.

    Tests that need a specific cap set it themselves via monkeypatch BEFORE
    constructing the store (the cap is parsed once at __init__).
    """
    monkeypatch.delenv(MAX_PER_BUCKET_ENV, raising=False)


def _store_with_cap(monkeypatch: pytest.MonkeyPatch, cap: int) -> InMemoryStore:
    """Construct an InMemoryStore with the given per-bucket cap env var."""
    monkeypatch.setenv(MAX_PER_BUCKET_ENV, str(cap))
    return InMemoryStore()


def test_cap_evicts_oldest_when_exceeded(monkeypatch: pytest.MonkeyPatch,
                                         _clear_cap_env):
    # cap=2: indexing 3 items in one bucket keeps the 2 NEWEST, evicts the OLDEST.
    store = _store_with_cap(monkeypatch, 2)
    store.index([
        MemoryItem(id="a", case_id="t001", content="alpha rtt first", kind="evidence"),
        MemoryItem(id="b", case_id="t001", content="bravo cpu second", kind="evidence"),
        MemoryItem(id="c", case_id="t001", content="charlie rtt third", kind="evidence"),
    ])
    bucket = store._items["t001"]
    assert [i.id for i in bucket] == ["b", "c"]  # 'a' (oldest) evicted
    # Oldest survivor is 'b', newest is 'c' — order preserved.
    assert len(bucket) == 2


def test_cap_evicted_oldest_is_not_retrievable(monkeypatch: pytest.MonkeyPatch,
                                               _clear_cap_env):
    # After eviction, the dropped item must not surface in retrieval.
    store = _store_with_cap(monkeypatch, 2)
    store.index([
        MemoryItem(id="old", case_id=GLOBAL, content="rtt latency old item",
                   kind="domain_fact"),
        MemoryItem(id="mid", case_id=GLOBAL, content="cpu contention mid item",
                   kind="domain_fact"),
        MemoryItem(id="new", case_id=GLOBAL, content="rtt cpu new item",
                   kind="domain_fact"),
    ])
    res = store.retrieve(MemoryQuery(case_id="t001", text="", top_k=10))
    ids = {i.id for i in res}
    assert "old" not in ids
    assert ids == {"mid", "new"}


def test_cap_survivors_still_rank_correctly(monkeypatch: pytest.MonkeyPatch,
                                            _clear_cap_env):
    # cap=2: after eviction, retrieve still ranks survivors by relevance.
    store = _store_with_cap(monkeypatch, 2)
    store.index([
        MemoryItem(id="a", case_id=GLOBAL, content="noise unrelated filler",
                   kind="domain_fact"),
        MemoryItem(id="b", case_id=GLOBAL, content="rtt spike network latency",
                   kind="domain_fact"),
        MemoryItem(id="c", case_id=GLOBAL, content="rtt cpu contention high",
                   kind="domain_fact"),
    ])
    # 'a' (oldest) evicted; querying for rtt should rank the rtt-bearing survivors.
    res = store.retrieve(
        MemoryQuery(case_id="t001", text="rtt cpu contention latency", top_k=5)
    )
    ids = [i.id for i in res]
    assert ids[0] in {"b", "c"}  # a survivor ranks top
    assert "a" not in ids
    # scores descending
    scores = [i.score for i in res if i.score is not None]
    assert scores == sorted(scores, reverse=True)


def test_cap_zero_default_is_unbounded(monkeypatch: pytest.MonkeyPatch,
                                       _clear_cap_env):
    # REGRESSION: cap=0 (default) never evicts, no matter how many items.
    monkeypatch.delenv(MAX_PER_BUCKET_ENV, raising=False)
    store = InMemoryStore()
    assert store._max_per_bucket == 0
    many = [
        MemoryItem(id=f"m{i}", case_id="t001", content=f"item number {i}",
                   kind="evidence")
        for i in range(50)
    ]
    store.index(many)
    assert len(store._items["t001"]) == 50
    assert [i.id for i in store._items["t001"]] == [f"m{i}" for i in range(50)]


def test_cap_applies_per_bucket_independently(monkeypatch: pytest.MonkeyPatch,
                                              _clear_cap_env):
    # The cap is PER BUCKET — multiple buckets each bounded separately.
    store = _store_with_cap(monkeypatch, 2)
    store.index([
        MemoryItem(id="a1", case_id="t001", content="alpha", kind="evidence"),
        MemoryItem(id="a2", case_id="t001", content="bravo", kind="evidence"),
        MemoryItem(id="a3", case_id="t001", content="charlie", kind="evidence"),
    ])
    store.index([
        MemoryItem(id="b1", case_id="t002", content="delta rtt", kind="evidence"),
        MemoryItem(id="b2", case_id="t002", content="echo cpu", kind="evidence"),
    ])
    assert [i.id for i in store._items["t001"]] == ["a2", "a3"]  # a1 evicted
    assert [i.id for i in store._items["t002"]] == ["b1", "b2"]  # under cap, untouched


def test_cap_invalid_env_defaults_unbounded(monkeypatch: pytest.MonkeyPatch,
                                            _clear_cap_env):
    # Non-integer / negative env values must NOT silently truncate memory.
    for bad in ("not-a-number", "-5", "", "0", "3.5"):
        monkeypatch.setenv(MAX_PER_BUCKET_ENV, bad)
        store = InMemoryStore()
        assert store._max_per_bucket == 0, f"bad value {bad!r} should be unbounded"
        store.index([
            MemoryItem(id=f"x{i}", case_id="t001", content=f"c{i}",
                       kind="evidence")
            for i in range(10)
        ])
        assert len(store._items["t001"]) == 10


def test_cap_load_seed_respects_cap(monkeypatch: pytest.MonkeyPatch,
                                    _clear_cap_env, tmp_path: Path):
    # Seed loading goes through index(), so the cap applies there too.
    monkeypatch.setenv(MAX_PER_BUCKET_ENV, "2")
    for i in range(5):
        (tmp_path / f"f{i}.md").write_text(f"body number {i}\n", encoding="utf-8")
    store = InMemoryStore.load_seed(tmp_path)
    assert len(store._items[GLOBAL]) == 2
    # newest 2 seeds survive (sorted by filename: f3, f4)
    assert {i.id for i in store._items[GLOBAL]} == {"f3", "f4"}


# --------------------------------------------------------------------------- #
# CJK ranking + tokenization
# --------------------------------------------------------------------------- #
def test_cjk_query_ranks_matching_doc_above_nonmatching():
    # A Chinese query should retrieve the matching Chinese doc above a
    # non-matching one (single-ideograph tokenization).
    store = InMemoryStore()
    store.index([
        MemoryItem(id="cn_match", case_id=GLOBAL,
                   content="网络延迟升高通常由CPU争用引起",
                   kind="domain_fact"),
        MemoryItem(id="cn_other", case_id=GLOBAL,
                   content="数据库连接池耗尽导致五错误",
                   kind="domain_fact"),
        MemoryItem(id="en", case_id=GLOBAL,
                   content="database connection pool exhaustion raises 5xx",
                   kind="domain_fact"),
    ])
    res = store.retrieve(
        MemoryQuery(case_id="t001", text="网络延迟CPU", top_k=3)
    )
    assert res, "expected at least one result"
    # The matching Chinese doc must rank strictly above the non-matching Chinese doc.
    ids = [i.id for i in res]
    assert "cn_match" in ids
    assert ids.index("cn_match") < ids.index("cn_other")


def test_cjk_mixed_with_ascii_tokenization():
    # Mixed CJK + ASCII content is searchable by either an ASCII or CJK query.
    store = InMemoryStore()
    store.index([
        MemoryItem(id="mix", case_id=GLOBAL,
                   content="checkout服务发生crashloop 网络延迟",
                   kind="domain_fact"),
        MemoryItem(id="noise", case_id=GLOBAL,
                   content="totally unrelated content here about cats",
                   kind="domain_fact"),
        MemoryItem(id="noise2", case_id=GLOBAL,
                   content="more filler about weather and rain",
                   kind="domain_fact"),
    ])
    # CJK query hits the Chinese ideographs in the mixed doc.
    res_cjk = store.retrieve(MemoryQuery(case_id="t001", text="网络延迟", top_k=3))
    assert res_cjk and res_cjk[0].id == "mix"
    # ASCII query hits the english tokens in the mixed doc.
    res_ascii = store.retrieve(MemoryQuery(case_id="t001", text="crashloop", top_k=3))
    assert res_ascii and res_ascii[0].id == "mix"


# --------------------------------------------------------------------------- #
# id monotonicity regression (guard the earlier counter bug)
# --------------------------------------------------------------------------- #
def test_ids_monotonic_unique_across_buckets_and_calls():
    # The counter is a monotonic id source shared across ALL buckets and
    # index() calls — ids must be unique and strictly increasing in the order
    # they were assigned (regression guard for the earlier bucket-length bug).
    store = InMemoryStore()
    items_in_order: list[MemoryItem] = []
    batch1 = [MemoryItem(id="", case_id="t001", content="a", kind="evidence")]
    batch2 = [MemoryItem(id="", case_id="t002", content="b", kind="evidence")]
    batch3 = [
        MemoryItem(id="", case_id=GLOBAL, content="c", kind="domain_fact"),
        MemoryItem(id="", case_id="t001", content="d", kind="evidence"),
    ]
    store.index(batch1); items_in_order.extend(batch1)
    store.index(batch2); items_in_order.extend(batch2)
    store.index(batch3); items_in_order.extend(batch3)
    ids = [it.id for it in items_in_order if it.id and it.id.startswith("mem-")]
    nums = [int(i.split("-", 1)[1]) for i in ids]
    assert nums == [1, 2, 3, 4], f"ids must be assigned 1..4 in order, got {nums}"
    assert len(set(nums)) == len(nums), "ids must be unique"


def test_id_counter_not_reset_by_bucket_swap():
    # Indexing into a fresh bucket after others must not reset the counter
    # (regression: counter used to be derived from bucket length).
    store = InMemoryStore()
    store.index([
        MemoryItem(id="", case_id="t001", content="a", kind="evidence"),
        MemoryItem(id="", case_id="t001", content="b", kind="evidence"),
    ])
    store.index([MemoryItem(id="", case_id="brand_new_bucket", content="c",
                            kind="evidence")])
    new_ids = [it.id for it in store._items["brand_new_bucket"]]
    assert new_ids == ["mem-3"], f"expected mem-3, got {new_ids}"


# --------------------------------------------------------------------------- #
# Empty / whitespace-only queries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("empty_text", ["", "   ", "\t\n", None])
def test_empty_or_whitespace_query_returns_no_error(empty_text):
    # An empty / whitespace-only / None query must not raise and must return
    # either [] (empty pool) or top_k items with score 0.0 (non-empty pool).
    store = InMemoryStore()
    _seed_corpus(store)
    res = store.retrieve(MemoryQuery(case_id="t001", text=empty_text, top_k=5))
    assert isinstance(res, list)
    if res:
        assert all(i.score == 0.0 for i in res)


def test_empty_query_on_empty_store_returns_empty_list():
    store = InMemoryStore()
    res = store.retrieve(MemoryQuery(case_id="nobody", text="   ", top_k=5))
    assert res == []
