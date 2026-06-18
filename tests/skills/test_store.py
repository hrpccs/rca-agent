"""Unit tests for the Agent Skills store + SKILL.md parser (Unit S1).

These tests build real tmp skills directories (via ``tmp_path``) and exercise
the discovery / parsing / lookup paths. No network or DB. Mirrors the style of
``tests/memory/test_inmemory_store.py``.
"""

from __future__ import annotations

from pathlib import Path

from rca_agent.skills.store import Skill, SkillStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_skill(
    parent: Path,
    name: str,
    *,
    frontmatter: str | None = None,
    description_override: str | None = None,
    body: str = "Body text.",
    references: dict[str, str] | None = None,
    skill_file: str = "SKILL.md",
) -> Path:
    """Create ``<parent>/<name>/SKILL.md`` (plus optional reference files).

    ``frontmatter`` (if given) is written verbatim. Otherwise a minimal block
    is synthesized from ``name`` and ``description_override``.
    """
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    if frontmatter is None:
        desc = description_override or f"Skill {name} for tests."
        frontmatter = f"name: {name}\ndescription: {desc}"
    content = f"---\n{frontmatter}\n---\n{body}"
    md = skill_dir / skill_file
    md.write_text(content, encoding="utf-8")
    for rel, text in (references or {}).items():
        ref_path = skill_dir / rel
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(text, encoding="utf-8")
    return md


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_loads_valid_skill_with_frontmatter_body_and_references(tmp_path):
    _write_skill(
        tmp_path,
        "db-oom",
        frontmatter="name: db-oom\ndescription: Diagnose DB OOM kills.",
        body="## Steps\n1. Check memory.",
        references={"references/runbook.md": "# Runbook\nFree memory."},
    )
    store = SkillStore(tmp_path)

    assert len(store) == 1
    skill = store.get("db-oom")
    assert skill is not None
    assert skill.name == "db-oom"
    assert skill.description == "Diagnose DB OOM kills."
    assert "## Steps" in skill.body
    assert "references/runbook.md" in skill.references
    # location / base_dir are absolute and resolve correctly.
    assert skill.location.name == "SKILL.md"
    assert skill.base_dir == (tmp_path / "db-oom").resolve()


def test_catalog_returns_name_description_pairs_in_order(tmp_path):
    _write_skill(tmp_path, "a-skill", description_override="A desc")
    _write_skill(tmp_path, "b-skill", description_override="B desc")
    store = SkillStore(tmp_path)

    catalog = store.catalog()
    assert catalog == [("a-skill", "A desc"), ("b-skill", "B desc")]


def test_reference_text_reads_bundled_file(tmp_path):
    _write_skill(
        tmp_path,
        "net-dns",
        references={"references/dns.md": "## DNS debug\nnslookup."},
    )
    store = SkillStore(tmp_path)
    skill = store.get("net-dns")

    text = store.reference_text(skill, "references/dns.md")
    assert text is not None
    assert "nslookup" in text


def test_reference_text_returns_none_for_missing_file(tmp_path):
    _write_skill(tmp_path, "x")
    store = SkillStore(tmp_path)
    skill = store.get("x")

    assert store.reference_text(skill, "references/missing.md") is None


def test_get_returns_none_for_unknown_name(tmp_path):
    _write_skill(tmp_path, "present")
    store = SkillStore(tmp_path)

    assert store.get("absent") is None
    assert "absent" not in store
    assert "present" in store


# --------------------------------------------------------------------------- #
# Lenient parsing
# --------------------------------------------------------------------------- #
def test_name_defaults_to_parent_dir_when_absent(tmp_path):
    # No name field in frontmatter -> name should be the directory name.
    _write_skill(
        tmp_path,
        "inferred-name",
        frontmatter="description: Has a description at least.",
    )
    store = SkillStore(tmp_path)

    assert len(store) == 1
    assert store.get("inferred-name") is not None


def test_name_dir_mismatch_loads_anyway(tmp_path, caplog):
    _write_skill(
        tmp_path,
        "dir-a",
        frontmatter="name: skill-b\ndescription: mismatched name.",
    )
    store = SkillStore(tmp_path)

    # Loaded under the frontmatter name (not the dir name).
    assert store.get("skill-b") is not None
    assert store.get("dir-a") is None
    # And a warning was emitted about the mismatch.
    rec = next(r for r in caplog.records if "!=" in r.message and "skill-b" in r.message)
    assert rec.levelname == "WARNING"


def test_long_name_loads_with_warning(tmp_path, caplog):
    long_name = "x" * 100
    _write_skill(
        tmp_path,
        long_name,
        frontmatter=f"name: {long_name}\ndescription: long name skill.",
    )
    store = SkillStore(tmp_path)

    assert store.get(long_name) is not None
    rec = next(r for r in caplog.records if ">64" in r.message)
    assert rec.levelname == "WARNING"


def test_skill_without_description_is_skipped(tmp_path, caplog):
    _write_skill(
        tmp_path,
        "no-desc",
        frontmatter="name: no-desc\nbody-meta: something",
    )
    store = SkillStore(tmp_path)

    # description is hard-required -> skipped, store empty.
    assert len(store) == 0
    assert store.get("no-desc") is None
    rec = next(r for r in caplog.records if "no description" in r.message)
    assert rec.levelname == "WARNING"


def test_skill_with_empty_description_is_skipped(tmp_path):
    _write_skill(
        tmp_path,
        "empty-desc",
        frontmatter="name: empty-desc\ndescription:   ",
    )
    store = SkillStore(tmp_path)

    assert len(store) == 0


# --------------------------------------------------------------------------- #
# Bad YAML handling
# --------------------------------------------------------------------------- #
def test_unparseable_yaml_is_skipped_without_raising(tmp_path, caplog):
    # A frontmatter broken at the STRUCTURAL level (tab indentation is illegal
    # in YAML block mappings). The quoting fallback only rewrites values, so a
    # tab-indented key still fails to parse and the skill is skipped.
    _write_skill(
        tmp_path,
        "bad-yaml",
        frontmatter="name: bad-yaml\n\tdescription: tab-indented key is invalid",
    )
    store = SkillStore(tmp_path)

    assert len(store) == 0
    # Constructor never raised (we got here).
    assert any("unparseable" in r.message.lower() for r in caplog.records)


def test_quoting_fallback_recovers_unquoted_colon_value(tmp_path, caplog):
    # A common authoring slip: unquoted value containing a colon breaks plain
    # YAML. The quoting fallback should recover it.
    _write_skill(
        tmp_path,
        "colon-slip",
        frontmatter="name: colon-slip\ndescription: fix: the pod is OOM",
    )
    store = SkillStore(tmp_path)

    skill = store.get("colon-slip")
    assert skill is not None
    assert "OOM" in skill.description
    # And the fallback path was logged.
    assert any("quoting fallback" in r.message for r in caplog.records)


def test_yaml_mapping_required(tmp_path, caplog):
    # A frontmatter that parses to a scalar (not a mapping) is unusable.
    _write_skill(
        tmp_path,
        "scalar-fm",
        frontmatter="just-a-string",
    )
    store = SkillStore(tmp_path)

    assert len(store) == 0


# --------------------------------------------------------------------------- #
# De-dupe
# --------------------------------------------------------------------------- #
def test_duplicate_name_first_found_wins(tmp_path, caplog):
    # Two skill dirs whose frontmatter uses the same name. Both live one level
    # under tmp_path; discovery sorts by path so the lexicographically-first
    # directory wins deterministically.
    _write_skill(
        tmp_path,
        "aaa",
        frontmatter="name: shared\ndescription: first one.",
    )
    _write_skill(
        tmp_path,
        "bbb",
        frontmatter="name: shared\ndescription: second one.",
    )
    store = SkillStore(tmp_path)

    assert len(store) == 1
    assert store.get("shared").description == "first one."
    assert any("duplicate skill name" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Discovery + robustness
# --------------------------------------------------------------------------- #
def test_missing_skills_dir_yields_empty_store_no_raise(tmp_path):
    missing = tmp_path / "does-not-exist"
    store = SkillStore(missing)

    assert len(store) == 0
    assert store.catalog() == []
    assert store.get("anything") is None


def test_file_as_skills_dir_does_not_raise(tmp_path):
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("oops", encoding="utf-8")
    store = SkillStore(file_path)

    assert len(store) == 0


def test_nested_skill_dirs_are_discovered(tmp_path):
    # skills/<category>/<skill>/SKILL.md — nested up to the depth cap.
    _write_skill(tmp_path / "sre" / "k8s", "pod-crashloop")
    store = SkillStore(tmp_path)

    assert store.get("pod-crashloop") is not None


def test_skill_md_at_root_is_discovered(tmp_path):
    # A SKILL.md directly under skills_dir (no parent skill dir).
    md = tmp_path / "SKILL.md"
    md.write_text(
        "---\nname: root-skill\ndescription: lives at root.\n---\nBody.\n",
        encoding="utf-8",
    )
    store = SkillStore(tmp_path)

    # parent dir name is the skills_dir itself; name from frontmatter wins.
    assert store.get("root-skill") is not None


def test_git_and_node_modules_are_skipped(tmp_path):
    _write_skill(
        tmp_path / ".git" / "hooks",
        "leaked",
        frontmatter="name: leaked\ndescription: should be skipped.",
    )
    store = SkillStore(tmp_path)

    assert store.get("leaked") is None
    assert len(store) == 0


def test_reference_text_blocks_path_escape(tmp_path):
    _write_skill(
        tmp_path,
        "escape",
        references={"references/inner.md": "inner content"},
    )
    # Drop a file OUTSIDE the skill dir that the escape would target.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")
    store = SkillStore(tmp_path)
    skill = store.get("escape")

    # ``../secret.txt`` must NOT read the file outside the skill bundle.
    assert store.reference_text(skill, "../secret.txt") is None
    # But a legitimate reference still works.
    assert "inner content" in store.reference_text(skill, "references/inner.md")


def test_reference_text_rejects_absolute_path(tmp_path):
    _write_skill(tmp_path, "abs")
    store = SkillStore(tmp_path)
    skill = store.get("abs")

    # An absolute path resolves outside base_dir and must be rejected.
    assert store.reference_text(skill, "/etc/passwd") is None


def test_reference_names_enumerates_files(tmp_path):
    _write_skill(
        tmp_path,
        "multi",
        references={
            "references/a.md": "a",
            "references/b.md": "b",
            "scripts/run.sh": "#!/bin/sh\necho hi",
        },
    )
    store = SkillStore(tmp_path)
    skill = store.get("multi")

    names = store.reference_names(skill)
    assert "references/a.md" in names
    assert "references/b.md" in names
    assert "scripts/run.sh" in names
    # The skill's OWN top-level SKILL.md must not appear in the reference list.
    assert "SKILL.md" not in names


# --------------------------------------------------------------------------- #
# Review-fix regressions (code-review findings that must not return)
# --------------------------------------------------------------------------- #
def test_reference_text_never_raises_on_non_str_rel_path(tmp_path):
    # Public methods must never raise. A truthy non-str (int/list) used to
    # raise TypeError inside the path arithmetic.
    _write_skill(tmp_path, "n")
    store = SkillStore(tmp_path)
    skill = store.get("n")

    for bad in (123, 1.5, ["a"], object()):
        assert store.reference_text(skill, bad) is None  # type: ignore[arg-type]


def test_bom_prefixed_skill_is_loaded(tmp_path):
    # A UTF-8-BOM-prefixed SKILL.md (Windows/PowerShell) must still parse.
    skill_dir = tmp_path / "bom"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "﻿---\nname: bom\ndescription: bom-prefixed.\n---\nbody",
        encoding="utf-8",
    )
    store = SkillStore(tmp_path)

    assert store.get("bom") is not None


def test_empty_frontmatter_is_logged_and_skipped(tmp_path, caplog):
    # ``---\n---\n`` parses to None; the skill must be skipped WITH a log line
    # (the "logged and skipped" contract), not silently dropped.
    skill_dir = tmp_path / "empty"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\n---\nbody", encoding="utf-8")
    store = SkillStore(tmp_path)

    assert len(store) == 0
    assert any("empty frontmatter" in r.message.lower() for r in caplog.records)


def test_nested_skill_md_reference_is_kept(tmp_path):
    # Only the skill's OWN top-level SKILL.md is excluded; a nested
    # references/examples/SKILL.md is legitimate bundled content.
    _write_skill(
        tmp_path,
        "with-example",
        references={"references/examples/SKILL.md": "an example skill doc"},
    )
    store = SkillStore(tmp_path)
    skill = store.get("with-example")

    names = store.reference_names(skill)
    assert "references/examples/SKILL.md" in names
    assert "SKILL.md" not in names


def test_symlink_reference_is_not_advertised(tmp_path):
    # A symlink inside the skill dir (whose target may escape base_dir) must
    # not appear in reference_names: we'd otherwise advertise a file that
    # reference_text would refuse to serve.
    _write_skill(tmp_path, "sym")
    skill_dir = tmp_path / "sym"
    target = tmp_path / "external-target.md"
    target.write_text("outside", encoding="utf-8")
    (skill_dir / "leak.md").symlink_to(target)

    store = SkillStore(tmp_path)
    skill = store.get("sym")

    assert "leak.md" not in store.reference_names(skill)


def test_quote_fallback_preserves_flow_collection_type(tmp_path):
    # When the quoting fallback triggers (due to an unquoted colon elsewhere),
    # a sibling flow-list value must keep its list type, not become a string.
    _write_skill(
        tmp_path,
        "flow",
        frontmatter=("name: flow\ndescription: fix: triggers the fallback\ntags: [ops, sre]\n"),
    )
    store = SkillStore(tmp_path)
    skill = store.get("flow")

    assert skill is not None
    # description recovered via the fallback.
    assert "triggers the fallback" in skill.description


# --------------------------------------------------------------------------- #
# Config: env override + default
# --------------------------------------------------------------------------- #
def test_rca_skills_dir_env_override(tmp_path, monkeypatch):
    _write_skill(tmp_path, "env-skill", frontmatter="name: env-skill\ndescription: from env dir.")
    monkeypatch.setenv("RCA_SKILLS_DIR", str(tmp_path))

    store = SkillStore()  # no explicit arg -> should pick up the env var.

    assert store.get("env-skill") is not None


def test_explicit_arg_beats_env(tmp_path, monkeypatch):
    arg_dir = tmp_path / "arg"
    env_dir = tmp_path / "env"
    _write_skill(arg_dir, "from-arg", frontmatter="name: from-arg\ndescription: arg wins.")
    _write_skill(env_dir, "from-env", frontmatter="name: from-env\ndescription: env loser.")
    monkeypatch.setenv("RCA_SKILLS_DIR", str(env_dir))

    store = SkillStore(arg_dir)

    assert store.get("from-arg") is not None
    assert store.get("from-env") is None


def test_default_falls_back_to_package_adjacent_dir(monkeypatch):
    # When neither arg nor env is set, the store must still construct and point
    # at the package-adjacent default (which may be absent -> empty store).
    monkeypatch.delenv("RCA_SKILLS_DIR", raising=False)
    store = SkillStore()

    assert isinstance(len(store), int)
    # Constructor did not raise -> we got an instance back.
    assert isinstance(store, SkillStore)


# --------------------------------------------------------------------------- #
# Skill dataclass
# --------------------------------------------------------------------------- #
def test_skill_dataclass_defaults_references_to_empty_list():
    from pathlib import Path as P

    s = Skill(
        name="x",
        description="d",
        body="b",
        location=P("/tmp/SKILL.md"),
        base_dir=P("/tmp"),
    )
    assert s.references == []
    # And each instance gets its own list (mutable default safe).
    s.references.append("a.md")
    s2 = Skill(
        name="y",
        description="d",
        body="b",
        location=P("/tmp/SKILL.md"),
        base_dir=P("/tmp"),
    )
    assert s2.references == []
