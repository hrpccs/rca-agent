"""Agent Skills store + ``SKILL.md`` parser (agentskills.io discovery).

Implements lifecycle Steps 1-2 of the agentskills.io skill model:

  * **Step 1 — Discover:** scan a skills directory for ``<skill>/SKILL.md``
    files (one level deep by default, with a bounded recursive fallback for
    nested layouts).
  * **Step 2 — Parse:** split YAML frontmatter (``---`` delimited) from the
    markdown body, ``yaml.safe_load`` the frontmatter, and build a
    :class:`Skill`.

This module is the self-contained discovery + parsing layer. A sibling unit
(S2) builds recall/matching on top of the store; S4 wires it into the agent.
**It is independently mergeable** — siblings duck-type it as
``from rca_agent.skills.store import SkillStore``.

Robustness policy (mirrors :mod:`rca_agent.memory.inmemory_store`):

  * A bad skill file is **logged and skipped, never raised**.
  * A missing / nonexistent skills directory is a **no-op** (empty store).
  * The constructor never raises on bad configuration; callers can construct
    ``SkillStore()`` unconditionally at startup.

Validation is deliberately **lenient** (see :meth:`SkillStore._parse`):

  * ``description`` is the ONLY hard-required field — without it the skill
    cannot be disclosed to users or matched by the recall layer, so it is
    SKIPPED (with a warning).
  * ``name`` defaults to the parent directory name when absent; mismatches
    between ``name`` and the directory name (and over-long names) merely warn
    — the skill still loads, because authored content shouldn't be invisible
    due to a cosmetic metadata slip.
  * Unparseable YAML is retried once with a quoting fallback; if still bad,
    the skill is SKIPPED.
  * Duplicate ``name`` collisions resolve first-found-wins with a warning.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["Skill", "SkillStore"]

logger = logging.getLogger(__name__)

# Env override for the skills directory. Empty/unset -> package-adjacent default.
SKILLS_DIR_ENV = "RCA_SKILLS_DIR"

# Cap on how many bundled reference files we enumerate per skill. Keeps the
# in-memory footprint bounded for skills that ship large asset trees.
_MAX_REFERENCES = 50

# Recursion depth cap for nested ``<skills_dir>/<name>/.../SKILL.md`` layouts.
# Skills are conventionally one level under the skills dir; we still allow a
# little nesting (e.g. ``skills/<category>/<skill>/SKILL.md``) but bound it so
# a pathological tree can't blow up discovery.
_MAX_DISCOVERY_DEPTH = 3

# Directory names to never descend into during discovery (VCS / build junk).
_SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv"})


@dataclass
class Skill:
    """A parsed agent skill (one ``SKILL.md``).

    Attributes:
        name: Skill identifier (frontmatter ``name``, falling back to the
            parent directory name). Used for de-dupe, lookup, and disclosure.
        description: Human-readable summary used for recall/matching and for
            disclosure to the user. Always non-empty (skills missing it are
            skipped at parse time).
        body: The markdown body (everything after the frontmatter fences).
        location: Absolute path to the source ``SKILL.md`` file.
        base_dir: Parent directory of ``SKILL.md`` — the root for resolving
            bundled reference files (``references/*.md``, scripts, etc.).
        references: Relative paths of enumerated bundled files under
            ``base_dir`` (capped at :data:`_MAX_REFERENCES`). May be empty.
    """

    name: str
    description: str
    body: str
    location: Path
    base_dir: Path
    references: list[str] = field(default_factory=list)


class SkillStore:
    """Discover and serve :class:`Skill` objects from a skills directory.

    Construction never raises: a missing/broken ``skills_dir`` yields an empty
    store. All public methods are safe to call on an empty store.
    """

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        # Resolution order: explicit arg > RCA_SKILLS_DIR env > package-adjacent
        # default (the repo-root ``skills/`` dir). The package-adjacent default
        # keeps the agent usable with zero configuration while letting ops
        # override the location per-deployment via env.
        resolved = skills_dir or os.environ.get(SKILLS_DIR_ENV)
        if not resolved:
            # Path(__file__)  = <repo>/rca_agent/skills/store.py
            # parents[0]      = <repo>/rca_agent/skills
            # parents[1]      = <repo>/rca_agent
            # parents[2]      = <repo>            <- repo root
            # We default to <repo>/skills (package-adjacent) so the agent works
            # with zero config, while ops can override via RCA_SKILLS_DIR.
            resolved = Path(__file__).resolve().parents[2] / "skills"

        self._skills_dir: Path = Path(resolved)
        # name -> Skill, first-found-wins. Insertion-ordered for stable catalog.
        self._skills: dict[str, Skill] = {}
        try:
            self._load()
        except Exception:  # noqa: BLE001 - constructor MUST NOT raise
            # Any unexpected failure during discovery is contained: an empty
            # store is always preferable to crashing agent startup.
            logger.exception(
                "skill discovery failed for dir %s; continuing with empty store",
                self._skills_dir,
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def catalog(self) -> list[tuple[str, str]]:
        """Return ``[(name, description), ...]`` for disclosure / matching.

        Stable, insertion-ordered (first-found order from discovery).
        """
        return [(s.name, s.description) for s in self._skills.values()]

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name, or ``None`` if unknown."""
        return self._skills.get(name)

    def reference_text(self, skill: Skill, rel_path: str) -> str | None:
        """Read a bundled reference file as text.

        ``rel_path`` is resolved against ``skill.base_dir``. Parent-directory
        escapes (``..``) and absolute paths are rejected so a crafted
        reference path can't exfiltrate files outside the skill bundle.

        Returns the file contents (utf-8), or ``None`` if the file is missing,
        unreadable, or escapes the skill directory. Never raises.
        """
        # Type/falsy guard FIRST: a truthy non-str (int, list, ...) would raise
        # TypeError on ``base / rel_path`` below, violating the never-raises
        # contract. ``skill.base_dir`` is always set at construction.
        if not isinstance(rel_path, str) or not rel_path or not skill.base_dir:
            return None
        try:
            # Normalize and confine to base_dir. We resolve relative to base,
            # then verify the resolved path is still inside base — this blocks
            # ``..`` segments and symlink-ish escapes alike. ``base_dir`` was
            # already .resolve()'d at construction; re-resolving is cheap and
            # picks up any deploy-time symlink swap.
            base = skill.base_dir.resolve()
            candidate = (base / rel_path).resolve()
        except (OSError, ValueError):
            return None
        # is_relative_to guard: rejects paths that escaped base_dir (covers
        # ``..`` escapes, absolute paths, and symlinks whose target resolves
        # outside the bundle). ``is_relative_to`` exists since py3.9; we target
        # py3.11, so no fallback is needed.
        if not candidate.is_relative_to(base):
            logger.warning(
                "reference path %r escapes skill base_dir %s; refusing",
                rel_path,
                base,
            )
            return None
        if not candidate.is_file():
            return None
        try:
            return candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning("could not read skill reference %s", candidate, exc_info=True)
            return None

    def reference_names(self, skill: Skill) -> list[str]:
        """Enumerate bundled reference files for ``skill`` as relative paths.

        Returns the cached list captured at load time (stored on
        ``skill.references``). This avoids re-walking the bundle on every call —
        discovery already paid that cost once.
        """
        return list(skill.references)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._skills

    # ------------------------------------------------------------------ #
    # Discovery + parse
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Discover ``SKILL.md`` files under ``self._skills_dir`` and parse each.

        A nonexistent directory is a no-op (logged at debug). Directory listing
        errors are contained per-entry.
        """
        root = self._skills_dir
        if not root.exists():
            # Absent skills dir is the common case during early development and
            # on deployments that don't ship skills — debug-level only.
            logger.debug("skills dir does not exist: %s", root)
            return
        if not root.is_dir():
            logger.warning("skills path is not a directory: %s", root)
            return

        for skill_md in self._discover(root):
            self._parse_and_register(skill_md)

    def _discover(self, root: Path) -> list[Path]:
        """Find ``SKILL.md`` files under ``root``.

        Conventional layout is ``<root>/<skill>/SKILL.md`` (one level deep).
        We also tolerate a bounded amount of nesting (up to
        :data:`_MAX_DISCOVERY_DEPTH`) so ``skills/<category>/<skill>/SKILL.md``
        works, while skipping VCS/build junk dirs.
        """
        found: list[Path] = []
        # Walk with an explicit depth budget to bound work on deep trees.
        # Each stack entry: (directory, depth) where depth counts levels below root.
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = list(current.iterdir())
            except (OSError, PermissionError):
                logger.warning("cannot list skills dir %s; skipping", current)
                continue
            for entry in entries:
                if entry.name in _SKIP_DIRS:
                    continue
                if entry.is_file() and entry.name == "SKILL.md":
                    found.append(entry)
                    continue
                if entry.is_dir() and depth < _MAX_DISCOVERY_DEPTH:
                    stack.append((entry, depth + 1))
        # Stable order so first-found-wins de-dupe is deterministic across runs.
        found.sort()
        return found

    def _parse_and_register(self, skill_md: Path) -> None:
        """Parse one ``SKILL.md`` and register it (or skip on hard failure)."""
        parent_name = skill_md.parent.name
        try:
            raw = skill_md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning("cannot read %s; skipping skill", skill_md)
            return

        frontmatter, body = self._split_frontmatter(raw)
        if frontmatter is None:
            # No frontmatter delimiters at all: treat the whole file as body and
            # synthesize a minimal metadata block (name = dir). A skill with no
            # frontmatter can still be valid as long as it has a description in
            # the body? No — description is a frontmatter field; without it we
            # cannot disclose. Skip with a warning.
            logger.warning("skill %s has no frontmatter; skipping", skill_md)
            return

        data = self._safe_yaml(frontmatter, skill_md)
        if data is None:
            return  # _safe_yaml already logged the skip reason
        if not isinstance(data, dict):
            logger.warning(
                "skill %s frontmatter is not a mapping (%s); skipping",
                skill_md,
                type(data).__name__,
            )
            return

        name = str(data.get("name") or parent_name).strip() or parent_name
        description_raw = data.get("description")
        description = str(description_raw).strip() if description_raw is not None else ""

        # description is the ONLY hard-required field: it drives disclosure and
        # recall matching. Absent/empty -> the skill is invisible and useless,
        # so we skip rather than load a broken entry.
        if not description:
            logger.warning(
                "skill %s (name=%s) has no description; skipping",
                skill_md,
                name,
            )
            return

        # Lenient warnings: load anyway, but surface the metadata smell.
        if name != parent_name:
            logger.warning(
                "skill %s: frontmatter name %r != directory name %r (loaded anyway)",
                skill_md,
                name,
                parent_name,
            )
        if len(name) > 64:
            # agentskills.io convention caps names at 64 chars; longer names
            # still work but may break downstream UIs, so warn.
            logger.warning(
                "skill %s: name %r is %d chars (>64); loaded anyway",
                skill_md,
                name,
                len(name),
            )

        base_dir = skill_md.parent.resolve()
        references = self._enumerate_references(base_dir)

        skill = Skill(
            name=name,
            description=description,
            body=body,
            location=skill_md.resolve(),
            base_dir=base_dir,
            references=references,
        )

        # De-dupe by name: first-found wins. Discovery order is deterministic
        # (sorted), so this is reproducible across runs.
        if name in self._skills:
            logger.warning(
                "duplicate skill name %r (in %s and %s); keeping first-found",
                name,
                self._skills[name].location,
                skill.location,
            )
            return
        self._skills[name] = skill

    # ------------------------------------------------------------------ #
    # Parse helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_frontmatter(raw: str) -> tuple[str | None, str]:
        """Split a ``SKILL.md`` into ``(frontmatter_text, body)``.

        Frontmatter is the text between the first two ``---`` lines when the
        file starts with ``---``. Returns ``(None, raw)`` when no frontmatter
        block is present (caller decides whether to skip). The body is returned
        stripped of leading/trailing whitespace.
        """
        # Tolerate a leading UTF-8 BOM (Windows/PowerShell-authored files):
        # ``read_text(encoding='utf-8')`` preserves it as ``﻿``, which would
        # otherwise defeat the ``startswith('---')`` check and silently skip an
        # otherwise-valid skill.
        if raw.startswith("﻿"):
            raw = raw[1:]
        # Require the file to START with a fence (no leading whitespace): a
        # ``---`` later in the body (e.g. a horizontal rule, or a padded
        # ``  ---  `` row in a markdown table) is NOT frontmatter. Matching the
        # raw start position (not ``.strip()``) is what distinguishes the two.
        if not raw.startswith("---"):
            return None, raw.strip()
        lines = raw.splitlines()
        # lines[0] is the opening fence. Find the closing fence: the next line
        # whose content (ignoring surrounding whitespace) is exactly ``---``.
        # We match on ``.strip() == '---'`` rather than ``== '---'`` so a fence
        # with trailing whitespace (common from some editors) is still detected;
        # a body horizontal rule would need leading whitespace to be ambiguous,
        # and bare ``---`` as an HR is conventionally preceded by a blank line
        # which lands it at a ``close_idx`` AFTER the real frontmatter.
        close_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                close_idx = i
                break
        if close_idx == -1:
            # Opening fence but no closing fence: malformed; treat as no FM.
            return None, raw.strip()
        front = "\n".join(lines[1:close_idx])
        body = "\n".join(lines[close_idx + 1 :])
        return front, body.strip()

    @staticmethod
    def _safe_yaml(frontmatter: str, source: Path) -> dict | None:
        """Parse frontmatter YAML leniently.

        First attempt is a plain ``yaml.safe_load``. If it fails (common cause:
        an unquoted value containing a ``:``), retry once with a quoting
        fallback that wraps each scalar-looking value in double quotes. Returns
        ``None`` (and logs a skip warning) only if both attempts fail, OR if
        the frontmatter is empty/whitespace-only (``yaml.safe_load('')`` returns
        ``None`` — treated as "no usable metadata", logged + skipped so the
        caller's "logged and skipped" contract holds).
        """
        try:
            data = yaml.safe_load(frontmatter)
        except yaml.YAMLError as first_err:
            # Fallback: quote bare values that contain colons/special chars.
            quoted = SkillStore._quote_fallback(frontmatter)
            try:
                data = yaml.safe_load(quoted)
                logger.warning(
                    "skill %s frontmatter needed the quoting fallback (original error: %s)",
                    source,
                    first_err,
                )
            except yaml.YAMLError:
                logger.warning(
                    "skill %s frontmatter is unparseable YAML even after the "
                    "quoting fallback; skipping",
                    source,
                )
                return None
        # ``yaml.safe_load`` returns None for empty/whitespace-only input. That
        # is not a usable frontmatter mapping — log + skip so a missing skill is
        # always traceable (matches the "logged and skipped" contract).
        if data is None:
            logger.warning(
                "skill %s has empty frontmatter; skipping",
                source,
            )
        return data  # type: ignore[no-any-return]

    @staticmethod
    def _quote_fallback(frontmatter: str) -> str:
        """Wrap bare YAML scalar values in double quotes as a last resort.

        Only rewrites lines that look like ``key: value`` where ``value`` is a
        non-quoted scalar; leaves already-quoted values, lists, and block
        scalars untouched. This recovers common authoring slips like
        ``description: fix: the pod is OOM`` (an unquoted colon in the value).
        """
        out: list[str] = []
        for line in frontmatter.splitlines():
            stripped = line.strip()
            # Skip blank lines, comments, list items, and already-quoted values.
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("- ")
                or stripped.startswith("-")
            ):
                out.append(line)
                continue
            key, sep, value = stripped.partition(":")
            if not sep:
                out.append(line)
                continue
            value = value.strip()
            if not value:
                out.append(line)
                continue
            # Already quoted, a block scalar marker, or a flow collection
            # (``[...]`` / ``{...}``) -> leave alone. Wrapping a flow collection
            # in quotes would turn ``tags: [a, b]`` into the string ``"[a, b]"``,
            # silently changing its type; preserving it lets the retry parse it
            # natively (it parsed fine on the first attempt — only a sibling
            # line triggered the fallback).
            if (
                value.startswith(('"', "'"))
                or value.startswith("[")
                or value.startswith("{")
                or value in ("|", ">", "|-", ">-", "|+", ">+")
            ):
                out.append(line)
                continue
            # Preserve indentation of the original line.
            indent = line[: len(line) - len(line.lstrip())]
            # Escape any embedded double quotes so the wrapped value is valid.
            escaped = value.replace('"', '\\"')
            out.append(f'{indent}{key}: "{escaped}"')
        return "\n".join(out)

    @staticmethod
    def _enumerate_references(base_dir: Path) -> list[str]:
        """List bundled reference files under ``base_dir`` as relative paths.

        Looks under conventional reference subdirs (``references/``,
        ``scripts/``, ``assets/``) plus any other nested files, skipping the
        skill's OWN top-level ``SKILL.md``, VCS/build junk dirs, and symlinks.
        Capped at :data:`_MAX_REFERENCES` to bound memory.

        Symlinks are skipped because a symlink whose target resolves outside
        ``base_dir`` would be advertised in the reference list but then refused
        by :meth:`reference_text`'s confinement guard — better to never list it
        than to expose a name whose backing file we won't serve. (Nested files
        named ``SKILL.md`` — e.g. ``references/examples/SKILL.md`` — ARE kept;
        only the skill's own top-level ``SKILL.md`` is excluded.)
        """
        refs: list[str] = []
        own_skill_md = (base_dir / "SKILL.md").resolve()
        try:
            for child in sorted(base_dir.rglob("*")):
                if not child.is_file():
                    continue
                # Skip symlinks: their target may live outside base_dir, and we
                # never want to advertise a reference we'd refuse to serve.
                if child.is_symlink():
                    continue
                # Exclude ONLY the skill's own top-level SKILL.md (by resolved
                # path, so a rename/symlink still matches). Nested SKILL.md
                # reference docs are legitimate bundled content.
                try:
                    if child.resolve() == own_skill_md:
                        continue
                except OSError:
                    # Unresolvable path (dangling link, perms): skip safely.
                    continue
                # Skip files inside junk dirs.
                if any(part in _SKIP_DIRS for part in child.parts):
                    continue
                try:
                    rel = child.relative_to(base_dir).as_posix()
                except ValueError:
                    continue
                refs.append(rel)
                if len(refs) >= _MAX_REFERENCES:
                    break
        except (OSError, PermissionError):
            logger.debug("could not enumerate references under %s", base_dir)
        return refs
