"""SQL Files adapter — works with any .sql files in a git repo."""

from __future__ import annotations

import fnmatch
import glob
import itertools
from pathlib import Path

from adapters import ChangedSQL
from config import ActionConfig
from gitutil import GitError, explain_git_failure, run_git

# A file the base branch changed after the PR's branch started, which the PR changed too: the
# merge holds both changes, so neither the branch point's version nor the base's is "before"
BASE_MOVED_REASON = (
    "the base branch changed this file after this PR's branch started, so this PR's own change can't be "
    "priced: update the branch (merge or rebase the base branch into it)"
)


def add_unknown_reason(metadata: dict, reason: str, first: bool = False) -> None:
    """Add `reason` to an item's `unknown_reason`: after any reason it already has, or before
    it (`first`) when it must survive the comment's cut of a long reason."""
    parts = (reason, metadata.get("unknown_reason")) if first else (metadata.get("unknown_reason"), reason)
    metadata["unknown_reason"] = "; ".join(r for r in parts if r)


def find_changed_sql(config: ActionConfig) -> list[ChangedSQL]:
    """
    Find SQL files changed between the base and HEAD.
    Returns list of ChangedSQL with production and PR versions.
    Raises GitError when git cannot produce a diff (never reported as "no changes").

    Both the list of changed files and each file's earlier version come from one commit,
    where the PR's branch meets the base (`ActionConfig.compare_base`), so a change the base
    branch gained after the branch started is never read as one this PR made. A file the
    base changed after that commit and the PR changed too is unknown cost (`BASE_MOVED_REASON`):
    after the merge it holds both changes, which neither version shows.
    """
    workspace = config.workspace or "."
    sql_paths = config.sql_paths or "**/*.sql"
    ignored = config.ignored_paths

    if config.git_error:
        raise GitError(config.git_error)
    base_ref = config.compare_base()

    # Changed files with status (A=added, D=deleted, M=modified, R=renamed, T=type change)
    changed_files = _git_diff_files(workspace, base_ref)
    base_moved = config.base_moved_paths()

    def analyzed(path: str) -> bool:
        """A `.sql` file `sql_paths` covers and `ignored_paths` doesn't exempt."""
        return (
            _matches_glob(path, sql_paths)
            and not any(_matches_glob(path, pattern) for pattern in ignored)
            and path.endswith(".sql")
        )

    results = []
    for status, old_path, new_path in changed_files:
        path = new_path or old_path
        if not analyzed(path):
            continue
        moved_from = old_path if status == "R" else None
        if moved_from is not None and not analyzed(moved_from):
            # Moved in from a file DiffCost doesn't analyze (an ignored path, outside
            # `sql_paths`, not `.sql`): that file was never priced, so this query is new
            status = "A"
        pr_path = Path(workspace) / path
        if status == "M" and pr_path.exists() and not pr_path.is_file() and not pr_path.is_symlink():
            continue  # a submodule whose name ends in .sql, moved to another commit: no SQL on either side

        # Read both versions
        production_sql = None
        pr_sql = None
        metadata = {"git_status": status}

        if status != "A":  # Not a new file — read production version
            source_path = old_path if status == "R" else path
            production_sql = _git_show(workspace, base_ref, source_path)
            if production_sql is None and status == "T" and _is_submodule(workspace, base_ref, path):
                # git can't show a gitlink: the path held a submodule, not SQL
                metadata["unknown_reason"] = "the file replaces a submodule, so there is no base version to compare"
            elif production_sql is None and status != "D":
                # git lists the file as changed from that commit, so its earlier version
                # exists there: reading it as new would drop it from a relative threshold
                metadata["unknown_reason"] = "the base version couldn't be read from git"

        if status != "D":  # Not deleted — read PR version from working tree
            if pr_path.is_symlink():
                # Never follow a PR-controlled symlink (it could point anywhere on the runner)
                metadata["unknown_reason"] = "the file is a symlink in this PR"
            elif pr_path.is_file():
                pr_sql = pr_path.read_text(encoding="utf-8", errors="replace")
            elif not pr_path.exists():
                # Changed in git but not in the working tree (a sparse checkout, a step that
                # deleted it): reading it as deleted would hide the change
                metadata["unknown_reason"] = "the file isn't in the checkout"
            # Otherwise it isn't a file (a submodule whose name ends in .sql): no SQL to read

        if status != "D" and (path in base_moved or moved_from in base_moved):
            add_unknown_reason(metadata, BASE_MOVED_REASON, first=True)

        # Skip if we couldn't read either version
        if production_sql is None and pr_sql is None and "unknown_reason" not in metadata:
            continue

        name = Path(path).stem
        results.append(
            ChangedSQL(
                name=name,
                path=path,
                production_sql=production_sql,
                pr_sql=pr_sql,
                adapter="sql_files",
                metadata=metadata,
            )
        )

    return results


def git_changes(workspace: str, base_ref: str) -> tuple[dict[str, str], dict[str, str]]:
    """(each changed path, the new one for a rename -> its git status letter,
    each renamed file's new path -> its old path)."""
    files = _git_diff_files(workspace, base_ref)
    return (
        {(new or old): status for status, old, new in files},
        {new: old for status, old, new in files if status == "R" and new},
    )


def _git_diff_files(workspace: str, base_ref: str) -> list[tuple[str, str, str | None]]:
    """
    Run git diff to find changed files.
    Returns list of (status, old_path, new_path). new_path is only set for renames.
    Raises GitError if git fails (missing base history, ownership, not a repo).
    """
    result = run_git(
        workspace,
        # -M: renames are paired whatever the repository's `diff.renames` setting says
        ["diff", "--name-status", "-z", "-M", "--diff-filter=ADMRT", f"{base_ref}...HEAD"],
        timeout=30,
    )
    if result.returncode != 0:
        raise GitError(f"Could not diff against `{base_ref}`: {explain_git_failure(result.stderr)}")

    # -z output: STATUS\0PATH\0 (or STATUS\0OLD\0NEW\0 for renames). Paths are not quoted.
    tokens = result.stdout.split("\0")
    # run_git replaces bytes that aren't UTF-8, so such a path can't be read back: never
    # skip the file silently (that would report "no changes")
    bad = next((t for t in tokens if "\ufffd" in t), None)
    if bad is not None:
        raise GitError(
            f"A changed file's name isn't valid UTF-8 (`{bad}`), so DiffCost can't read it. "
            "Rename the file to a UTF-8 name."
        )
    files: list[tuple[str, str, str | None]] = []
    i = 0
    while i < len(tokens):
        status_token = tokens[i].strip()
        if not status_token:
            i += 1
            continue
        status = status_token[0]
        if status == "R":
            if i + 2 >= len(tokens):
                break
            files.append((status, tokens[i + 1], tokens[i + 2]))
            i += 3
        else:
            if i + 1 >= len(tokens):
                break
            files.append((status, tokens[i + 1], None))
            i += 2
    return files


def _is_submodule(workspace: str, ref: str, path: str) -> bool:
    """Whether `path` is a submodule (a gitlink, mode 160000) at `ref`."""
    try:
        result = run_git(workspace, ["ls-tree", "-z", ref, "--", path], timeout=10)
    except GitError:
        return False
    return result.returncode == 0 and result.stdout.startswith("160000 ")


def _git_show(workspace: str, ref: str, path: str) -> str | None:
    """Read a file from a specific git ref. Returns None if the file doesn't exist at that ref."""
    try:
        result = run_git(workspace, ["show", f"{ref}:{path}"], timeout=10)
    except GitError:
        return None
    if result.returncode == 0:
        return result.stdout
    return None


# More `**/` than this in one glob are matched as `*/` (one or more directories), which
# keeps the number of fnmatch variants bounded
_MAX_ZERO_DIR_WILDCARDS = 6


def _matches_glob(path: str, pattern: str) -> bool:
    """Whether `path` matches a glob. Several globs can be separated by commas (a literal
    comma is written `[,]`).

    fnmatch semantics (`*` also matches `/`, as the docs say), plus `**/`, which matches
    zero or more directories. Every directory named in the glob must match:
    `**/legacy/**` matches `legacy/a.sql` and `x/legacy/a.sql`, never `models/a.sql`.
    """
    for p in (p.strip() for p in _split_globs(pattern)):
        if not p:
            continue
        # `**/` is either "one or more directories" (`*/`, since `*` crosses `/`) or
        # "no directory" (removed): try every combination
        parts = p.split("**/")
        joins = len(parts) - 1
        free = min(joins, _MAX_ZERO_DIR_WILDCARDS)
        for choice in itertools.product(("*/", ""), repeat=free):
            seps = list(choice) + ["*/"] * (joins - free)
            variant = parts[0] + "".join(sep + part for sep, part in zip(seps, parts[1:]))
            if fnmatch.fnmatch(path, variant):
                return True
    return False


def _split_globs(pattern: str) -> list[str]:
    """Split on the commas outside `[...]` character classes."""
    parts, start, i = [], 0, 0
    while i < len(pattern):
        if pattern[i] == "[":
            # fnmatch: `]` right after `[` or `[!` is part of the class; no closing `]`: literal `[`
            j = i + 1
            if j < len(pattern) and pattern[j] == "!":
                j += 1
            if j < len(pattern) and pattern[j] == "]":
                j += 1
            close = pattern.find("]", j)
            if close != -1:
                i = close + 1
                continue
        elif pattern[i] == ",":
            parts.append(pattern[start:i])
            start = i + 1
        i += 1
    parts.append(pattern[start:])
    return parts


def exempt_entry(path: str) -> str:
    """An `ignored_paths` entry that matches exactly `path` (a file, or a directory followed
    by `/**`): glob characters escaped, a comma (which separates globs) written `[,]`, and
    whitespace at either end (which the matcher strips) written in brackets (`[ ]`)."""
    entry = glob.escape(path).replace(",", "[,]")
    core = entry.strip()
    if not core:
        return "".join(f"[{ch}]" for ch in entry)
    lead = entry[: len(entry) - len(entry.lstrip())]
    trail = entry[len(entry.rstrip()):]
    return "".join(f"[{ch}]" for ch in lead) + core + "".join(f"[{ch}]" for ch in trail)
