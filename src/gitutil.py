"""Git helpers shared by config loading and the SQL / dbt adapters.

Every git failure raises `GitError` with an actionable message. Callers must
surface it ("DiffCost could not run") instead of treating it as "no changes".
"""

from __future__ import annotations

import re
import subprocess


class GitError(RuntimeError):
    """A git command failed in a way that prevents analysis."""


def run_git(workspace: str, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run `git <args>` in `workspace`. Raises GitError on timeout or missing git.

    Output is decoded as UTF-8 with undecodable bytes replaced (a Latin-1 comment in a
    base-branch .sql file), as the PR side reads working-tree files.
    """
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workspace or None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise GitError(f"`git {args[0]}` timed out after {timeout}s") from e
    except FileNotFoundError as e:
        raise GitError("git is not installed in the DiffCost container") from e


def explain_git_failure(stderr: str) -> str:
    """Turn common git stderr output into an actionable hint."""
    err = (stderr or "").strip()
    lower = err.lower()
    if "dubious ownership" in lower:
        hint = "the workspace is owned by a different user (git safe.directory)."
    elif "not a git repository" in lower:
        hint = "the workspace is not a git checkout. Add `actions/checkout` before DiffCost."
    elif "no merge base" in lower or "unknown revision" in lower or "bad revision" in lower:
        hint = "the base branch history is missing. Set `fetch-depth: 0` on `actions/checkout`."
    else:
        hint = ""
    first_line = err.splitlines()[0][:200] if err else "unknown git error"
    return f"{first_line}" + (f" — {hint}" if hint else "")


def _rev_parse(workspace: str, rev: str) -> str:
    """Resolve `rev` to a commit SHA, or "" if it does not name a commit."""
    result = run_git(workspace, ["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], timeout=10)
    return result.stdout.strip() if result.returncode == 0 else ""


_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
# Full refs accepted as an explicit `base_ref` (never tags or other refs)
_BRANCH_REF_PREFIXES = ("refs/heads/", "refs/remotes/")


def resolve_base_ref(workspace: str, base_ref: str, base_sha: str = "", merge_group: bool = False) -> str:
    """Resolve the diff base to a commit SHA.

    `base_sha` (the PR event's `pull_request.base.sha`, or a merge queue event's
    `merge_group.base_sha`; `merge_group` words the error for it) wins when given: it is
    content-addressed, so neither a fork used as `origin` nor a tag named
    `origin/<branch>` can substitute another commit. Otherwise `base_ref` is a
    branch name, looked up only as `refs/remotes/origin/<name>` and then
    `refs/heads/<name>` (never as a tag); a full `refs/heads/…` or
    `refs/remotes/…` ref; or a full 40-character commit SHA. `actions/checkout`
    on a pull_request event stores branches only as `origin/<name>`, so a bare
    `main` does not resolve locally.
    """
    probe = run_git(workspace, ["rev-parse", "--git-dir"], timeout=10)
    if probe.returncode != 0:
        raise GitError(f"Could not read the git repository: {explain_git_failure(probe.stderr)}")

    if base_sha:
        resolved = _rev_parse(workspace, base_sha)
        if resolved:
            return resolved
        whose = "merge queue's" if merge_group else "pull request's"
        raise GitError(
            f"The {whose} base commit `{base_sha[:12]}` is not in the checkout. "
            "Set `fetch-depth: 0` on `actions/checkout`."
        )

    if base_ref.startswith("refs/"):
        if not base_ref.startswith(_BRANCH_REF_PREFIXES):
            raise GitError(
                f"`base_ref` must be a branch name, a `refs/heads/` or `refs/remotes/` ref, or a full commit SHA "
                f"(got `{base_ref[:80]}`). Tags and other refs are not used as the base."
            )
        candidates = [base_ref]
    elif _FULL_SHA_RE.match(base_ref):
        candidates = [base_ref]
    else:
        name = base_ref[len("origin/"):] if base_ref.startswith("origin/") else base_ref
        candidates = [f"refs/remotes/origin/{name}", f"refs/heads/{name}"]

    for candidate in candidates:
        resolved = _rev_parse(workspace, candidate)
        if resolved:
            return resolved

    raise GitError(
        f"Base branch `{base_ref}` was not found (tried {', '.join(f'`{c}`' for c in candidates)}). "
        "Set `fetch-depth: 0` on `actions/checkout`, or set the `base_ref` input."
    )


def head_commit_info(workspace: str) -> tuple[str, list[str]]:
    """(HEAD sha, HEAD's parent shas). Parents are read from the raw commit object,
    so they are listed even in a shallow clone where the parent commits are missing."""
    head = _rev_parse(workspace, "HEAD")
    if not head:
        return "", []
    result = run_git(workspace, ["cat-file", "-p", head], timeout=10)
    parents = [
        line.split()[1] for line in result.stdout.splitlines()
        if line.startswith("parent ") and len(line.split()) == 2
    ] if result.returncode == 0 else []
    return head, parents


def check_checkout_matches_pr(workspace: str, head_sha: str) -> str:
    """Require that HEAD is the PR head or GitHub's merge commit of it.

    Returns the merge commit's first parent (the base commit GitHub merged into)
    when HEAD is that merge commit, else "". Raises GitError otherwise: e.g. a
    `pull_request_target` workflow that checked out the base branch would diff
    the base against itself and report a clean PR.
    """
    head, parents = head_commit_info(workspace)
    want = head_sha.lower()
    if head.lower() == want:
        return ""
    if len(parents) == 2 and parents[1].lower() == want:
        return parents[0]
    raise GitError(
        f"The checked-out commit does not contain this pull request's head commit `{head_sha[:12]}`. "
        "DiffCost must analyze the PR's code: with `pull_request_target`, check out "
        "`${{ github.event.pull_request.head.sha }}`."
    )


def check_checkout_is_merge_group(workspace: str, head_sha: str) -> None:
    """Require that HEAD is the merge queue commit (`merge_group.head_sha`), which
    `actions/checkout` checks out by default on a `merge_group` event. Raises GitError
    otherwise, so another checkout is never analyzed in its place."""
    head = _rev_parse(workspace, "HEAD")
    if head.lower() != head_sha.lower():
        raise GitError(
            f"The checked-out commit is not the merge queue commit `{head_sha[:12]}`. "
            "DiffCost must analyze the queue commit: check out `${{ github.event.merge_group.head_sha }}` "
            "(the default of `actions/checkout` on a `merge_group` event)."
        )


def is_tracked(workspace: str, rel_path: str) -> bool:
    """Whether `rel_path` is committed in the current checkout (PR-controlled).

    Raises GitError when git can't tell (a timeout, a broken repository): callers use
    this to refuse PR-controlled files, so "can't tell" must never read as "not committed".
    """
    result = run_git(workspace, ["ls-files", "--error-unmatch", "--", rel_path], timeout=10)
    if result.returncode not in (0, 1):
        raise GitError(f"`git ls-files` failed: {explain_git_failure(result.stderr)}")
    return result.returncode == 0


def is_committed_copy(workspace: str, rel_path: str) -> bool:
    """Whether `rel_path` is committed in HEAD and unchanged in the working tree.

    A file generated during the run (such as a freshly compiled dbt manifest) differs
    from any committed copy; an unchanged one is exactly what the PR author committed.
    Two comparisons, either of which counts as "unchanged": the raw bytes against the
    committed blob (`.gitattributes` can make git's filtered diff report a committed
    file as modified), and git's filtered diff (a checkout filter such as `eol=crlf`
    or `ident` changes the bytes written to the working tree).

    Raises GitError when git can't tell (a timeout, a failed diff): "can't tell" must
    never read as "generated in this run".
    """
    blob = run_git(workspace, ["rev-parse", "--verify", "--quiet", f"HEAD:{rel_path}"], timeout=10)
    if blob.returncode not in (0, 1):
        raise GitError(f"`git rev-parse` failed: {explain_git_failure(blob.stderr)}")
    if blob.returncode != 0 or not blob.stdout.strip():
        return False  # not in HEAD (`--quiet` exits 1 without output when the path is absent)
    raw = run_git(workspace, ["hash-object", "--no-filters", "--", rel_path], timeout=30)
    if raw.returncode == 0 and raw.stdout.strip() == blob.stdout.strip():
        return True
    result = run_git(workspace, ["diff", "--quiet", "HEAD", "--", rel_path], timeout=30)
    if result.returncode not in (0, 1):
        raise GitError(f"`git diff` failed: {explain_git_failure(result.stderr)}")
    return result.returncode == 0


# Directories dbt installs packages into: what is under them is never a project of the repository
INSTALLED_PACKAGE_DIRS = frozenset({"dbt_packages", "dbt_modules"})


def base_dbt_project_dirs(workspace: str, base_rev: str) -> list[str] | None:
    """Directories ("" = repo root) holding a dbt_project.yml on the base commit.

    Installed packages (`dbt_packages/`, `dbt_modules/`) are not projects. Shallowest
    first. None when the base commit can't be listed.
    """
    try:
        result = run_git(workspace, ["ls-tree", "-r", "-z", "--name-only", base_rev], timeout=30)
    except GitError:
        return None
    if result.returncode != 0:
        return None
    dirs = []
    for path in result.stdout.split("\0"):
        parts = path.split("/")
        if parts[-1] != "dbt_project.yml" or INSTALLED_PACKAGE_DIRS & set(parts[:-1]):
            continue
        dirs.append("/".join(parts[:-1]))
    return sorted(dirs, key=lambda d: (d.count("/") + bool(d), d))
