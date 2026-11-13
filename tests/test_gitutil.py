"""Tests for base-ref resolution against a real checkout layout."""

import pytest

from gitutil import (
    GitError,
    check_checkout_matches_pr,
    explain_git_failure,
    is_committed_copy,
    is_tracked,
    resolve_base_ref,
)


def _sha(repo, rev):
    return repo.git("rev-parse", rev).strip()


class TestResolveBaseRef:
    def test_bare_main_resolves_to_origin_main_in_pr_checkout(self, checkout_repo):
        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        # A bare `main` does not exist locally after actions/checkout
        assert resolve_base_ref(str(repo.path), "main") == _sha(repo, "origin/main")

    def test_non_default_base_branch(self, checkout_repo):
        repo = checkout_repo({"queries/a.sql": "SELECT 1"}, base_branch="develop")
        assert resolve_base_ref(str(repo.path), "develop") == _sha(repo, "origin/develop")

    def test_explicit_origin_ref_kept(self, checkout_repo):
        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        assert resolve_base_ref(str(repo.path), "origin/main") == _sha(repo, "refs/remotes/origin/main")

    def test_tag_named_origin_main_does_not_shadow_the_branch(self, checkout_repo):
        """`git rev-parse origin/main` prefers refs/tags/origin/main; the base must be the branch."""
        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        repo.write("queries/a.sql", "SELECT * FROM big")
        repo.commit("pr")
        repo.git("tag", "origin/main", "HEAD")  # pushed by someone with write access
        assert resolve_base_ref(str(repo.path), "main") == _sha(repo, "refs/remotes/origin/main")
        assert resolve_base_ref(str(repo.path), "main") != _sha(repo, "HEAD")

    def test_event_base_sha_wins_and_must_exist(self, checkout_repo):
        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        base = _sha(repo, "origin/main")
        assert resolve_base_ref(str(repo.path), "ignored", base_sha=base) == base
        with pytest.raises(GitError) as exc:
            resolve_base_ref(str(repo.path), "main", base_sha="f" * 40)
        assert "fetch-depth: 0" in str(exc.value)

    def test_missing_branch_raises_actionable_error(self, checkout_repo):
        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        with pytest.raises(GitError) as exc:
            resolve_base_ref(str(repo.path), "release")
        assert "fetch-depth: 0" in str(exc.value)

    def test_explicit_tag_ref_is_rejected(self, checkout_repo):
        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        repo.git("tag", "v1", "HEAD")
        with pytest.raises(GitError) as exc:
            resolve_base_ref(str(repo.path), "refs/tags/v1")
        assert "Tags" in str(exc.value)
        assert resolve_base_ref(str(repo.path), "refs/remotes/origin/main") == _sha(repo, "origin/main")

    def test_full_sha_accepted_abbreviated_is_a_branch_name(self, checkout_repo):
        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        sha = _sha(repo, "origin/main")
        assert resolve_base_ref(str(repo.path), sha) == sha
        with pytest.raises(GitError):
            resolve_base_ref(str(repo.path), sha[:12])

    def test_not_a_repository(self, tmp_path):
        with pytest.raises(GitError) as exc:
            resolve_base_ref(str(tmp_path), "main")
        assert "not a git" in str(exc.value).lower()


class TestExplainGitFailure:
    def test_dubious_ownership_hint(self):
        msg = explain_git_failure("fatal: detected dubious ownership in repository at '/github/workspace'")
        assert "safe.directory" in msg

    def test_shallow_clone_hint(self):
        assert "fetch-depth: 0" in explain_git_failure("fatal: origin/main...HEAD: no merge base")


class TestIsTracked:
    def test_committed_file_is_tracked(self, checkout_repo):
        repo = checkout_repo({"target/prod-manifest.json": "{}"})
        assert is_tracked(str(repo.path), "target/prod-manifest.json")

    def test_untracked_file(self, checkout_repo):
        repo = checkout_repo({"a.sql": "SELECT 1"})
        repo.write(".diffcost-baseline/manifest.json", "{}")
        assert not is_tracked(str(repo.path), ".diffcost-baseline/manifest.json")

    def test_git_failure_is_not_untracked(self, monkeypatch):
        """A committed baseline must never pass because git timed out or failed."""
        import subprocess

        import gitutil

        def timeout(workspace, args, timeout=30):
            raise GitError("`git ls-files` timed out after 10s")

        monkeypatch.setattr(gitutil, "run_git", timeout)
        with pytest.raises(GitError):
            gitutil.is_tracked(".", ".diffcost-baseline/manifest.json")

        def broken(workspace, args, timeout=30):
            return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: not a git repository")

        monkeypatch.setattr(gitutil, "run_git", broken)
        with pytest.raises(GitError):
            gitutil.is_tracked(".", ".diffcost-baseline/manifest.json")


class TestRunGit:
    def test_non_utf8_output_is_replaced_not_raised(self, checkout_repo):
        """A Latin-1 comment in a base-branch .sql file must not crash the run."""
        from gitutil import run_git

        repo = checkout_repo({"a.sql": "SELECT 1"})
        (repo.path / "a.sql").write_bytes(b"-- caf\xe9\nSELECT 1\n")
        repo.commit("latin-1")
        out = run_git(str(repo.path), ["show", "HEAD:a.sql"]).stdout
        assert out == "-- caf\ufffd\nSELECT 1\n"


class TestCheckoutMatchesPr:
    def test_head_is_pr_head(self, checkout_repo):
        repo = checkout_repo({"a.sql": "SELECT 1"})
        repo.write("a.sql", "SELECT 2")
        repo.commit("pr")
        assert check_checkout_matches_pr(str(repo.path), _sha(repo, "HEAD")) == ""

    def test_merge_commit_returns_its_base_parent(self, checkout_repo):
        """actions/checkout on pull_request checks out GitHub's merge commit (base, head)."""
        repo = checkout_repo({"a.sql": "SELECT 1"})
        base = _sha(repo, "HEAD")
        repo.git("checkout", "-q", "-b", "feature")
        repo.write("a.sql", "SELECT 2")
        repo.commit("pr")
        head = _sha(repo, "HEAD")
        repo.git("checkout", "-q", "--detach", base)
        repo.git("-c", "user.name=t", "-c", "user.email=t@e", "merge", "-q", "--no-ff", "-m", "merge", "feature")
        assert check_checkout_matches_pr(str(repo.path), head) == base

    def test_base_checkout_on_pull_request_target_is_rejected(self, checkout_repo):
        """pull_request_target with the default checkout analyzes the base, not the PR: never 'clean'."""
        repo = checkout_repo({"a.sql": "SELECT 1"})
        with pytest.raises(GitError) as exc:
            check_checkout_matches_pr(str(repo.path), "a" * 40)
        assert "pull_request_target" in str(exc.value)


class TestIsCommittedCopy:
    def test_unchanged_and_regenerated(self, checkout_repo):
        repo = checkout_repo({"target/manifest.json": "{}"})
        assert is_committed_copy(str(repo.path), "target/manifest.json")
        repo.write("target/manifest.json", '{"metadata": {}}')
        assert not is_committed_copy(str(repo.path), "target/manifest.json")
        assert not is_committed_copy(str(repo.path), "missing.json")

    def test_raw_bytes_match_wins_over_filtered_diff(self, monkeypatch):
        """`.gitattributes` can make git's filtered diff call an untouched committed file
        modified; matching raw bytes still mark it as the committed copy."""
        import subprocess

        import gitutil

        def fake(workspace, args, timeout=30):
            out = {"rev-parse": "abc123\n", "hash-object": "abc123\n"}.get(args[0], "")
            code = 1 if args[0] == "diff" else 0  # filtered diff says "modified"
            return subprocess.CompletedProcess(args, code, stdout=out, stderr="")

        monkeypatch.setattr(gitutil, "run_git", fake)
        assert gitutil.is_committed_copy(".", "target/manifest.json")

        def smudged(workspace, args, timeout=30):
            out = {"rev-parse": "abc123\n", "hash-object": "def456\n"}.get(args[0], "")
            return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")  # filtered diff: unchanged

        monkeypatch.setattr(gitutil, "run_git", smudged)
        assert gitutil.is_committed_copy(".", "target/manifest.json")  # e.g. an eol=crlf checkout filter

    def test_git_failure_is_not_regenerated(self, monkeypatch):
        """A committed PR manifest must never pass because git timed out or failed."""
        import subprocess

        import gitutil

        def diff_times_out(workspace, args, timeout=30):
            if args[0] == "diff":
                raise GitError("`git diff` timed out after 30s")
            out = {"rev-parse": "abc123\n", "hash-object": "def456\n"}.get(args[0], "")
            return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

        monkeypatch.setattr(gitutil, "run_git", diff_times_out)
        with pytest.raises(GitError):
            gitutil.is_committed_copy(".", "target/manifest.json")

        def diff_fails(workspace, args, timeout=30):
            out = {"rev-parse": "abc123\n", "hash-object": "def456\n"}.get(args[0], "")
            code = 128 if args[0] == "diff" else 0
            return subprocess.CompletedProcess(args, code, stdout=out, stderr="fatal: bad object")

        monkeypatch.setattr(gitutil, "run_git", diff_fails)
        with pytest.raises(GitError):
            gitutil.is_committed_copy(".", "target/manifest.json")


class TestMergeQueueCheckout:
    """Merge queue runs analyze the queue commit."""

    def test_head_must_be_the_queue_commit(self, checkout_repo):
        from gitutil import check_checkout_is_merge_group

        repo = checkout_repo({"a.sql": "SELECT 1"})
        check_checkout_is_merge_group(str(repo.path), _sha(repo, "HEAD"))  # no error
        with pytest.raises(GitError, match="not the merge queue commit"):
            check_checkout_is_merge_group(str(repo.path), "b" * 40)

    def test_missing_queue_base_names_the_merge_queue(self, checkout_repo):
        repo = checkout_repo({"a.sql": "SELECT 1"})
        with pytest.raises(GitError, match="merge queue's base commit `ffffffffffff` is not in the checkout"):
            resolve_base_ref(str(repo.path), "main", base_sha="f" * 40, merge_group=True)
