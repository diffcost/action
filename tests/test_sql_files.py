"""Tests for the SQL files adapter."""

from unittest.mock import patch, MagicMock

import pytest

from adapters.sql_files import BASE_MOVED_REASON, find_changed_sql, _git_diff_files, _git_show, _matches_glob
from config import ActionConfig, ThresholdConfig
from gitutil import GitError


def _make_config(**overrides) -> ActionConfig:
    defaults = dict(
        sql_paths="**/*.sql",
        manifest_path="",
        gcp_credentials="",
        gcp_project="",
        api_key="",
        config_path=".diffcost.yml",
        base_ref="main",
        cost_per_tb=6.25,
        comment_on_clean=False,
        github_token="",
        github_repository="",
        pr_number=1,
        github_event_name="pull_request",
        workspace="/tmp/test-repo",
        thresholds=ThresholdConfig(),
        ignored_paths=[],
        slack_webhook="",
    )
    defaults.update(overrides)
    return ActionConfig(**defaults)


class TestMatchesGlob:
    def test_simple_pattern(self):
        assert _matches_glob("queries/report.sql", "**/*.sql")
        assert _matches_glob("report.sql", "**/*.sql")
        assert not _matches_glob("report.py", "**/*.sql")

    def test_specific_path(self):
        assert _matches_glob("queries/reports/daily.sql", "queries/**/*.sql")
        assert not _matches_glob("models/daily.sql", "queries/**/*.sql")

    def test_double_star_matches_zero_or_more_directories(self):
        assert _matches_glob("queries/daily.sql", "queries/**/*.sql")
        assert _matches_glob("legacy/a.sql", "**/legacy/**")
        assert _matches_glob("x/y/legacy/a.sql", "**/legacy/**")
        assert _matches_glob("dbt/legacy/a.sql", "dbt/**/legacy/*.sql")
        assert _matches_glob("dbt/a/b/legacy/a.sql", "dbt/**/legacy/*.sql")
        assert _matches_glob("a/b/c/d.sql", "a/**/b/**/c/**/d.sql")

    def test_directories_in_the_glob_are_enforced(self):
        # `ignored_paths: ["**/legacy/**"]` must not skip every SQL file in the repository
        assert not _matches_glob("models/orders.sql", "**/legacy/**")
        assert not _matches_glob("models/orders.sql", "**/archive/*.sql")
        assert not _matches_glob("dbt/models/x.sql", "dbt/**/legacy/*.sql")
        assert not _matches_glob("other/queries/a.sql", "queries/**/*.sql")
        assert not _matches_glob("orders.sql", "**/queries/*.sql")

    def test_star_also_matches_slash(self):
        # Documented on /docs/configuration: `queries/*.sql` matches nested files too
        assert _matches_glob("queries/a/b.sql", "queries/*.sql")
        assert _matches_glob("analytics/x/y.sql", "analytics/**")

    def test_comma_separated(self):
        assert _matches_glob("a.sql", "**/*.sql, **/*.py")
        assert _matches_glob("a.py", "**/*.sql, **/*.py")
        assert not _matches_glob("a.js", "**/*.sql, **/*.py")


class TestGitDiffFiles:
    @patch("adapters.sql_files.run_git")
    def test_parses_modified_files(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="M\0queries/report.sql\0A\0queries/new.sql\0D\0queries/old.sql\0",
            stderr="",
        )
        files = _git_diff_files("/workspace", "origin/main")
        assert files == [
            ("M", "queries/report.sql", None),
            ("A", "queries/new.sql", None),
            ("D", "queries/old.sql", None),
        ]
        assert "origin/main...HEAD" in mock_run.call_args.args[1]
        assert "-z" in mock_run.call_args.args[1]

    @patch("adapters.sql_files.run_git")
    def test_parses_rename(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="R100\0old/path.sql\0new/path.sql\0", stderr="")
        assert _git_diff_files("/workspace", "main") == [("R", "old/path.sql", "new/path.sql")]

    @patch("adapters.sql_files.run_git")
    def test_replaced_file_name_bytes_raise(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="M\0q/caf\ufffd.sql\0", stderr="")
        import pytest

        from gitutil import GitError

        with pytest.raises(GitError, match="isn't valid UTF-8"):
            _git_diff_files("/workspace", "main")

    def test_non_utf8_file_name_fails_instead_of_being_skipped(self, checkout_repo):
        """Its replaced name can't be read back: never "no changes" (fails closed)."""
        import os

        import pytest

        from gitutil import GitError

        repo = checkout_repo({"a.sql": "SELECT 1"})
        try:
            with open(os.path.join(os.fsencode(repo.path), b"caf\xe9.sql"), "w") as f:
                f.write("SELECT 2")
        except OSError:
            pytest.skip("this filesystem only stores UTF-8 file names")
        repo.commit("latin-1 name")
        with pytest.raises(GitError, match="isn't valid UTF-8"):
            _git_diff_files(str(repo.path), "origin/main")

    @patch("adapters.sql_files.run_git")
    def test_paths_with_spaces_and_tabs_are_not_quoted(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="M\0q/my report\t1.sql\0", stderr="")
        assert _git_diff_files("/workspace", "main") == [("M", "q/my report\t1.sql", None)]

    @patch("adapters.sql_files.run_git")
    def test_git_failure_raises_instead_of_reporting_no_changes(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="fatal: bad revision 'main...HEAD'")
        with pytest.raises(GitError) as exc:
            _git_diff_files("/workspace", "main")
        assert "fetch-depth: 0" in str(exc.value)

    @patch("adapters.sql_files.run_git")
    def test_dubious_ownership_raises(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=128, stdout="", stderr="fatal: detected dubious ownership in repository at '/github/workspace'"
        )
        with pytest.raises(GitError):
            _git_diff_files("/workspace", "main")

    @patch("adapters.sql_files.run_git")
    def test_empty_diff(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert _git_diff_files("/workspace", "main") == []


class TestGitShow:
    @patch("adapters.sql_files.run_git")
    def test_reads_file(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="SELECT 1")
        assert _git_show("/workspace", "main", "query.sql") == "SELECT 1"

    @patch("adapters.sql_files.run_git")
    def test_file_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="", stderr="fatal: path not found")
        assert _git_show("/workspace", "main", "nonexistent.sql") is None


class TestRealCheckout:
    """End to end against a repo laid out like actions/checkout leaves it (no local `main`)."""

    def test_detects_modified_added_and_renamed_files(self, checkout_repo):
        repo = checkout_repo({
            "queries/report.sql": "SELECT id FROM t WHERE dt = '2026-01-01'",
            "queries/old_name.sql": "SELECT 1",
        })
        repo.write("queries/report.sql", "SELECT id FROM t")
        repo.write("queries/new.sql", "SELECT 2")
        repo.git("mv", "queries/old_name.sql", "queries/renamed.sql")
        repo.commit()

        config = _make_config(workspace=str(repo.path), base_ref="main", resolved_base_ref="origin/main")
        result = {r.path: r for r in find_changed_sql(config)}

        assert result["queries/report.sql"].production_sql == "SELECT id FROM t WHERE dt = '2026-01-01'"
        assert result["queries/report.sql"].pr_sql == "SELECT id FROM t"
        assert result["queries/new.sql"].production_sql is None
        assert result["queries/renamed.sql"].production_sql == "SELECT 1"

    def test_unresolved_base_ref_raises(self, checkout_repo):
        repo = checkout_repo({"queries/report.sql": "SELECT 1"})
        config = _make_config(workspace=str(repo.path), git_error="Base branch `main` was not found")
        with pytest.raises(GitError):
            find_changed_sql(config)


class TestFindChangedSQL:
    @pytest.fixture(autouse=True)
    def _no_git(self, monkeypatch):
        """These tests stub git's answers; the commit to compare with is the base itself."""
        monkeypatch.setattr(ActionConfig, "compare_base", lambda self: self.base_rev)

    @patch("adapters.sql_files._git_show")
    @patch("adapters.sql_files._git_diff_files")
    def test_modified_file(self, mock_diff, mock_show, tmp_path):
        mock_diff.return_value = [("M", "queries/report.sql", None)]
        mock_show.return_value = "SELECT id FROM old_table"

        # Create PR version on disk
        sql_dir = tmp_path / "queries"
        sql_dir.mkdir()
        (sql_dir / "report.sql").write_text("SELECT id FROM new_table")

        config = _make_config(workspace=str(tmp_path))
        result = find_changed_sql(config)

        assert len(result) == 1
        assert result[0].name == "report"
        assert result[0].production_sql == "SELECT id FROM old_table"
        assert result[0].pr_sql == "SELECT id FROM new_table"

    @patch("adapters.sql_files._git_show")
    @patch("adapters.sql_files._git_diff_files")
    def test_new_file(self, mock_diff, mock_show, tmp_path):
        mock_diff.return_value = [("A", "queries/new.sql", None)]
        mock_show.return_value = None  # Shouldn't be called for new files

        sql_dir = tmp_path / "queries"
        sql_dir.mkdir()
        (sql_dir / "new.sql").write_text("SELECT * FROM table")

        config = _make_config(workspace=str(tmp_path))
        result = find_changed_sql(config)

        assert len(result) == 1
        assert result[0].production_sql is None
        assert result[0].pr_sql == "SELECT * FROM table"

    @patch("adapters.sql_files._git_show")
    @patch("adapters.sql_files._git_diff_files")
    def test_deleted_file(self, mock_diff, mock_show, tmp_path):
        mock_diff.return_value = [("D", "queries/old.sql", None)]
        mock_show.return_value = "SELECT * FROM old_table"

        config = _make_config(workspace=str(tmp_path))
        result = find_changed_sql(config)

        assert len(result) == 1
        assert result[0].production_sql == "SELECT * FROM old_table"
        assert result[0].pr_sql is None

    @patch("adapters.sql_files._git_show")
    @patch("adapters.sql_files._git_diff_files")
    def test_ignored_paths(self, mock_diff, mock_show, tmp_path):
        mock_diff.return_value = [
            ("M", "queries/report.sql", None),
            ("M", "tests/test_query.sql", None),
        ]
        mock_show.return_value = "SELECT 1"

        for path in ["queries", "tests"]:
            d = tmp_path / path
            d.mkdir()
            (d / (path.rstrip("s") + ".sql" if path != "queries" else "report.sql")).write_text("SELECT 2")

        (tmp_path / "tests" / "test_query.sql").write_text("SELECT 2")

        config = _make_config(workspace=str(tmp_path), ignored_paths=["tests/**/*.sql"])
        result = find_changed_sql(config)

        names = [r.name for r in result]
        assert "report" in names
        assert "test_query" not in names

    @patch("adapters.sql_files._git_show")
    @patch("adapters.sql_files._git_diff_files")
    def test_non_sql_files_filtered(self, mock_diff, mock_show, tmp_path):
        mock_diff.return_value = [
            ("M", "queries/report.sql", None),
            ("M", "README.md", None),
            ("M", "app.py", None),
        ]
        mock_show.return_value = "SELECT 1"
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "report.sql").write_text("SELECT 2")

        config = _make_config(workspace=str(tmp_path))
        result = find_changed_sql(config)

        assert len(result) == 1
        assert result[0].name == "report"


class TestSymlinksAndTypeChanges:
    def test_sql_file_turned_into_symlink_is_unknown_not_followed(self, checkout_repo, tmp_path):
        """A type change (T) was dropped by --diff-filter=ADMR; the symlink must not be followed."""
        secret = tmp_path / "secret.txt"
        secret.write_text("runner secret")
        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        (repo.path / "queries/a.sql").unlink()
        (repo.path / "queries/a.sql").symlink_to(secret)
        repo.commit()

        config = _make_config(workspace=str(repo.path), base_ref="main", resolved_base_ref="origin/main")
        [item] = find_changed_sql(config)

        assert item.metadata["git_status"] == "T"
        assert item.pr_sql is None
        assert "symlink" in item.metadata["unknown_reason"]


def _base_moves_on(checkout_repo, base_files, pr_files, base_changes):
    """A PR branched from the base, then the base gained `base_changes` (None deletes a file).

    Returns (repo, branch point, PR head, base tip); HEAD is the PR head (a head checkout,
    as `pull_request_target` workflows check out), and `origin/main` is the base tip.
    """
    from tests.conftest import _git

    repo = checkout_repo(base_files)
    branch_point = repo.git("rev-parse", "HEAD").strip()
    for path, content in pr_files.items():
        repo.write(path, content)
    repo.commit("pr")
    head = repo.git("rev-parse", "HEAD").strip()
    for path, content in base_changes.items():
        target = repo.upstream / path
        if content is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    _git(repo.upstream, "add", "-A")
    _git(repo.upstream, "commit", "-q", "-m", "base moves on")
    repo.git("fetch", "-q", "origin")
    return repo, branch_point, head, repo.git("rev-parse", "origin/main").strip()


class TestComparedWithWhereTheBranchStarted:
    """Files are compared with the commit where the PR's branch meets the base: commits the
    base gained since then are not the PR's changes."""

    def test_a_change_only_the_base_made_later_is_not_the_pr_s(self, checkout_repo, capsys):
        """The base added a filter to another file after the branch started: the PR's own file
        is priced from the branch point, and the other file isn't listed."""
        repo, branch_point, _, tip = _base_moves_on(
            checkout_repo,
            {"queries/revenue.sql": "SELECT * FROM t\n-- owner: data\n", "queries/other.sql": "SELECT * FROM u\n"},
            {"queries/revenue.sql": "SELECT * FROM t\n-- owner: finance\n"},
            {"queries/other.sql": "SELECT * FROM u WHERE event_date >= '2026-01-01'\n"},
        )
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=tip))
        assert item.path == "queries/revenue.sql"
        assert item.production_sql == "SELECT * FROM t\n-- owner: data\n"
        assert item.pr_sql == "SELECT * FROM t\n-- owner: finance\n"
        assert "unknown_reason" not in item.metadata
        out = capsys.readouterr().out
        assert f"::notice::The changed files are compared with commit {branch_point[:12]}" in out

    def test_a_file_the_base_also_changed_later_is_unknown(self, checkout_repo):
        """After the merge the file holds both changes: neither version is "before"."""
        repo, branch_point, _, tip = _base_moves_on(
            checkout_repo,
            {"queries/revenue.sql": "SELECT * FROM t\n-- owner: data\n"},
            {"queries/revenue.sql": "SELECT * FROM t\n-- owner: finance\n"},
            {"queries/revenue.sql": "SELECT * FROM t WHERE event_date >= '2026-01-01'\n-- owner: data\n"},
        )
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=tip))
        assert item.production_sql == "SELECT * FROM t\n-- owner: data\n"
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON
        assert "update the branch" in BASE_MOVED_REASON

    def test_a_base_saving_copied_into_the_pr_can_t_hide_its_increase(self, checkout_repo):
        """The base gained a filter O after the branch started; the PR has O too (cherry-picked)
        plus a join X of its own. Priced from the branch point, O's saving would offset X."""
        before = "SELECT e.a\nFROM `p.d.events` e\n-- joins\n\n\n\n\nWHERE TRUE\n"
        saving = before.replace("WHERE TRUE", "WHERE e.dt = CURRENT_DATE()")
        both = saving.replace("-- joins", "CROSS JOIN `p.d.huge` h")
        repo, _, _, tip = _base_moves_on(checkout_repo, {"q.sql": before}, {"q.sql": both}, {"q.sql": saving})
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=tip))
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON

    def test_a_rename_whose_old_path_the_base_changed_is_unknown(self, checkout_repo):
        from tests.conftest import _git

        repo = checkout_repo({"queries/old.sql": "SELECT a FROM t\n-- 1\n-- 2\n-- 3\n"})
        repo.git("mv", "queries/old.sql", "queries/new.sql")
        repo.commit("pr")
        (repo.upstream / "queries/old.sql").write_text("SELECT a FROM t\n-- 1\n-- 2\n-- 3, base\n")
        _git(repo.upstream, "commit", "-q", "-am", "base moves on")
        repo.git("fetch", "-q", "origin")
        tip = repo.git("rev-parse", "origin/main").strip()
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=tip))
        assert (item.path, item.metadata["git_status"]) == ("queries/new.sql", "R")
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON

    def test_a_rename_from_an_ignored_path_the_base_changed_is_unknown(self, checkout_repo):
        """New (the ignored file was never priced), and the base changed the file it came from."""
        from tests.conftest import _git

        repo = checkout_repo({"sandbox/r.sql": "SELECT a FROM t\n-- 1\n-- 2\n-- 3\n"})
        (repo.path / "queries").mkdir()
        repo.git("mv", "sandbox/r.sql", "queries/r.sql")
        repo.commit("pr")
        (repo.upstream / "sandbox/r.sql").write_text("SELECT a FROM t\n-- 1\n-- 2\n-- 3, base\n")
        _git(repo.upstream, "commit", "-q", "-am", "base moves on")
        repo.git("fetch", "-q", "origin")
        tip = repo.git("rev-parse", "origin/main").strip()
        config = _make_config(workspace=str(repo.path), resolved_base_ref=tip, ignored_paths=["sandbox/**"])
        [item] = find_changed_sql(config)
        assert (item.production_sql, item.metadata["git_status"]) == (None, "A")
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON

    def test_a_failed_diff_of_the_base_fails_the_run(self, checkout_repo, monkeypatch):
        """Like every other git failure: never "no changes", never a silently empty set."""
        repo, _, _, tip = _base_moves_on(
            checkout_repo, {"q.sql": "SELECT 1", "o.sql": "SELECT 8"}, {"q.sql": "SELECT 2"}, {"o.sql": "SELECT 9"},
        )
        def broken(*_):
            raise GitError("Could not diff: boom")

        monkeypatch.setattr("config.paths_changed_between", broken)
        with pytest.raises(GitError, match="boom"):
            find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=tip))

    def test_a_query_deleted_by_the_pr_stays_removed(self, checkout_repo):
        """Removing a query can't add cost, whatever the base did to it."""
        repo, _, _, tip = _base_moves_on(checkout_repo, {"q.sql": "SELECT 1"}, {"notes.txt": "x"}, {"q.sql": "SELECT 3"})
        (repo.path / "q.sql").unlink()
        repo.commit("delete")
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=tip))
        assert (item.production_sql, item.pr_sql) == ("SELECT 1", None)
        assert "unknown_reason" not in item.metadata

    def test_a_file_the_base_renamed_or_deleted_later_is_still_modified(self, checkout_repo):
        repo, _, _, tip = _base_moves_on(
            checkout_repo,
            {"queries/old.sql": "SELECT a FROM t", "queries/gone.sql": "SELECT b FROM t"},
            {"queries/old.sql": "SELECT a, c FROM t", "queries/gone.sql": "SELECT b, c FROM t"},
            {"queries/old.sql": None, "queries/renamed.sql": "SELECT a FROM t", "queries/gone.sql": None},
        )
        items = {i.path: i for i in find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=tip))}
        assert set(items) == {"queries/old.sql", "queries/gone.sql"}
        assert items["queries/old.sql"].production_sql == "SELECT a FROM t"  # not "new"
        assert items["queries/gone.sql"].production_sql == "SELECT b FROM t"
        # The merge conflicts with what the base did to both files
        assert {i.metadata["unknown_reason"] for i in items.values()} == {BASE_MOVED_REASON}

    def test_a_file_both_made_dearer_is_unknown_not_a_saving(self, checkout_repo):
        """The base made the file dearer after the branch started; the PR made it dearer too.
        Comparing with the base's tip would read the PR's version as a saving."""
        repo, _, _, tip = _base_moves_on(
            checkout_repo,
            {"q.sql": "SELECT id FROM small WHERE dt = CURRENT_DATE()"},
            {"q.sql": "SELECT id FROM small"},
            {"q.sql": "SELECT id FROM huge WHERE dt = CURRENT_DATE()"},
        )
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=tip))
        assert item.production_sql == "SELECT id FROM small WHERE dt = CURRENT_DATE()"
        assert item.pr_sql == "SELECT id FROM small"
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON

    def test_an_explicit_base_ahead_of_the_branch(self, checkout_repo, tmp_path):
        """`base_ref: main` resolves to the branch's tip, which has moved on since the PR started."""
        import json
        import os

        from config import load_config

        repo, branch_point, head, tip = _base_moves_on(
            checkout_repo, {"q.sql": "SELECT 1"}, {"q.sql": "SELECT 2"}, {"q.sql": "SELECT 3"},
        )
        event = tmp_path / "event.json"
        event.write_text(json.dumps({"pull_request": {"number": 5, "base": {"sha": tip}, "head": {"sha": head}}}))
        env = {"INPUT_BASE_REF": "main", "INPUT_CONFIG_PATH": ".diffcost.yml", "GITHUB_WORKSPACE": str(repo.path),
               "GITHUB_EVENT_NAME": "pull_request", "GITHUB_EVENT_PATH": str(event), "GITHUB_REF": "refs/pull/5/merge"}
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.git_error == "" and config.resolved_base_ref == tip
        config.sql_paths = "**/*.sql"
        [item] = find_changed_sql(config)
        assert (item.production_sql, item.pr_sql) == ("SELECT 1", "SELECT 2")
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON  # the base changed q.sql too
        assert config.compare_base() == branch_point

    def test_github_merge_commit_is_compared_with_its_base_parent_as_before(self, checkout_repo, capsys, monkeypatch):
        """On GitHub's merge commit the base is HEAD^1, which is where HEAD meets it: unchanged,
        even for a file both sides changed (the merge commit holds both changes)."""
        repo, _, head, tip = _base_moves_on(
            checkout_repo,
            {"q.sql": "SELECT 1\n-- a\n\n\n\n-- b\n", "other.sql": "SELECT 9"},
            {"q.sql": "SELECT 2\n-- a\n\n\n\n-- b\n"},
            {"q.sql": "SELECT 1\n-- a\n\n\n\n-- b, from the base\n", "other.sql": "SELECT 10"},
        )
        repo.git("checkout", "-q", "--detach", tip)
        repo.git("-c", "user.name=t", "-c", "user.email=t@example.com", "merge", "-q", "--no-ff", "-m", "merge", head)
        monkeypatch.setattr("config.paths_changed_between", lambda *a: pytest.fail("diffed the base against itself"))
        config = _make_config(workspace=str(repo.path), resolved_base_ref=tip)
        [item] = find_changed_sql(config)
        assert (item.path, item.production_sql, item.pr_sql) == (
            "q.sql", "SELECT 1\n-- a\n\n\n\n-- b, from the base\n", "SELECT 2\n-- a\n\n\n\n-- b, from the base\n"
        )
        assert "unknown_reason" not in item.metadata
        assert config.compare_base() == tip
        assert "::notice::" not in capsys.readouterr().out

    def test_a_merge_queue_commit_is_compared_with_its_base(self, checkout_repo, capsys):
        repo = checkout_repo({"q.sql": "SELECT 1"})
        base = repo.git("rev-parse", "HEAD").strip()
        repo.write("q.sql", "SELECT 2")
        repo.commit("queue commit")
        config = _make_config(workspace=str(repo.path), resolved_base_ref=base, merge_group=True)
        [item] = find_changed_sql(config)
        assert (item.production_sql, item.pr_sql) == ("SELECT 1", "SELECT 2")
        assert config.compare_base() == base
        assert "::notice::" not in capsys.readouterr().out

    def test_unrelated_histories_fail_instead_of_reporting_no_changes(self, checkout_repo):
        repo = checkout_repo({"q.sql": "SELECT 1"})
        repo.git("checkout", "-q", "--orphan", "unrelated")
        repo.write("q.sql", "SELECT 2")
        repo.commit("unrelated root")
        with pytest.raises(GitError, match="shares no commit"):
            find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref="origin/main"))

    def test_shallow_clone_fails_with_the_fetch_depth_hint(self, checkout_repo, tmp_path):
        from tests.conftest import _git

        repo, _, _, _ = _base_moves_on(checkout_repo, {"q.sql": "SELECT 1"}, {"q.sql": "SELECT 2"}, {"q.sql": "SELECT 3"})
        head = repo.git("rev-parse", "HEAD").strip()
        shallow = tmp_path / "shallow"
        _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{repo.upstream}", str(shallow))
        _git(shallow, "fetch", "-q", "--depth", "1", f"file://{repo.path}", head)
        _git(shallow, "checkout", "-q", "--detach", head)
        config = _make_config(workspace=str(shallow), resolved_base_ref="origin/main")
        with pytest.raises(GitError, match="fetch-depth: 0"):
            find_changed_sql(config)


class TestFilesMissingFromTheWorkingTree:
    def test_a_changed_file_missing_from_the_checkout_is_unknown_not_removed(self, checkout_repo):
        """A sparse checkout (or a step that deleted it) must not turn an edit into a removal."""
        repo = checkout_repo({"reports/x.sql": "SELECT id FROM t WHERE dt = CURRENT_DATE()"})
        repo.write("reports/x.sql", "SELECT id FROM t")
        repo.write("reports/new.sql", "SELECT * FROM t")
        repo.commit()
        (repo.path / "reports/x.sql").unlink()
        (repo.path / "reports/new.sql").unlink()
        config = _make_config(workspace=str(repo.path), resolved_base_ref="origin/main")
        items = {i.path: i for i in find_changed_sql(config)}
        assert items["reports/x.sql"].pr_sql is None
        assert items["reports/x.sql"].metadata["unknown_reason"] == "the file isn't in the checkout"
        assert items["reports/new.sql"].metadata["unknown_reason"] == "the file isn't in the checkout"

    def test_a_submodule_named_like_a_sql_file_is_not_read(self, checkout_repo, tmp_path):
        import subprocess

        from tests.conftest import _git

        repo = checkout_repo({"a.sql": "SELECT 1"})
        sub = tmp_path / "sub"
        sub.mkdir()
        _git(sub, "init", "-q", "-b", "main")
        (sub / "f").write_text("x")
        _git(sub, "add", "-A")
        _git(sub, "commit", "-q", "-m", "s")
        subprocess.run(
            ["git", "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "vendor.sql"],
            cwd=repo.path, check=True, capture_output=True,
        )
        repo.commit("submodule")
        assert (repo.path / "vendor.sql").is_dir()
        assert find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref="origin/main")) == []

    def _submodule_on_the_base(self, checkout_repo, tmp_path):
        """A base commit with a submodule at `vendor.sql`; HEAD is a branch from it."""
        import subprocess

        from tests.conftest import _git

        sub = tmp_path / "sub"
        sub.mkdir()
        _git(sub, "init", "-q", "-b", "main")
        (sub / "f").write_text("x")
        _git(sub, "add", "-A")
        _git(sub, "commit", "-q", "-m", "s")
        repo = checkout_repo({"a.sql": "SELECT 1"})
        subprocess.run(
            ["git", "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "vendor.sql"],
            cwd=repo.path, check=True, capture_output=True,
        )
        repo.commit("submodule")
        base = repo.git("rev-parse", "HEAD").strip()
        return repo, sub, base

    def test_a_submodule_moved_to_another_commit_is_not_read(self, checkout_repo, tmp_path):
        from tests.conftest import _git

        repo, sub, base = self._submodule_on_the_base(checkout_repo, tmp_path)
        (sub / "f").write_text("y")
        _git(sub, "commit", "-q", "-am", "s2")
        _git(repo.path / "vendor.sql", "pull", "-q", str(sub), "main")
        repo.git("add", "vendor.sql")
        repo.commit("bump")
        assert repo.git("diff", "--name-status", base, "HEAD").split() == ["M", "vendor.sql"]
        assert find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=base)) == []

    def test_a_file_that_replaces_a_submodule_is_unknown(self, checkout_repo, tmp_path):
        import shutil

        repo, _, base = self._submodule_on_the_base(checkout_repo, tmp_path)
        repo.git("rm", "-q", "--cached", "vendor.sql")
        shutil.rmtree(repo.path / "vendor.sql")
        (repo.path / ".gitmodules").unlink()
        repo.write("vendor.sql", "SELECT * FROM `p.d.huge`")
        repo.commit("replace the submodule")
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=base))
        assert item.metadata["git_status"] == "T"
        assert item.metadata["unknown_reason"] == "the file replaces a submodule, so there is no base version to compare"


class TestRenamesFromFilesThatAreNotAnalyzed:
    """A file moved in from a path DiffCost skips was never priced: it is new, at full cost."""

    @pytest.mark.parametrize("old_path, config", [
        ("sandbox/report.sql", {"ignored_paths": ["sandbox/**"]}),  # an ignored path
        ("notes/report.txt", {}),  # not a .sql file
        ("scratch/report.sql", {"sql_paths": "queries/**/*.sql"}),  # outside sql_paths
    ])
    def test_a_rename_into_the_analyzed_files_is_new(self, checkout_repo, old_path, config):
        sql = "SELECT * FROM `p.d.huge`\n"
        repo = checkout_repo({old_path: sql})
        (repo.path / "queries").mkdir()
        repo.git("mv", old_path, "queries/report.sql")
        repo.commit()
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref="origin/main", **config))
        assert item.path == "queries/report.sql"
        assert (item.production_sql, item.pr_sql) == (None, sql)
        assert item.metadata["git_status"] == "A"

    def test_a_rename_between_analyzed_files_keeps_its_base_version(self, checkout_repo):
        repo = checkout_repo({"queries/old.sql": "SELECT a FROM t\n"})
        repo.git("mv", "queries/old.sql", "queries/new.sql")
        repo.commit()
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref="origin/main"))
        assert (item.production_sql, item.metadata["git_status"]) == ("SELECT a FROM t\n", "R")


class TestUnreadableBaseVersion:
    def test_a_changed_file_whose_base_version_can_t_be_read_is_unknown_not_new(self, checkout_repo, monkeypatch):
        """git listed it as changed from the base, so its earlier version exists: reading it as
        new would leave it out of a relative threshold."""
        repo = checkout_repo({"queries/a.sql": "SELECT a FROM t WHERE dt = CURRENT_DATE()"})
        repo.write("queries/a.sql", "SELECT a FROM t")
        repo.commit()
        monkeypatch.setattr("adapters.sql_files._git_show", lambda *a, **k: None)
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref="origin/main"))
        assert item.production_sql is None
        assert item.metadata["unknown_reason"] == "the base version couldn't be read from git"

    def test_a_deleted_file_whose_base_version_can_t_be_read_is_skipped(self, checkout_repo, monkeypatch):
        repo = checkout_repo({"queries/a.sql": "SELECT a FROM t"})
        (repo.path / "queries/a.sql").unlink()
        repo.commit()
        monkeypatch.setattr("adapters.sql_files._git_show", lambda *a, **k: None)
        assert find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref="origin/main")) == []

    def test_an_unreadable_symlink_turned_into_a_file_is_not_called_a_submodule(self, checkout_repo, monkeypatch):
        """A type change whose base can't be read is a submodule only when git says the base
        entry is a gitlink (here a symlink, as a partial clone that can't fetch it would see)."""
        import os

        repo = checkout_repo({"target.sql": "SELECT 1"})
        os.symlink("target.sql", repo.path / "link.sql")
        repo.commit("link")
        base = repo.git("rev-parse", "HEAD").strip()
        (repo.path / "link.sql").unlink()
        repo.write("link.sql", "SELECT * FROM `p.d.huge`")
        repo.commit("a file now")
        monkeypatch.setattr("adapters.sql_files._git_show", lambda *a, **k: None)
        [item] = find_changed_sql(_make_config(workspace=str(repo.path), resolved_base_ref=base))
        assert item.metadata["git_status"] == "T"
        assert item.metadata["unknown_reason"] == "the base version couldn't be read from git"
