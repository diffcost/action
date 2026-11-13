"""Tests for the SQL files adapter."""

from unittest.mock import patch, MagicMock

import pytest

from adapters.sql_files import find_changed_sql, _git_diff_files, _git_show, _matches_glob
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
