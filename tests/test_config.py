"""Tests for configuration loading."""

import os
from unittest.mock import patch

import pytest

from config import load_config, _resolve_env_var, _get_pr_number


class TestResolveEnvVar:
    def test_resolves_allowed_env_var(self):
        """Only allowlisted env vars are resolved (prevents secret exfiltration)."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK": "https://hooks.slack.com/xxx"}):
            result = _resolve_env_var("${{ SLACK_WEBHOOK }}")
            assert result == "https://hooks.slack.com/xxx"

    def test_resolves_with_spaces(self):
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "abc123"}):
            result = _resolve_env_var("${{  SLACK_WEBHOOK_URL  }}")
            assert result == "abc123"

    def test_disallowed_env_var_returns_empty(self):
        """Non-allowlisted vars (like GITHUB_TOKEN) are NOT resolved."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_secret"}):
            result = _resolve_env_var("${{ GITHUB_TOKEN }}")
            assert result == ""

    def test_missing_env_var(self):
        result = _resolve_env_var("${{ NONEXISTENT_VAR }}")
        assert result == ""

    def test_no_pattern(self):
        result = _resolve_env_var("https://plain-url.com")
        assert result == "https://plain-url.com"


class TestGetPrNumber:
    def test_from_github_ref(self):
        with patch.dict(os.environ, {"GITHUB_REF": "refs/pull/42/merge"}, clear=False):
            assert _get_pr_number() == 42

    def test_invalid_ref(self):
        with patch.dict(os.environ, {"GITHUB_REF": "refs/heads/main"}, clear=False):
            # Should return 0 (not a PR ref) if no event path
            with patch.dict(os.environ, {"GITHUB_EVENT_PATH": ""}, clear=False):
                assert _get_pr_number() == 0


class TestLoadConfig:
    def test_loads_defaults(self):
        env = {
            "INPUT_SQL_PATHS": "",
            "INPUT_MANIFEST_PATH": "",
            "INPUT_GCP_CREDENTIALS": "",
            "INPUT_GCP_PROJECT": "",
            "INPUT_API_KEY": "",
            "INPUT_CONFIG_PATH": ".diffcost.yml",
            "INPUT_BASE_REF": "main",
            "INPUT_COST_PER_TB": "6.25",
            "INPUT_COMMENT_ON_CLEAN": "false",
            "GITHUB_TOKEN": "ghp_test",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_WORKSPACE": "/tmp",
            "GITHUB_REF": "refs/pull/5/merge",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.base_ref == "main"
            assert config.cost_per_tb == 6.25
            assert config.comment_on_clean is False
            assert config.github_repository == "owner/repo"
            assert config.pr_number == 5

    def test_custom_cost_per_tb(self):
        env = {
            "INPUT_COST_PER_TB": "5.00",
            "INPUT_SQL_PATHS": "",
            "INPUT_MANIFEST_PATH": "",
            "INPUT_GCP_CREDENTIALS": "",
            "INPUT_GCP_PROJECT": "",
            "INPUT_API_KEY": "",
            "INPUT_CONFIG_PATH": ".diffcost.yml",
            "INPUT_BASE_REF": "main",
            "INPUT_COMMENT_ON_CLEAN": "false",
            "GITHUB_TOKEN": "",
            "GITHUB_REPOSITORY": "",
            "GITHUB_EVENT_NAME": "",
            "GITHUB_WORKSPACE": "",
            "GITHUB_REF": "",
        }
        with patch.dict(os.environ, env, clear=False):
            config = load_config()
            assert config.cost_per_tb == 5.00


def _env(workspace: str, **extra) -> dict:
    env = {
        "INPUT_SQL_PATHS": "",
        "INPUT_MANIFEST_PATH": "",
        "INPUT_GCP_CREDENTIALS": "",
        "INPUT_GCP_PROJECT": "",
        "INPUT_API_KEY": "",
        "INPUT_CONFIG_PATH": ".diffcost.yml",
        "INPUT_BASE_REF": "",
        "INPUT_COST_PER_TB": "6.25",
        "INPUT_COMMENT_ON_CLEAN": "false",
        "INPUT_GITHUB_TOKEN": "",
        "GITHUB_TOKEN": "",
        "GITHUB_BASE_REF": "",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_WORKSPACE": workspace,
        "GITHUB_REF": "refs/pull/5/merge",
        "GITHUB_EVENT_PATH": "",
    }
    env.update(extra)
    return env


class TestInputsAndPolicyFile:
    def test_github_token_input_preferred(self, tmp_path):
        with patch.dict(os.environ, _env(str(tmp_path), INPUT_GITHUB_TOKEN="ghs_input", GITHUB_TOKEN="ghp_env")):
            assert load_config().github_token == "ghs_input"

    def test_github_token_env_fallback(self, tmp_path):
        with patch.dict(os.environ, _env(str(tmp_path), GITHUB_TOKEN="ghp_env")):
            assert load_config().github_token == "ghp_env"

    def test_base_ref_defaults_to_pr_target_branch(self, checkout_repo):
        repo = checkout_repo({"a.sql": "SELECT 1"}, base_branch="develop")
        with patch.dict(os.environ, _env(str(repo.path), GITHUB_BASE_REF="develop")):
            config = load_config()
        assert config.base_ref == "develop"
        assert config.resolved_base_ref == repo.git("rev-parse", "origin/develop").strip()
        assert config.git_error == ""

    def test_unresolvable_base_ref_recorded_not_raised(self, tmp_path):
        with patch.dict(os.environ, _env(str(tmp_path), INPUT_BASE_REF="main")):
            config = load_config()
        assert config.resolved_base_ref == ""
        assert config.git_error

    def test_invalid_cost_per_tb_is_a_warning(self, tmp_path, capsys):
        with patch.dict(os.environ, _env(str(tmp_path), INPUT_COST_PER_TB="")):
            assert load_config().cost_per_tb == 6.25
        with patch.dict(os.environ, _env(str(tmp_path), INPUT_COST_PER_TB="abc")):
            assert load_config().cost_per_tb == 6.25
        assert "must be a number" in capsys.readouterr().out

    @pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity"])
    def test_non_finite_cost_per_tb_is_a_warning(self, tmp_path, capsys, raw):
        """`float()` accepts these, and a NaN rate would turn every cost into NaN."""
        with patch.dict(os.environ, _env(str(tmp_path), INPUT_COST_PER_TB=raw)):
            assert load_config().cost_per_tb == 6.25
        assert "Input `cost_per_tb` must be a number" in capsys.readouterr().out

    def test_yml_read_from_base_branch_not_pr(self, checkout_repo):
        repo = checkout_repo({
            ".diffcost.yml": "thresholds:\n  absolute: 5\n  relative: 50\ndefault_runs_per_month: 60\n",
        })
        # The PR tries to weaken the threshold — must be ignored
        repo.write(".diffcost.yml", "thresholds:\n  absolute: 100000\n")
        repo.commit()
        with patch.dict(os.environ, _env(str(repo.path))):
            config = load_config()
        assert config.thresholds.absolute_usd == 5.0
        assert config.yml_thresholds.relative_pct == 50.0
        assert config.default_runs_per_month == 60

    def test_invalid_yml_values_are_warnings(self, checkout_repo, capsys):
        repo = checkout_repo({
            ".diffcost.yml": "thresholds:\n  absolute: \"$5\"\n  relative: null\ndefault_runs_per_month: 4.5x\n",
        })
        with patch.dict(os.environ, _env(str(repo.path))):
            config = load_config()
        out = capsys.readouterr().out
        assert config.thresholds.absolute_usd is None
        assert config.thresholds.relative_pct is None
        assert config.default_runs_per_month == 30
        assert "thresholds.absolute must be a number" in out

    def test_malformed_yaml_is_a_warning(self, checkout_repo, capsys):
        repo = checkout_repo({".diffcost.yml": "thresholds: [unclosed\n"})
        with patch.dict(os.environ, _env(str(repo.path))):
            load_config()
        out = capsys.readouterr().out
        assert "not valid YAML" in out
        # PyYAML's error spans several lines; the annotation must keep the whole warning
        warning = next(line for line in out.splitlines() if "not valid YAML" in line)
        assert warning.startswith("::warning::") and warning.endswith("Its thresholds were not applied.")


def _event(tmp_path, base_sha="", head_sha="", login="alice"):
    import json

    path = tmp_path / "event.json"
    path.write_text(json.dumps({"pull_request": {
        "number": 5, "base": {"sha": base_sha}, "head": {"sha": head_sha}, "user": {"login": login},
    }}))
    return str(path)


class TestPullRequestEvent:
    def test_event_base_sha_is_the_diff_base(self, checkout_repo, tmp_path):
        repo = checkout_repo({"a.sql": "SELECT 1"})
        base = repo.git("rev-parse", "origin/main").strip()
        repo.write("a.sql", "SELECT 2")
        repo.commit("pr")
        head = repo.git("rev-parse", "HEAD").strip()
        repo.git("tag", "origin/main", "HEAD")  # must not shadow the base
        with patch.dict(os.environ, _env(str(repo.path), GITHUB_EVENT_PATH=_event(tmp_path, base, head))):
            config = load_config()
        assert config.git_error == ""
        assert config.resolved_base_ref == base
        assert config.pr_author == "alice"

    def test_base_branch_checkout_is_rejected(self, checkout_repo, tmp_path):
        """pull_request_target + default checkout = the base branch: must not look like 'no changes'."""
        repo = checkout_repo({"a.sql": "SELECT 1"})
        base = repo.git("rev-parse", "origin/main").strip()
        with patch.dict(os.environ, _env(str(repo.path), GITHUB_EVENT_PATH=_event(tmp_path, base, "b" * 40))):
            config = load_config()
        assert "pull_request_target" in config.git_error
        assert config.resolved_base_ref == ""

    def test_merge_commit_checkout_diffs_against_its_base_parent(self, checkout_repo, tmp_path):
        repo = checkout_repo({"a.sql": "SELECT 1"})
        base = repo.git("rev-parse", "HEAD").strip()
        repo.git("checkout", "-q", "-b", "feature")
        repo.write("a.sql", "SELECT 2")
        repo.commit("pr")
        head = repo.git("rev-parse", "HEAD").strip()
        repo.git("checkout", "-q", "--detach", base)
        repo.git("merge", "-q", "--no-ff", "-m", "merge", "feature")
        with patch.dict(os.environ, _env(str(repo.path), GITHUB_EVENT_PATH=_event(tmp_path, base, head))):
            config = load_config()
        assert config.git_error == ""
        assert config.resolved_base_ref == base

    def test_explicit_base_ref_input_wins_over_event(self, checkout_repo, tmp_path):
        repo = checkout_repo({"a.sql": "SELECT 1"})
        head = repo.git("rev-parse", "HEAD").strip()
        with patch.dict(os.environ, _env(str(repo.path), INPUT_BASE_REF="main",
                                         GITHUB_EVENT_PATH=_event(tmp_path, "f" * 40, head))):
            config = load_config()
        assert config.git_error == ""
        assert config.resolved_base_ref == repo.git("rev-parse", "origin/main").strip()


class TestInvalidInputs:
    def test_invalid_base_ref_is_recorded_not_raised(self, tmp_path):
        with patch.dict(os.environ, _env(str(tmp_path), INPUT_BASE_REF="--upload-pack=x")):
            config = load_config()
        assert "not a valid branch name" in config.git_error

    def test_branch_names_git_allows_are_accepted(self, checkout_repo):
        repo = checkout_repo({"a.sql": "SELECT 1"}, base_branch="release/1.0+hf")
        with patch.dict(os.environ, _env(str(repo.path), GITHUB_BASE_REF="release/1.0+hf")):
            config = load_config()
        assert config.git_error == ""

    def test_path_traversal_is_a_config_error(self, tmp_path):
        with patch.dict(os.environ, _env(str(tmp_path), INPUT_MANIFEST_PATH="../../etc/x.json")):
            config = load_config()
        assert "relative path" in config.config_error
        assert config.manifest_path == ""

    def test_unreadable_policy_is_recorded_for_the_comment(self, checkout_repo):
        repo = checkout_repo({".diffcost.yml": "thresholds: [unclosed\n"})
        with patch.dict(os.environ, _env(str(repo.path))):
            config = load_config()
        assert config.policy_warnings and "not valid YAML" in config.policy_warnings[0]


class TestYmlKeysAndBounds:
    def test_only_bq_pricing_model_key_is_read(self, checkout_repo):
        """Only `bq_pricing_model` is read: `pricing_model` is not an alias."""
        repo = checkout_repo({".diffcost.yml": "pricing_model: editions\n"})
        with patch.dict(os.environ, _env(str(repo.path))):
            assert load_config().bq_pricing_model == "on_demand"

    def test_bq_pricing_model_key(self, checkout_repo):
        repo = checkout_repo({".diffcost.yml": "bq_pricing_model: editions\n"})
        with patch.dict(os.environ, _env(str(repo.path))):
            assert load_config().bq_pricing_model == "editions"

    def test_yml_thresholds_match_backend_bounds(self, checkout_repo):
        """Out-of-range values are dropped with a policy warning (the backend would 422 the whole
        comparison); accepted values are rounded to the backend's 4 decimal places."""
        repo = checkout_repo({".diffcost.yml": "thresholds:\n  absolute: 2000000\n  relative: 12.345678\n"})
        with patch.dict(os.environ, _env(str(repo.path))):
            config = load_config()
        assert config.yml_thresholds.absolute_usd is None
        assert config.yml_thresholds.relative_pct == 12.3457
        assert any("thresholds.absolute" in w for w in config.policy_warnings)

    def test_relative_threshold_cap_matches_the_backend(self, checkout_repo):
        """The backend accepts relative limits up to 100,000%; the Action must not drop them."""
        repo = checkout_repo({".diffcost.yml": "thresholds:\n  relative: 50000\n"})
        with patch.dict(os.environ, _env(str(repo.path))):
            config = load_config()
        assert config.yml_thresholds.relative_pct == 50000.0
        assert not config.policy_warnings

    def test_relative_threshold_above_the_cap_is_dropped(self, checkout_repo):
        repo = checkout_repo({".diffcost.yml": "thresholds:\n  relative: 100001\n"})
        with patch.dict(os.environ, _env(str(repo.path))):
            config = load_config()
        assert config.yml_thresholds.relative_pct is None
        assert any("thresholds.relative" in w and "100,000" in w for w in config.policy_warnings)


def _queue_event(tmp_path, head_sha="", base_sha="", head_ref=None, base_ref="refs/heads/main", **extra):
    import json

    group = {"head_sha": head_sha, "base_sha": base_sha, "base_ref": base_ref,
             "head_ref": head_ref if head_ref is not None else f"refs/heads/gh-readonly-queue/main/pr-77-{base_sha}"}
    group.update(extra)
    path = tmp_path / "queue_event.json"
    path.write_text(json.dumps({"merge_group": group}))
    return str(path)


def _queue_env(workspace: str, event_path: str, **extra) -> dict:
    return _env(workspace, GITHUB_EVENT_NAME="merge_group", GITHUB_REF="refs/heads/gh-readonly-queue/main/pr-77-x",
                GITHUB_EVENT_PATH=event_path, GITHUB_ACTOR="alice", **extra)


class TestMergeQueueConfig:
    """`merge_group` runs diff the queue commit against its parent."""

    def _queue_repo(self, checkout_repo, base_files=None):
        repo = checkout_repo(base_files or {"a.sql": "SELECT 1"})
        base = repo.git("rev-parse", "origin/main").strip()
        repo.write("a.sql", "SELECT 2")
        repo.write(".diffcost.yml", "thresholds:\n  absolute: 100000\n")  # the queue commit weakens it
        repo.commit("queue commit")
        return repo, base, repo.git("rev-parse", "HEAD").strip()

    def test_queue_commit_is_analyzed_against_its_parent(self, checkout_repo, tmp_path):
        repo, base, head = self._queue_repo(checkout_repo, {"a.sql": "SELECT 1", ".diffcost.yml": "thresholds:\n  absolute: 5\n"})
        with patch.dict(os.environ, _queue_env(str(repo.path), _queue_event(tmp_path, head, base))):
            config = load_config()
        assert config.merge_group is True
        assert config.git_error == "" and config.config_error == ""
        assert config.pr_number == 77
        assert config.commit_sha == head
        assert config.resolved_base_ref == base
        assert config.base_ref == "main"
        assert config.pr_author == ""
        assert config.thresholds.absolute_usd == 5.0  # .diffcost.yml from the queue's base, not the queue commit

    def test_pr_number_from_queue_branch_names(self):
        from config import merge_group_pr_number

        sha = "0123456789abcdef0123456789abcdef01234567"
        assert merge_group_pr_number(f"refs/heads/gh-readonly-queue/main/pr-42-{sha}") == 42
        assert merge_group_pr_number(f"refs/heads/gh-readonly-queue/release/1.0/pr-9-{sha}") == 9
        assert merge_group_pr_number(f"refs/heads/gh-readonly-queue/pr-3-x/pr-12-{sha}") == 12
        assert merge_group_pr_number("refs/heads/gh-readonly-queue/main/pr-42-abc") == 0
        assert merge_group_pr_number(f"refs/heads/gh-readonly-queue/main/pr-x-{sha}") == 0
        assert merge_group_pr_number("") == 0

    def test_unparseable_queue_branch_is_a_config_error_not_a_skip(self, checkout_repo, tmp_path):
        repo, base, head = self._queue_repo(checkout_repo)
        event = _queue_event(tmp_path, head, base, head_ref="refs/heads/some-other-branch")
        with patch.dict(os.environ, _queue_env(str(repo.path), event)):
            config = load_config()
        assert config.merge_group is True
        assert config.pr_number == 0
        assert "merge queue branch `refs/heads/some-other-branch`" in config.config_error

    def test_event_without_merge_group_payload_fails_closed(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("{}")
        with patch.dict(os.environ, _queue_env(str(tmp_path), str(path))):
            config = load_config()
        assert config.merge_group is True
        assert config.pr_number == 0
        assert config.config_error
        assert config.git_error  # no base commit either: never "no changes"

    def test_missing_queue_base_commit(self, checkout_repo, tmp_path):
        repo, _, head = self._queue_repo(checkout_repo)
        event = _queue_event(tmp_path, head, "", head_ref=f"refs/heads/gh-readonly-queue/main/pr-77-{'a' * 40}")
        with patch.dict(os.environ, _queue_env(str(repo.path), event)):
            config = load_config()
        assert "no base commit" in config.git_error
        assert config.resolved_base_ref == ""

    def test_queue_base_not_in_checkout(self, checkout_repo, tmp_path):
        repo, _, head = self._queue_repo(checkout_repo)
        with patch.dict(os.environ, _queue_env(str(repo.path), _queue_event(tmp_path, head, "f" * 40))):
            config = load_config()
        assert "merge queue's base commit" in config.git_error
        assert "fetch-depth: 0" in config.git_error
        assert config.resolved_base_ref == ""

    def test_checkout_must_be_the_queue_commit(self, checkout_repo, tmp_path):
        repo, base, _ = self._queue_repo(checkout_repo)
        with patch.dict(os.environ, _queue_env(str(repo.path), _queue_event(tmp_path, "c" * 40, base))):
            config = load_config()
        assert "not the merge queue commit" in config.git_error
        assert config.resolved_base_ref == ""


class TestNoApiKeyCauses:
    """An empty api_key names its likely cause."""

    def _load(self, tmp_path, pr: dict, actor="alice"):
        import json

        path = tmp_path / "pr.json"
        path.write_text(json.dumps({"pull_request": {"number": 5, **pr}}))
        with patch.dict(os.environ, _env(str(tmp_path), GITHUB_EVENT_PATH=str(path), GITHUB_ACTOR=actor)):
            return load_config()

    def test_fork_pr(self, tmp_path):
        config = self._load(tmp_path, {"head": {"repo": {"full_name": "someone/repo"}},
                                       "base": {"repo": {"full_name": "owner/repo"}}})
        assert config.fork_pr is True and config.dependabot_pr is False

    def test_same_repo_pr(self, tmp_path):
        config = self._load(tmp_path, {"head": {"repo": {"full_name": "owner/repo"}},
                                       "base": {"repo": {"full_name": "owner/repo"}}})
        assert config.fork_pr is False and config.dependabot_pr is False

    def test_dependabot_pr_by_author_or_actor(self, tmp_path):
        assert self._load(tmp_path, {"user": {"login": "dependabot[bot]"}}).dependabot_pr is True
        assert self._load(tmp_path, {"user": {"login": "alice"}}, actor="dependabot[bot]").dependabot_pr is True


class TestEventEdgeCases:
    """Events whose defaults differ: `pull_request_target`, Dependabot, an explicit base on a queue run."""

    def test_pull_request_target_fork_gets_secrets_so_no_fork_cause(self, tmp_path):
        import json

        path = tmp_path / "pr.json"
        path.write_text(json.dumps({"pull_request": {"number": 5, "head": {"repo": {"full_name": "someone/repo"}},
                                                     "base": {"repo": {"full_name": "owner/repo"}}}}))
        env = _env(str(tmp_path), GITHUB_EVENT_PATH=str(path), GITHUB_ACTOR="alice",
                   GITHUB_EVENT_NAME="pull_request_target")
        with patch.dict(os.environ, env):
            config = load_config()
        assert config.fork_pr is False
        assert config.dependabot_pr is False

    def test_dependabot_cause_on_pull_request_target(self, tmp_path):
        import json

        path = tmp_path / "pr.json"
        path.write_text(json.dumps({"pull_request": {"number": 5, "user": {"login": "dependabot[bot]"}}}))
        env = _env(str(tmp_path), GITHUB_EVENT_PATH=str(path), GITHUB_ACTOR="dependabot[bot]",
                   GITHUB_EVENT_NAME="pull_request_target")
        with patch.dict(os.environ, env):
            assert load_config().dependabot_pr is True

    def test_explicit_base_ref_wins_over_the_queue_base(self, checkout_repo, tmp_path):
        """An explicit `base_ref` input wins on merge queue runs too (it can only add
        changes: the PRs ahead in the queue are diffed as well)."""
        repo = checkout_repo({"a.sql": "SELECT 1"})
        main = repo.git("rev-parse", "origin/main").strip()
        repo.write("b.sql", "SELECT 1")
        repo.commit("PR ahead in the queue")
        parent = repo.git("rev-parse", "HEAD").strip()
        repo.write("a.sql", "SELECT 2")
        repo.commit("queue commit")
        head = repo.git("rev-parse", "HEAD").strip()
        event = _queue_event(tmp_path, head, parent)
        with patch.dict(os.environ, _queue_env(str(repo.path), event)):
            assert load_config().resolved_base_ref == parent
        with patch.dict(os.environ, _queue_env(str(repo.path), event, INPUT_BASE_REF="main")):
            config = load_config()
        assert config.git_error == ""
        assert config.resolved_base_ref == main
        assert config.base_ref == "main"
