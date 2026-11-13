"""Tests for the Action's main execution flow (per-run costs, worst-query thresholds, backend contract)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from adapters import ChangedSQL
from config import ActionConfig, ThresholdConfig
from gitutil import GitError
from main import (
    _build_payload,
    _get_plan,
    _send_to_backend,
    main,
    run,
    summarize,
)

GIB = 2**30


def _make_config(**kwargs) -> ActionConfig:
    defaults = {
        "sql_paths": "queries/**/*.sql",
        "manifest_path": "",
        "gcp_credentials": "",
        "gcp_project": "test-project",
        "api_key": "test-key",
        "config_path": ".diffcost.yml",
        "base_ref": "main",
        "cost_per_tb": 6.25,
        "comment_on_clean": False,
        "github_token": "token",
        "github_repository": "org/repo",
        "pr_number": 12,
        "commit_sha": "a" * 40,
        "resolved_base_ref": "origin/main",
    }
    defaults.update(kwargs)
    return ActionConfig(**defaults)


def _item(name, before_sql="SELECT 1", pr_sql="SELECT 2", adapter="sql_files", **meta):
    return ChangedSQL(
        name=name, path=f"queries/{name}.sql", production_sql=before_sql, pr_sql=pr_sql, adapter=adapter,
        metadata=meta or None,
    )


def _pair(before=None, after=None, error=None, unmaterialized=False):
    """A dry_run_pair result with per-run dollar costs."""
    return {
        "cost_before": before,
        "cost_after": after,
        "bytes_before": int(before * 1000) if before is not None else None,
        "bytes_after": int(after * 1000) if after is not None else None,
        "error": error,
        "unmaterialized_upstream": unmaterialized,
    }


@pytest.fixture
def harness(monkeypatch):
    """Patch the adapters, BigQuery, GitHub and backend I/O; expose the mocks."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    with patch("adapters.sql_files.find_changed_sql") as find, \
            patch("analysis.bq_dryrun.create_bq_client") as bq, \
            patch("analysis.bq_dryrun.dry_run_pair") as pair, \
            patch("comment.post_or_update_comment") as post, \
            patch("main._get_plan") as get_plan, \
            patch("main._send_to_backend") as send, \
            patch("main.set_output") as set_output:
        bq.return_value = MagicMock()
        post.return_value = "https://github.com/org/repo/pull/12#issuecomment-1"
        get_plan.return_value = "team"
        send.return_value = None
        h = MagicMock()
        h.find, h.bq, h.pair, h.post, h.get_plan, h.send, h.set_output = find, bq, pair, post, get_plan, send, set_output

        def outputs():
            return {c.args[0]: c.args[1] for c in set_output.call_args_list}

        h.outputs = outputs
        yield h


def _paid(config: ActionConfig, **thresholds) -> ActionConfig:
    config.features = ["merge_blocking", "slack", "thresholds"]
    config.thresholds = ThresholdConfig(**thresholds)
    return config


class TestEarlyExits:
    def test_not_in_pr_sets_outputs(self, harness):
        assert run(_make_config(pr_number=0)) == 0
        assert harness.outputs()["blocked"] == "false"

    def test_git_failure_is_reported_not_treated_as_no_changes(self, harness):
        harness.find.side_effect = GitError("Base branch `main` was not found")
        assert run(_make_config()) == 0
        body = harness.post.call_args.args[3]
        assert "could not analyze" in body
        harness.send.assert_not_called()  # no success status for an analysis that did not run
        assert harness.outputs()["files_analyzed"] == "0"

    def test_git_failure_with_fail_on_error_exits_1(self, harness):
        harness.find.side_effect = GitError("boom")
        assert run(_make_config(fail_on_error=True)) == 1

    def test_no_changes_still_reports_to_backend_and_updates_stale_comment(self, harness):
        harness.find.return_value = []
        harness.send.return_value = {"blocked": False, "enforcement": "active"}
        assert run(_make_config()) == 0
        payload = harness.send.call_args.args[1]
        assert payload["files_analyzed"] == 0
        # Only an existing comment is replaced (create=False); none is created
        assert harness.post.call_count == 1
        assert harness.post.call_args.kwargs["create"] is False
        assert "No SQL changes in the latest push" in harness.post.call_args.args[3]
        assert harness.outputs()["blocked"] == "false"
        assert harness.outputs()["total_cost_delta"] == "0.0000"


class TestPerRunWorstQuery:
    def test_offsetting_changes_still_block_on_worst_query(self, harness):
        """+$10/run on one query and -$10/run on another nets to $0 but must block a $5 threshold."""
        harness.find.return_value = [_item("a"), _item("b")]
        harness.pair.side_effect = [_pair(1.0, 11.0), _pair(12.0, 2.0)]
        config = _paid(_make_config(), absolute_usd=5.0)

        run(config)

        out = harness.outputs()
        assert out["policy_exceeded"] == "true"
        assert float(out["total_cost_delta"]) == pytest.approx(0.0)
        assert float(out["max_query_cost_delta"]) == pytest.approx(10.0)
        assert "per-query threshold" in harness.post.call_args.args[3]

    def test_small_per_run_increase_is_not_multiplied_by_30(self, harness):
        """+$0.20/run is $6/mo, but thresholds are per run: a $5 threshold must not block."""
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(0.10, 0.30)
        config = _paid(_make_config(), absolute_usd=5.0)

        run(config)

        out = harness.outputs()
        assert out["blocked"] == "false"
        assert float(out["total_cost_delta"]) == pytest.approx(0.20)
        assert float(out["monthly_cost_delta"]) == pytest.approx(6.0)

    def test_relative_threshold_uses_worst_query_not_total(self, harness):
        harness.find.return_value = [_item("cheap"), _item("big")]
        harness.pair.side_effect = [_pair(0.01, 1.0), _pair(100.0, 100.0)]
        config = _paid(_make_config(), relative_pct=200.0)
        run(config)
        assert harness.outputs()["policy_exceeded"] == "true"

    def test_no_merge_blocking_feature_means_no_local_block(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 100.0)
        config = _make_config()
        config.features = []
        config.thresholds = ThresholdConfig(absolute_usd=1.0)
        run(config)
        assert harness.outputs()["blocked"] == "false"


class TestUnknownCost:
    def test_unmaterialized_upstream_is_excluded_not_counted_as_savings(self, harness):
        """dry_run_pair sets unmaterialized_upstream when the production side reads a missing
        table (both costs unknown); a PR-side miss with a working base is 'not costed' instead."""
        harness.find.return_value = [_item("downstream")]
        harness.pair.return_value = _pair(before=None, after=None, unmaterialized=True)
        run(_make_config())
        body = harness.post.call_args.args[3]
        assert "Cost unavailable" in body
        assert "Removed" not in body
        assert "reduces" not in body.lower()
        payload = harness.send.call_args.args[1]
        assert payload["items_unknown"] == 1
        assert payload["queries"] == []
        assert payload["total_cost_delta"] is None

    def test_unknown_cost_blocks_when_threshold_set(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(error="Access Denied")
        run(_paid(_make_config(), absolute_usd=5.0))
        assert harness.outputs()["policy_exceeded"] == "true"
        assert harness.outputs()["items_unknown"] == "1"

    def test_estimate_bigquery_cannot_give_blocks_and_suggests_ignored_paths(self, harness):
        """A lower-bound estimate (an external table here) is unknown cost, never a figure."""
        harness.find.return_value = [_item("ext")]
        pair = _pair(before=1.0, after=0.0)
        pair.update(accuracy_before="PRECISE", accuracy_after="LOWER_BOUND")
        harness.pair.return_value = pair
        run(_paid(_make_config(), absolute_usd=5.0))
        assert harness.outputs()["policy_exceeded"] == "true"
        payload = harness.send.call_args.args[1]
        assert payload["items_unknown"] == 1 and payload["queries"] == []
        body = harness.post.call_args.args[3]
        assert "Cost unavailable: the dry run of the PR version gives only a lower bound" in body
        assert '`ignored_paths: ["queries/ext.sql"]`' in body
        assert "1 query could not be costed" in body

    def test_no_bigquery_credentials_counts_items_unknown(self, harness):
        harness.bq.return_value = None
        harness.find.return_value = [_item("a"), _item("b")]
        run(_make_config())
        assert harness.send.call_args.args[1]["items_unknown"] == 2

    def test_a_deletion_without_bigquery_credentials_is_unknown_and_blocks(self, harness):
        """Without dry runs a deleted query isn't known to only lower the cost."""
        harness.bq.return_value = None
        harness.find.return_value = [_item("old", pr_sql=None)]
        run(_paid(_make_config(), absolute_usd=5.0))
        assert harness.outputs()["items_unknown"] == "1"
        assert harness.outputs()["policy_exceeded"] == "true"

    def test_a_deletion_bigquery_cannot_estimate_does_not_block(self, harness):
        harness.find.return_value = [_item("old", pr_sql=None)]
        harness.pair.return_value = _pair(error="Production dry-run failed: Access Denied")
        run(_paid(_make_config(), absolute_usd=5.0))
        assert harness.outputs()["items_unknown"] == "0"
        assert harness.outputs()["policy_exceeded"] == "false"
        assert harness.send.call_args.args[1]["queries"] == []

    def test_dry_run_exception_becomes_error_row(self, harness):
        harness.find.return_value = [_item("a"), _item("b")]
        harness.pair.side_effect = [RuntimeError("boom"), _pair(1.0, 2.0)]
        assert run(_make_config()) == 0
        assert harness.outputs()["files_analyzed"] == "2"
        assert harness.outputs()["items_unknown"] == "1"


class TestBackendVerdict:
    def test_comment_updated_when_backend_blocks(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        harness.send.return_value = {
            "blocked": True, "block_reasons": ["Dashboard threshold exceeded"], "enforcement": "active",
        }

        assert run(_make_config()) == 0  # merge blocking is the commit status, the step passes

        assert harness.post.call_count == 1  # posted once, after the backend's verdict
        assert "Dashboard threshold exceeded" in harness.post.call_args.args[3]
        assert harness.outputs()["blocked"] == "true"

    def test_backend_unblocks_local_verdict(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = {"blocked": False, "block_reasons": [], "enforcement": "plan"}
        run(_paid(_make_config(), absolute_usd=5.0))
        assert harness.outputs()["blocked"] == "false"
        assert "Merge blocked" not in harness.post.call_args.args[3]

    def test_dashboard_reasons_when_the_local_check_used_yml(self, harness):
        """/v1/config failed, so the local check used .diffcost.yml ($5); the backend applied
        the dashboard threshold ($10). The comment gives the reason the status gives."""
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = {
            "blocked": True, "block_reasons": ["Worst query +$19.00/run exceeds $10.00/run limit"],
            "enforcement": "active", "threshold_source": "org",
        }
        config = _make_config(backend_error="unreachable")
        config.thresholds = ThresholdConfig(absolute_usd=5.0)
        run(config)
        body = harness.post.call_args.args[3]
        assert "exceeds $10.00/run limit" in body
        assert "$5.00/run per-query threshold" not in body

    def test_both_block_on_the_same_limits_keeps_the_query_name(self, harness):
        harness.find.return_value = [_item("orders")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = {
            "blocked": True, "block_reasons": ["Worst query +$19.00/run exceeds $5.00/run limit"],
            "enforcement": "active", "threshold_source": "org",
            "threshold_absolute_usd": 5.0, "threshold_relative_pct": None,
        }
        run(_paid(_make_config(), absolute_usd=5.0))
        body = harness.post.call_args.args[3]
        assert "orders" in body and "$5.00/run per-query threshold" in body
        assert "Worst query +$19.00" not in body

    def test_config_failed_and_backend_used_yml_keeps_local_reasons(self, harness):
        harness.find.return_value = [_item("orders")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = {
            "blocked": True, "block_reasons": ["Worst query +$19.00/run exceeds $5.00/run limit"],
            "enforcement": "active", "threshold_source": "yml",
            "threshold_absolute_usd": 5.0, "threshold_relative_pct": None,
        }
        config = _make_config(backend_error="unreachable")
        config.thresholds = ThresholdConfig(absolute_usd=5.0)
        run(config)
        body = harness.post.call_args.args[3]
        assert "orders" in body and "$5.00/run per-query threshold" in body

    @pytest.mark.parametrize("local, theirs, expected", [
        ((5.0, 50.0), (5.0, 100.0), "the 100% per-query threshold"),  # relative-only mismatch
        ((None, 50.0), (5.0, 50.0), "the $5.00/run per-query threshold"),  # None vs a value
        ((5.0, None), (5.00000001, None), "the $5.00/run per-query threshold"),  # within tolerance
    ])
    def test_reasons_use_the_limits_the_backend_applied(self, harness, local, theirs, expected):
        harness.find.return_value = [_item("orders")]
        harness.pair.return_value = _pair(1.0, 20.0)
        reasons = []
        if theirs[0] is not None:
            reasons.append(f"Worst query +$19.00/run exceeds ${theirs[0]:.2f}/run limit")
        if theirs[1] is not None:
            reasons.append(f"Worst query +1900% exceeds {theirs[1]:g}% limit")
        harness.send.return_value = {
            "blocked": True, "block_reasons": reasons, "enforcement": "active", "threshold_source": "org",
            "threshold_absolute_usd": theirs[0], "threshold_relative_pct": theirs[1],
        }
        run(_paid(_make_config(), absolute_usd=local[0], relative_pct=local[1]))
        body = harness.post.call_args.args[3]
        assert "orders: +" in body and expected in body  # the local wording names the query
        assert "Worst query" not in body

    def test_threshold_changed_since_the_config_fetch(self, harness):
        harness.find.return_value = [_item("orders")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = {
            "blocked": True, "block_reasons": ["Worst query +$19.00/run exceeds $2.00/run limit"],
            "enforcement": "active", "threshold_source": "org",
            "threshold_absolute_usd": 2.0, "threshold_relative_pct": None,
        }
        run(_paid(_make_config(), absolute_usd=5.0))
        body = harness.post.call_args.args[3]
        assert "orders: +$19.00/run exceeds the $2.00/run per-query threshold" in body
        assert "$5.00/run" not in body

    def test_backend_reasons_when_the_local_rerun_disagrees(self, harness):
        """The rerun must give the backend's reasons (same count); otherwise the status's words win."""
        harness.find.return_value = [_item("orders")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = {
            "blocked": True, "block_reasons": ["Worst query +$19.00/run exceeds $30.00/run limit"],
            "enforcement": "active", "threshold_source": "org",
            "threshold_absolute_usd": 30.0, "threshold_relative_pct": None,  # the local rerun doesn't block
        }
        run(_paid(_make_config(), absolute_usd=5.0, relative_pct=10.0))
        body = harness.post.call_args.args[3]
        assert "Worst query +$19.00/run exceeds $30.00/run limit" in body
        assert "per-query threshold" not in body

    def test_backend_failure_with_fail_on_error(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        harness.send.return_value = None
        assert run(_make_config(fail_on_error=False)) == 0
        assert run(_make_config(fail_on_error=True)) == 1


class TestPayload:
    def _results(self, config):
        from comment import classify_cost

        rows = []
        for name, before, after in [("fct_orders", 1.0, 3.0), ("stg_new", None, 0.5)]:
            dr = _pair(before, after)
            rows.append({
                "name": name,
                "path": f"models/secret_dataset/{name}.sql",
                "dry_run": dr,
                "structural_findings": [],
                "cost": classify_cost(dr, has_production=before is not None, has_pr=True),
            })
        return rows

    def test_payload_has_no_names_paths_or_sql(self):
        config = _make_config(api_key="secret-api-key")
        config.yml_thresholds = ThresholdConfig(absolute_usd=5.0)
        results = self._results(config)
        payload = _build_payload(config, summarize(results), results)

        raw = json.dumps(payload)
        for forbidden in ("fct_orders", "stg_new", "secret_dataset", "models/", "SELECT"):
            assert forbidden not in raw
        assert "results" not in payload
        # No per-query identifier of any kind: only bytes and per-run costs
        assert all(set(q) == {"bytes_before", "bytes_after", "cost_before", "cost_after"} for q in payload["queries"])
        assert "pr_comment_url" not in payload
        new_q = next(q for q in payload["queries"] if q["cost_before"] is None)
        assert new_q["cost_after"] == 0.5 and new_q["bytes_before"] is None
        assert payload["total_cost_delta"] == pytest.approx(2.5)
        assert payload["thresholds"] == {"absolute_usd": 5.0, "relative_pct": None}
        assert payload["runs_per_month"] == 30
        assert payload["pr_author"] == ""

    def test_payload_drops_bad_sha(self):
        config = _make_config(commit_sha="not-a-sha")
        assert _build_payload(config, summarize([]), [])["commit_sha"] == ""
        config = _make_config(commit_sha="ABCDEF" + "0" * 34)
        assert _build_payload(config, summarize([]), [])["commit_sha"] == "abcdef" + "0" * 34

    def test_send_author_opt_in(self):
        config = _make_config(send_author=True, pr_author="alice")
        assert _build_payload(config, summarize([]), [])["pr_author"] == "alice"
        assert _build_payload(_make_config(pr_author="alice"), summarize([]), [])["pr_author"] == ""


class TestSendToBackend:
    @patch("requests.post")
    def test_403_is_a_warning(self, mock_post, capsys):
        resp = MagicMock(status_code=403)
        resp.json.return_value = {"detail": "Repository owner does not match"}
        mock_post.return_value = resp
        assert _send_to_backend(_make_config(), {"repo": "x/y"}) is None
        assert "HTTP 403" in capsys.readouterr().out

    @patch("requests.post")
    def test_422_is_a_warning_even_with_fail_on_error(self, mock_post):
        resp = MagicMock(status_code=422)
        resp.json.return_value = {"detail": [{"msg": "bad"}]}
        mock_post.return_value = resp
        assert _send_to_backend(_make_config(fail_on_error=True), {}) is None

    @patch("requests.post")
    def test_connection_error_retried_then_none(self, mock_post):
        import requests

        mock_post.side_effect = requests.ConnectionError("refused")
        assert _send_to_backend(_make_config(), {}) is None
        assert mock_post.call_count == 2


class TestServerConfig:
    def test_returns_free_without_api_key(self):
        assert _get_plan(_make_config(api_key="")) == "free"

    @patch("requests.get")
    def test_server_settings_win(self, mock_get):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "plan": "team",
            "trial_days_remaining": 7,
            "features": ["merge_blocking", "slack"],
            "bq_pricing": {"model": "editions", "cost_per_tb": 3.0},
            "thresholds": {"absolute_usd": 0.0, "relative_pct": None},
            "ignored_paths": ["legacy/**"],  # no longer served; ignored if an old backend sends it
        }
        mock_get.return_value = resp
        config = _make_config(api_key="valid-key")
        config.thresholds = ThresholdConfig(absolute_usd=50.0, relative_pct=10.0)  # from .diffcost.yml

        assert _get_plan(config) == "team"
        assert config.features == ["merge_blocking", "slack"]
        assert config.bq_pricing_model == "editions"
        assert config.cost_per_tb == 3.0
        # A $0 dashboard threshold is a real threshold and replaces the yml pair entirely
        assert config.thresholds.absolute_usd == 0.0
        assert config.thresholds.relative_pct is None
        assert config.ignored_paths == []  # ignored paths come from .diffcost.yml only (ZDR)
        assert config.backend_error == ""

    @patch("requests.get")
    def test_server_cost_per_tib_used_for_dry_runs(self, mock_get, harness):
        harness.get_plan.side_effect = lambda c: __import__("main")._apply_server_config(
            c, {"plan": "starter", "features": [], "bq_pricing": {"model": "on_demand", "cost_per_tb": 5.0}}
        )
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        run(_make_config())
        assert harness.pair.call_args.args[3] == 5.0

    @patch("requests.get")
    def test_rejected_key_runs_free(self, mock_get, capsys):
        mock_get.return_value = MagicMock(status_code=401)
        assert _get_plan(_make_config(api_key="bad")) == "free"
        assert "rejected the API key" in capsys.readouterr().out


class TestSlack:
    """The .diffcost.yml webhook fires only when the .diffcost.yml
    policy blocks the PR; a dashboard threshold's webhook is the backend's to send."""

    def _run(self, harness, backend, config=None, pair=None):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = pair or _pair(1.0, 3.0)
        harness.send.return_value = backend
        run(config or _paid(_make_config(slack_webhook="https://hooks.slack.com/x"), absolute_usd=1.0))

    @patch("requests.post")
    def test_fires_when_the_yml_policy_blocks(self, mock_post, harness):
        self._run(harness, {"blocked": True, "block_reasons": ["x"], "enforcement": "active",
                            "threshold_source": "yml"})
        text = mock_post.call_args.kwargs["json"]["text"]
        assert "exceeds the cost policy in .diffcost.yml" in text
        assert "+$2.00/run" in text
        assert "≈$60.00/mo at 30 runs/mo" in text

    @patch("requests.post")
    def test_not_sent_when_the_pr_is_not_blocked(self, mock_post, harness):
        """A cost increase under the threshold no longer notifies on every push."""
        self._run(harness, {"blocked": False, "block_reasons": [], "enforcement": "active",
                            "threshold_source": "yml"})
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_not_sent_when_a_dashboard_threshold_applied(self, mock_post, harness):
        self._run(harness, {"blocked": True, "block_reasons": ["x"], "enforcement": "active",
                            "threshold_source": "org"})
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_without_a_backend_answer_uses_the_local_yml_verdict(self, mock_post, harness):
        self._run(harness, None)
        assert mock_post.call_count == 1
        mock_post.reset_mock()
        config = _paid(_make_config(slack_webhook="https://hooks.slack.com/x"), absolute_usd=1.0)
        config.dashboard_thresholds = True  # the local verdict came from a dashboard threshold
        self._run(harness, None, config)
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_names_the_query_when_a_dashboard_threshold_was_removed_during_the_run(self, mock_post, harness):
        """The config fetch returned a dashboard threshold ($5); by the time the results arrived
        it was deleted, so the backend applied .diffcost.yml ($1). Slack still names the query."""
        config = _paid(_make_config(slack_webhook="https://hooks.slack.com/x"), absolute_usd=5.0)
        config.dashboard_thresholds = True
        self._run(harness, {"blocked": True, "block_reasons": ["Worst query +$2.00/run exceeds $1.00/run limit"],
                            "enforcement": "active", "threshold_source": "yml",
                            "threshold_absolute_usd": 1.0, "threshold_relative_pct": None}, config)
        text = mock_post.call_args.kwargs["json"]["text"]
        assert "• a: +$2.00/run exceeds the $1.00/run per-query threshold" in text
        assert "Worst query" not in text

    @patch("requests.post")
    def test_requires_the_slack_feature(self, mock_post, harness):
        config = _make_config(slack_webhook="https://hooks.slack.com/x")
        config.features = ["merge_blocking"]
        config.thresholds = ThresholdConfig(absolute_usd=1.0)
        self._run(harness, {"blocked": True, "block_reasons": ["x"], "enforcement": "active",
                            "threshold_source": "yml"}, config)
        mock_post.assert_not_called()


class TestEntrypoint:
    def test_unhandled_error_exits_zero_by_default(self, monkeypatch):
        monkeypatch.delenv("INPUT_FAIL_ON_ERROR", raising=False)
        with patch("main.load_config", side_effect=ValueError("bad input")):
            assert main() == 0

    def test_unhandled_error_exits_one_with_fail_on_error(self, monkeypatch):
        monkeypatch.setenv("INPUT_FAIL_ON_ERROR", "true")
        with patch("main.load_config", side_effect=ValueError("bad input")):
            assert main() == 1

    def test_run_exception_respects_config_fail_on_error(self, monkeypatch):
        monkeypatch.delenv("INPUT_FAIL_ON_ERROR", raising=False)
        with patch("main.load_config", return_value=_make_config(fail_on_error=False)), \
                patch("main.run", side_effect=RuntimeError("x")):
            assert main() == 0


class TestLogLines:
    """File and model names come from the pull request, and a file name may contain a newline.
    A workflow command ends at a newline, so the rest of the name would start a new log line,
    which the runner could read as another command."""

    _NAME = "a\n::stop-commands::x\r\n::error::forged"

    def test_every_log_helper_writes_one_line(self, capsys):
        from main import log_error, log_info, log_notice, log_warning

        for log in (log_info, log_notice, log_warning, log_error):
            log(f"query {self._NAME} failed")
        lines = capsys.readouterr().out.splitlines()
        assert lines == [
            "query a ::stop-commands::x ::error::forged failed",
            "::notice::query a ::stop-commands::x ::error::forged failed",
            "::warning::query a ::stop-commands::x ::error::forged failed",
            "::error::query a ::stop-commands::x ::error::forged failed",
        ]

    def test_blocked_run_logs_the_query_name_on_one_line(self, harness, capsys):
        harness.find.return_value = [_item(self._NAME)]
        harness.pair.return_value = _pair(1.0, 20.0)
        run(_paid(_make_config(api_key=""), absolute_usd=5.0))
        lines = capsys.readouterr().out.splitlines()
        assert not [line for line in lines if line.startswith(("::stop-commands::", "::error::forged"))]
        assert any("a ::stop-commands::x ::error::forged: +$19.00/run exceeds" in line for line in lines)

    def test_unhandled_error_message_stays_on_one_line(self, monkeypatch, capsys):
        monkeypatch.delenv("INPUT_FAIL_ON_ERROR", raising=False)
        with patch("main.load_config", side_effect=ValueError("bad\n::stop-commands::x")):
            main()
        assert capsys.readouterr().out.splitlines() == ["::error::DiffCost failed: ValueError: bad ::stop-commands::x"]


class TestNotCosted:
    """PR-side 'depends on objects this PR creates' blocks only with a structural risk."""

    def _depends(self, before=1.0):
        return {"cost_before": before, "cost_after": None, "bytes_before": 1000, "bytes_after": None,
                "error": None, "unmaterialized_upstream": False, "pr_depends_on_new_objects": True}

    def test_added_column_without_structural_risk_does_not_block(self, harness):
        harness.find.return_value = [_item(
            "fct", "SELECT id FROM t", "SELECT id, discount_pct FROM t", adapter="dbt", changed_upstream=True,
        )]
        harness.pair.return_value = self._depends()
        run(_paid(_make_config(), absolute_usd=5.0))
        out = harness.outputs()
        assert out["items_unknown"] == "0"
        assert out["policy_exceeded"] == "false"
        assert "Not costed: depends on objects this PR creates" in harness.post.call_args.args[3]
        assert harness.send.call_args.args[1]["items_unknown"] == 0

    def test_structural_risk_makes_it_count_as_unknown(self, harness):
        harness.find.return_value = [_item(
            "fct", "SELECT id FROM t WHERE event_date >= '2026-01-01'", "SELECT *, discount_pct FROM t", adapter="dbt",
            changed_upstream=True,
        )]
        harness.pair.return_value = self._depends()
        run(_paid(_make_config(), absolute_usd=5.0))
        out = harness.outputs()
        assert out["items_unknown"] == "1"
        assert out["policy_exceeded"] == "true"
        assert "structural cost risk" in harness.post.call_args.args[3]

    def test_qualified_star_makes_it_count_as_unknown(self, harness):
        """`SELECT o.*` is a SELECT * too (a Column whose name is a Star)."""
        harness.find.return_value = [_item(
            "fct", "SELECT o.id, o.amount FROM orders o", "SELECT o.*, d.pct FROM orders o JOIN d ON o.id = d.id",
            adapter="dbt", changed_upstream=True,
        )]
        harness.pair.return_value = self._depends()
        run(_paid(_make_config(), absolute_usd=5.0))
        assert harness.outputs()["policy_exceeded"] == "true"

    def test_structural_analysis_that_could_not_run_counts_as_unknown(self, harness):
        """An existing model whose SQL sqlglot can't parse (here
        BigQuery pipe syntax) or whose checks fail can't be shown to have no risk: it blocks."""
        harness.find.return_value = [_item(
            "fct", "SELECT id FROM orders WHERE d >= '2026-01-01'", "FROM orders |> SELECT id, discount",
            adapter="dbt", changed_upstream=True,
        )]
        harness.pair.return_value = self._depends()
        run(_paid(_make_config(), absolute_usd=5.0))
        out = harness.outputs()
        assert out["items_unknown"] == "1" and out["policy_exceeded"] == "true"
        assert "(structural analysis could not run on it, so it counts as unknown)" in harness.post.call_args.args[3]

    def test_failed_structural_check_counts_as_unknown(self, harness):
        harness.find.return_value = [_item("fct", *self._NO_RISK, adapter="dbt", changed_upstream=True)]
        harness.pair.return_value = self._depends()
        with patch("analysis.structural._run_checks", side_effect=RecursionError):
            run(_paid(_make_config(), absolute_usd=5.0))
        assert harness.outputs()["policy_exceeded"] == "true"

    def test_new_model_stays_exempt_without_structural_analysis(self, harness):
        """A new model has no earlier version to compare: no structural analysis, still exempt."""
        harness.find.return_value = [_item("fct_new", None, "SELECT id, discount FROM orders", adapter="dbt",
                                           changed_upstream=True)]
        harness.pair.return_value = self._depends(before=None)
        run(_paid(_make_config(), absolute_usd=5.0))
        out = harness.outputs()
        assert out["items_unknown"] == "0" and out["policy_exceeded"] == "false"

    def test_plain_sql_file_reading_a_missing_object_is_unknown(self, harness):
        """The exemption is for dbt models only."""
        harness.find.return_value = [_item("report", "SELECT id FROM t", "SELECT id FROM new_table")]
        harness.pair.return_value = self._depends()
        run(_paid(_make_config(), absolute_usd=5.0))
        out = harness.outputs()
        assert out["items_unknown"] == "1"
        assert out["policy_exceeded"] == "true"
        assert "Not costed" not in harness.post.call_args.args[3]

    def test_dbt_model_with_nothing_changed_upstream_is_unknown(self, harness):
        """A column typo in a new model, or a manifest compiled
        against another target, is not "depends on objects this PR creates"."""
        harness.find.return_value = [_item("fct_new", None, "SELECT col_typo FROM t", adapter="dbt",
                                           changed_upstream=False)]
        harness.pair.return_value = {**self._depends(before=None), "pr_error": "Unrecognized name: col_typo at [1:8]"}
        run(_paid(_make_config(), absolute_usd=5.0))
        out = harness.outputs()
        assert out["items_unknown"] == "1"
        assert out["policy_exceeded"] == "true"
        body = harness.post.call_args.args[3]
        assert "Not costed" not in body
        assert "nothing upstream changes in this PR: Unrecognized name: col\\_typo" in body

    # No structural risk, so on the old rule (no upstream condition) these were exempt
    _NO_RISK = ("SELECT id, amount FROM orders", "SELECT id, amount, discount FROM orders")

    def _not_found_run(self, harness, missing, **meta):
        harness.find.return_value = [_item("fct", *self._NO_RISK, adapter="dbt", **meta)]
        harness.pair.return_value = {**self._depends(), "pr_missing_object": missing}
        run(_paid(_make_config(), absolute_usd=5.0))
        return harness.outputs(), harness.post.call_args.args[3]

    def test_not_found_with_nothing_changed_upstream_is_unknown(self, harness):
        out, body = self._not_found_run(harness, None)
        assert out["items_unknown"] == "1"
        assert "reads an object that does not exist, and nothing upstream changes" in body

    def test_not_found_matching_a_relation_this_pr_creates_is_exempt(self, harness):
        out, body = self._not_found_run(
            harness, {"kind": "table", "name": "p:ds.stg_new"},
            changed_upstream=True, new_upstream_relations=["`p`.`ds`.`stg_new`"],
        )
        assert out["items_unknown"] == "0"
        assert "Not costed" in body
        assert "stg_new" not in body  # the missing object's name is never shown

    def test_not_found_dataset_holding_a_relation_this_pr_creates_is_exempt(self, harness):
        out, _ = self._not_found_run(
            harness, {"kind": "dataset", "name": "p:marts_new"},
            changed_upstream=True, new_upstream_relations=["`p`.`marts_new`.`fct`"],
        )
        assert out["items_unknown"] == "0"

    def test_not_found_for_an_object_this_pr_does_not_create_is_unknown(self, harness):
        """Touching a parent no longer exempts a manifest compiled against a CI target."""
        out, body = self._not_found_run(
            harness, {"kind": "dataset", "name": "p:ci_pr_12"},
            changed_upstream=True, new_upstream_relations=["`p`.`prod`.`stg_new`"],
        )
        assert out["items_unknown"] == "1"
        assert "nothing upstream creates it in this PR" in body
        assert "ci_pr_12" not in body

    def test_unparseable_not_found_falls_back_to_the_changed_upstream_rule(self, harness):
        out, _ = self._not_found_run(harness, None, changed_upstream=True)
        assert out["items_unknown"] == "0"


class TestEnforcement:
    def _blocking_run(self, harness, backend):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = backend
        run(_paid(_make_config(), absolute_usd=5.0))
        return harness.outputs(), harness.post.call_args.args[3]

    def test_active_enforcement_posts_comment_once(self, harness):
        out, body = self._blocking_run(harness, {"blocked": True, "block_reasons": ["x"], "enforcement": "active"})
        assert out["blocked"] == "true" and out["policy_exceeded"] == "true"
        assert "Merge blocked" in body
        assert harness.post.call_count == 1  # verdict unchanged: no second write

    def test_app_not_installed_is_not_called_merge_blocked(self, harness):
        out, body = self._blocking_run(
            harness, {"blocked": True, "block_reasons": ["x"], "enforcement": "no_installation"}
        )
        assert out["blocked"] == "false" and out["policy_exceeded"] == "true"
        assert "Merge blocked" not in body
        assert "GitHub App is not installed" in body

    def test_backend_unreachable_says_no_status_was_posted(self, harness):
        out, body = self._blocking_run(harness, None)
        assert out["blocked"] == "false" and out["policy_exceeded"] == "true"
        assert "could not be reached" in body
        assert "Merge blocked" not in body

    def test_analysis_status_outputs(self, harness):
        harness.find.return_value = []
        run(_make_config())
        assert harness.outputs()["analysis_status"] == "no_changes"
        harness.set_output.reset_mock()
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        run(_make_config())
        assert harness.outputs()["analysis_status"] == "ok"
        harness.set_output.reset_mock()
        harness.find.side_effect = GitError("boom")
        run(_make_config())
        assert harness.outputs()["analysis_status"] == "error"

    def test_config_error_posts_failure_comment(self, harness):
        assert run(_make_config(config_error="`manifest_path` must be a relative path")) == 0
        assert "could not analyze" in harness.post.call_args.args[3]
        harness.send.assert_not_called()
        harness.find.assert_not_called()

    def test_entrypoint_exception_sets_outputs(self, monkeypatch):
        monkeypatch.delenv("INPUT_FAIL_ON_ERROR", raising=False)
        with patch("main.load_config", side_effect=RuntimeError("x")), patch("main.set_output") as so:
            assert main() == 0
        outputs = {c.args[0]: c.args[1] for c in so.call_args_list}
        assert outputs["analysis_status"] == "error"
        assert outputs["blocked"] == "false"


class TestPayloadCap:
    def _row(self, i, before, after):
        from comment import classify_cost

        dr = _pair(before, after)
        return {"name": f"q{i}", "path": f"q{i}.sql", "dry_run": dr, "structural_findings": [],
                "cost": classify_cost(dr)}

    def test_worst_query_survives_the_500_cap(self):
        rows = [self._row(i, 1.0, 1.01) for i in range(600)]
        rows[550] = self._row(550, 1.0, 41.0)  # the regression sorts late
        rows[580] = self._row(580, 0.0001, 0.01)  # the largest % increase
        payload = _build_payload(_make_config(), summarize(rows), rows)
        sent = {(q["cost_before"], q["cost_after"]) for q in payload["queries"]}
        assert len(payload["queries"]) == 500
        assert (1.0, 41.0) in sent
        assert (0.0001, 0.01) in sent


class TestServerConfigFlags:
    def test_trial_ended_and_no_legacy_feature_fallback(self):
        from main import _apply_server_config

        config = _make_config()
        assert _apply_server_config(config, {"plan": "free", "trial_ended": True}) == "free"
        assert config.trial_ended is True
        assert config.features == []  # no implied paid features without an explicit list


class TestAdapterDetection:
    def test_committed_manifest_cannot_switch_plain_sql_repo_to_dbt(self, checkout_repo):
        from main import detect_adapter

        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        repo.write("target/manifest.json", "{}")
        repo.commit()
        config = _make_config(sql_paths="", workspace=str(repo.path),
                              resolved_base_ref=repo.git("rev-parse", "origin/main").strip())
        assert detect_adapter(config) == "sql_files"

    def test_subdirectory_dbt_project_auto_detects(self, checkout_repo):
        from main import detect_adapter

        repo = checkout_repo({"transform/dbt_project.yml": "name: p", "transform/models/a.sql": "select 1"})
        repo.write("transform/target/manifest.json", "{}")
        config = _make_config(sql_paths="", workspace=str(repo.path),
                              resolved_base_ref=repo.git("rev-parse", "origin/main").strip())
        assert detect_adapter(config) == "dbt"
        assert config.manifest_path == "transform/target/manifest.json"

    def test_subdirectory_project_added_in_pr_is_not_auto_detected(self, checkout_repo):
        from main import detect_adapter

        repo = checkout_repo({"queries/a.sql": "SELECT 1"})
        repo.write("transform/dbt_project.yml", "name: p")
        repo.write("transform/target/manifest.json", "{}")
        repo.commit()
        config = _make_config(sql_paths="", workspace=str(repo.path),
                              resolved_base_ref=repo.git("rev-parse", "origin/main").strip())
        assert detect_adapter(config) == "sql_files"

    def test_installed_packages_are_not_projects(self, checkout_repo):
        from main import detect_adapter

        repo = checkout_repo({"dbt_project.yml": "name: p", "dbt_packages/utils/dbt_project.yml": "name: u"})
        repo.write("target/manifest.json", "{}")
        repo.write("dbt_packages/utils/target/manifest.json", "{}")
        config = _make_config(sql_paths="", workspace=str(repo.path),
                              resolved_base_ref=repo.git("rev-parse", "origin/main").strip())
        assert detect_adapter(config) == "dbt"
        assert config.manifest_path == "target/manifest.json"

    def test_several_dbt_projects_need_manifest_path(self, checkout_repo):
        from adapters.dbt import ManifestError
        from main import detect_adapter

        repo = checkout_repo({"a/dbt_project.yml": "name: a", "b/dbt_project.yml": "name: b"})
        repo.write("a/target/manifest.json", "{}")
        repo.write("b/target/manifest.json", "{}")
        config = _make_config(sql_paths="", workspace=str(repo.path),
                              resolved_base_ref=repo.git("rev-parse", "origin/main").strip())
        with pytest.raises(ManifestError) as exc:
            detect_adapter(config)
        assert "manifest_path" in str(exc.value)

    def test_dbt_repo_auto_detects(self, checkout_repo):
        from main import detect_adapter

        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select 1"})
        repo.write("target/manifest.json", "{}")
        config = _make_config(sql_paths="", workspace=str(repo.path),
                              resolved_base_ref=repo.git("rev-parse", "origin/main").strip())
        assert detect_adapter(config) == "dbt"

    def test_base_commit_is_listed_once_for_detection_and_the_dbt_adapter(self, checkout_repo, monkeypatch):
        """Adapter detection and the dbt adapter use one listing of the base commit. HEAD is
        listed separately: it can hold projects the PR adds."""
        import json

        import gitutil
        from adapters.dbt import find_changed_dbt_models
        from main import detect_adapter

        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select 1"})
        repo.write("target/manifest.json", json.dumps({"nodes": {"model.p.a": {
            "resource_type": "model", "name": "a", "original_file_path": "models/a.sql",
            "depends_on": {"nodes": []}, "compiled_code": "select 1",
        }}}))
        base = repo.git("rev-parse", "origin/main").strip()
        config = _make_config(sql_paths="", workspace=str(repo.path), resolved_base_ref=base)
        listed = []
        run_git = gitutil.run_git

        def recording(workspace, args, timeout=30):
            if args[0] == "ls-tree":
                listed.append(args[-1])
            return run_git(workspace, args, timeout=timeout)

        monkeypatch.setattr(gitutil, "run_git", recording)
        assert detect_adapter(config) == "dbt"
        find_changed_dbt_models(config)
        assert listed == [base, "HEAD"]

    def test_a_base_commit_that_cannot_be_listed_is_read_again(self, checkout_repo, monkeypatch):
        import gitutil

        repo = checkout_repo({"dbt_project.yml": "name: p"})
        config = _make_config(sql_paths="", workspace=str(repo.path),
                              resolved_base_ref=repo.git("rev-parse", "origin/main").strip())
        run_git = gitutil.run_git

        def failing(workspace, args, timeout=30):
            raise gitutil.GitError("git timed out")

        monkeypatch.setattr(gitutil, "run_git", failing)
        assert config.base_dbt_project_dirs() is None
        monkeypatch.setattr(gitutil, "run_git", run_git)
        assert config.base_dbt_project_dirs() == [""]
        config.resolved_base_ref = ""
        assert config.base_dbt_project_dirs() is None


class TestAdapterDetectionErrors:
    def _cfg(self, repo, **kw):
        kw.setdefault("sql_paths", "")
        return _make_config(workspace=str(repo.path),
                            resolved_base_ref=repo.git("rev-parse", "origin/main").strip(), **kw)

    def test_missing_explicit_manifest_is_an_error_not_a_jinja_scan(self, checkout_repo):
        """No fallback to dry-running uncompiled models as SQL."""
        from adapters.dbt import ManifestError
        from main import detect_adapter

        repo = checkout_repo({"transform/dbt_project.yml": "name: p", "transform/models/a.sql": "select 1"})
        with pytest.raises(ManifestError) as exc:
            detect_adapter(self._cfg(repo, manifest_path="target/manifest.json"))
        assert "dbt compile --target prod" in str(exc.value)
        assert "`transform/target/manifest.json`" in str(exc.value)

    def test_base_dbt_project_without_manifest_is_an_error(self, checkout_repo):
        from adapters.dbt import ManifestError
        from main import detect_adapter

        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select {{ ref('b') }}"})
        with pytest.raises(ManifestError) as exc:
            detect_adapter(self._cfg(repo))
        assert "sql_paths" in str(exc.value)

    def test_explicit_sql_paths_still_scan_sql_in_a_dbt_repo(self, checkout_repo):
        from main import detect_adapter

        repo = checkout_repo({"dbt_project.yml": "name: p", "queries/a.sql": "select 1"})
        assert detect_adapter(self._cfg(repo, sql_paths="queries/**/*.sql")) == "sql_files"

    def test_several_projects_message_does_not_recommend_several_runs(self, checkout_repo):
        """Several DiffCost runs would share one status and one comment."""
        from adapters.dbt import ManifestError
        from main import detect_adapter

        repo = checkout_repo({"a/dbt_project.yml": "name: a", "b/dbt_project.yml": "name: b"})
        repo.write("a/target/manifest.json", "{}")
        repo.write("b/target/manifest.json", "{}")
        with pytest.raises(ManifestError) as exc:
            detect_adapter(self._cfg(repo))
        msg = str(exc.value)
        assert "more than once on the same pull request is not supported" in msg
        assert "one DiffCost step per project" not in msg

    def test_run_reports_detection_errors_as_could_not_run(self, harness, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: p"})
        assert run(self._cfg(repo)) == 0
        assert "could not analyze" in harness.post.call_args.args[3]
        harness.find.assert_not_called()


class TestNoStatusPosted:
    def _run(self, harness, config, backend, pair=None):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = pair or _pair(1.0, 20.0)
        harness.send.return_value = backend
        code = run(config)
        return code, harness.outputs(), harness.post.call_args.args[3]

    def test_rejected_key_evaluates_yml_policy_and_says_not_enforced(self, harness):
        """A paying org with a bad key gets 'not enforced (key rejected)', not a Free upsell."""
        config = _make_config()
        config.thresholds = ThresholdConfig(absolute_usd=5.0)  # from .diffcost.yml
        config.backend_error = "rejected"  # set by _get_plan
        harness.send.side_effect = lambda c, p: setattr(c, "backend_error", "rejected")
        code, out, body = self._run(harness, config, None)
        assert code == 0
        assert out["policy_exceeded"] == "true" and out["blocked"] == "false"
        assert "API key was rejected" in body
        assert "DIFFCOST\\_API\\_KEY" in body or "DIFFCOST_API_KEY" in body
        assert "Starter plan" not in body  # no Free upsell

    def test_passing_pr_without_a_posted_status_gets_a_note(self, harness, capsys):
        """Without the note, the required check would wait with no explanation."""
        code, out, body = self._run(
            harness, _paid(_make_config(), absolute_usd=50.0),
            {"blocked": False, "block_reasons": [], "enforcement": "no_installation"},
        )
        assert out["policy_exceeded"] == "false"
        assert "No `diffcost/cost-check` status was posted" in body
        assert "Install the DiffCost GitHub App" in body
        assert "Merge blocking enabled" not in body
        assert "earlier run posted on this commit still applies" in body
        out = capsys.readouterr().out
        assert "No diffcost/cost-check commit status was posted" in out
        assert "earlier run posted on this commit still applies" in out
        assert "can't merge until it is" not in out

    def test_footer_claims_merge_blocking_only_when_active(self, harness):
        _, _, body = self._run(harness, _paid(_make_config(), absolute_usd=50.0),
                               {"blocked": False, "block_reasons": [], "enforcement": "active"})
        assert "Merge blocking enabled" in body
        assert "status was posted" not in body
        assert harness.post.call_count == 1

    def test_no_changes_run_without_status_creates_a_comment(self, harness):
        """With no earlier comment, the note must still reach the PR."""
        harness.find.return_value = []
        harness.send.return_value = {"blocked": False, "enforcement": "failed"}
        run(_make_config())
        assert harness.post.call_count == 1
        last = harness.post.call_args
        assert last.kwargs["create"] is True
        assert "No `diffcost/cost-check` status was posted" in last.args[3]

    def test_fail_on_error_does_not_cover_a_status_the_backend_could_not_post(self, harness, capsys):
        """fail_on_error covers 'the backend did not accept the results' only. A status the
        backend could not post is reported; a required check then keeps the PR waiting."""
        code, out, body = self._run(
            harness, _paid(_make_config(fail_on_error=True), absolute_usd=5.0),
            {"blocked": True, "block_reasons": ["x"], "enforcement": "failed"},
        )
        assert code == 0
        assert out["blocked"] == "false" and out["policy_exceeded"] == "true"
        assert "No `diffcost/cost-check` status was posted" in body
        assert "No diffcost/cost-check commit status was posted" in capsys.readouterr().out

    def test_invalid_policy_file_fails_the_step_with_fail_on_error(self, harness):
        """fail_on_error covers an unreadable .diffcost.yml (its thresholds were dropped)."""
        config = _make_config(fail_on_error=True)
        config.policy_warnings = [".diffcost.yml on main is not valid YAML"]
        code, _, _ = self._run(harness, config, {"blocked": False, "enforcement": "active"}, _pair(1.0, 1.0))
        assert code == 1
        config = _make_config(fail_on_error=False)
        config.policy_warnings = [".diffcost.yml on main is not valid YAML"]
        code, _, _ = self._run(harness, config, {"blocked": False, "enforcement": "active"}, _pair(1.0, 1.0))
        assert code == 0

    def test_report_written_to_the_job_summary(self, harness, tmp_path, monkeypatch):
        """Fork PRs can't be commented on; the job summary still has the report."""
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        self._run(harness, _make_config(), {"blocked": False, "enforcement": "plan"}, _pair(1.0, 2.0))
        text = summary.read_text()
        assert "BigQuery Cost Report" in text
        assert "diffcost-report" not in text

    def test_job_summary_keeps_a_report_too_long_for_a_comment(self, tmp_path, monkeypatch):
        """The comment is cut at GitHub's comment limit and points to the job summary, so the
        summary must keep the full report, up to the summary's own 1 MiB limit."""
        from comment import MAX_COMMENT_CHARS, MAX_SUMMARY_BYTES
        from main import _write_step_summary

        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        report = "<!-- diffcost-report -->\n" + "| `m` | $1 | $2 |\n" * 40_000 + "LAST ROW"
        assert MAX_COMMENT_CHARS * 10 < len(report.encode()) < MAX_SUMMARY_BYTES
        _write_step_summary(report)
        assert summary.read_text().rstrip().endswith("LAST ROW")

        # The limit counts UTF-8 bytes: 400,000 four-byte emoji are 1.6 MB.
        summary.write_text("")
        _write_step_summary("💰\n" * 400_000)
        text = summary.read_text()
        assert len(text.encode()) < 2**20 and "job summary size limit" in text


class TestBackendFailures:
    @patch("requests.post")
    def test_read_timeout_is_not_retried(self, mock_post):
        """The backend may already have stored the comparison and posted the status."""
        import requests

        mock_post.side_effect = requests.ReadTimeout("slow")
        config = _make_config()
        assert _send_to_backend(config, {}) is None
        assert mock_post.call_count == 1
        assert config.backend_error == "timeout"

    @patch("requests.post")
    def test_only_gateway_errors_are_retried(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        config = _make_config()
        assert _send_to_backend(config, {}) is None
        assert mock_post.call_count == 1 and config.backend_error == "error"
        mock_post.reset_mock()
        mock_post.return_value = MagicMock(status_code=503)
        assert _send_to_backend(config, {}) is None
        assert mock_post.call_count == 2

    @patch("requests.post")
    def test_rejected_and_invalid_are_distinguished(self, mock_post):
        config = _make_config()
        mock_post.return_value = MagicMock(status_code=401)
        _send_to_backend(config, {})
        assert config.backend_error == "rejected"
        mock_post.return_value = MagicMock(status_code=422)
        _send_to_backend(config, {})
        assert config.backend_error == "invalid"

    @patch("requests.post")
    @patch("requests.get")
    def test_rate_limited_is_its_own_reason(self, mock_get, mock_post):
        """The backend returns 429 for an IP with too many failed key checks;
        telling the user to update the Action ("invalid") would be wrong."""
        from comment import status_note

        config = _make_config(api_key="k")
        mock_get.return_value = MagicMock(status_code=429)
        assert _get_plan(config) == "free"
        assert config.backend_error == "rate_limited"
        mock_post.return_value = MagicMock(status_code=429)
        assert _send_to_backend(config, {}) is None
        assert config.backend_error == "rate_limited"
        note = status_note("rate_limited")
        assert "No `diffcost/cost-check` status was posted" in note
        assert "Wait a minute" in note

    @patch("requests.get")
    def test_config_fetch_records_why_it_failed(self, mock_get):
        import requests

        config = _make_config(api_key="bad")
        mock_get.return_value = MagicMock(status_code=403)
        _get_plan(config)
        assert config.backend_error == "rejected"
        mock_get.side_effect = requests.ConnectionError("down")
        config = _make_config(api_key="k")
        _get_plan(config)
        assert config.backend_error == "unreachable"


class TestCommentPostedOnce:
    """The comment is posted once, after the backend's verdict."""

    def test_rejected_config_never_says_merge_blocked(self, harness):
        config = _make_config()
        config.thresholds = ThresholdConfig(absolute_usd=5.0)
        config.backend_error = "rejected"  # GET /v1/config returned 401
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.side_effect = lambda c, p: setattr(c, "backend_error", "rejected")
        run(config)
        assert harness.post.call_count == 1
        body = harness.post.call_args.args[3]
        assert "Merge blocked" not in body
        assert "not enforced" in body

    def test_backend_called_before_the_comment(self, harness):
        order = []
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        harness.send.side_effect = lambda c, p: order.append("backend") or {"blocked": False, "enforcement": "active"}
        harness.post.side_effect = lambda *a, **k: order.append("comment")
        run(_make_config())
        assert order == ["backend", "comment"]


class TestMessages:
    def _blocking(self, harness, backend, send_effect=None, **cfg):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = backend
        if send_effect:
            harness.send.side_effect = send_effect
        run(_paid(_make_config(**cfg), absolute_usd=5.0))
        return harness.post.call_args.args[3]

    def test_timeout_says_unconfirmed_not_no_status(self, harness, capsys):
        """The backend may have posted the status after the Action gave up."""
        body = self._blocking(harness, None, send_effect=lambda c, p: setattr(c, "backend_error", "timeout"))
        assert "enforcement unconfirmed" in body
        assert "could not be confirmed" in body
        assert "No `diffcost/cost-check` status was posted" not in body
        out = capsys.readouterr().out
        assert "could not be confirmed" in out and "No diffcost/cost-check commit status was posted" not in out

    def test_invalid_fix_text_is_actionable(self, harness):
        body = self._blocking(harness, None, send_effect=lambda c, p: setattr(c, "backend_error", "invalid"))
        assert "latest `diffcost/action@v1`" in body and "support@diffcost.dev" in body

    def test_verdict_comes_before_the_table(self, harness):
        """Truncating a long report must not drop the verdict."""
        body = self._blocking(harness, {"blocked": True, "block_reasons": ["x"], "enforcement": "active"})
        assert body.index("Merge blocked") < body.index("| Query / Model")

    def test_editions_reason_is_labelled_on_demand_equivalent(self, harness):
        body = self._blocking(harness, {"blocked": True, "block_reasons": ["x"], "enforcement": "active"},
                              bq_pricing_model="editions")
        assert "(on-demand equivalent) exceeds" in body

    def test_failure_with_api_key_says_no_status(self, harness):
        harness.find.side_effect = GitError("boom")
        run(_make_config())
        assert "No `diffcost/cost-check` status was posted" in harness.post.call_args.args[3]
        harness.post.reset_mock()
        run(_make_config(api_key=""))
        assert "status was posted" not in harness.post.call_args.args[3]


class TestOutsideProjectHint:
    def test_unknown_outside_sql_blocks_and_shows_the_ignore_hint(self, harness):
        item = ChangedSQL(
            name="SQL outside the analyzed dbt project", path="queries/q.sql", production_sql=None, pr_sql=None,
            adapter="dbt", metadata={"unknown_reason": "1 changed .sql file(s) outside the dbt project were not "
                                     "analyzed, so their cost is unknown", "ignore_globs": ["queries/**"]},
        )
        harness.find.return_value = [item]
        harness.send.return_value = {"blocked": True, "block_reasons": ["1 query could not be costed"],
                                     "enforcement": "active"}
        run(_paid(_make_config(), absolute_usd=5.0))
        out = harness.outputs()
        assert out["items_unknown"] == "1" and out["policy_exceeded"] == "true"
        assert harness.send.call_args.args[1]["items_unknown"] == 1
        body = harness.post.call_args.args[3]
        assert 'ignored_paths: ["queries/**"]' in body
        harness.pair.assert_not_called()  # nothing to dry-run


class TestRepliesAndReasons:
    @patch("requests.post")
    @patch("requests.get")
    def test_non_dict_json_is_treated_as_empty(self, mock_get, mock_post):
        """A list body must not raise later (the comment would be lost)."""
        resp = MagicMock(status_code=200)
        resp.json.return_value = ["unexpected"]
        mock_get.return_value = resp
        mock_post.return_value = resp
        config = _make_config(api_key="k")
        assert _get_plan(config) == "free" and config.backend_error == ""
        # Not a verdict: unconfirmed, never "not blocked"
        assert _send_to_backend(config, {}) is None and config.backend_error == "unconfirmed"
        bad = MagicMock(status_code=500)
        bad.json.return_value = ["x"]
        mock_post.return_value = bad
        assert _send_to_backend(config, {}) is None

    def test_modified_model_without_baseline_is_not_dry_run(self, harness):
        """Its cost is unknown whatever the dry run says: don't spend the dry-run time budget."""
        harness.find.return_value = [
            _item("fct", None, "SELECT 2", adapter="dbt", baseline_unknown=True),
            _item("fct_new", None, "SELECT 3", adapter="dbt"),
        ]
        harness.pair.return_value = _pair(None, 1.0)
        run(_make_config())
        assert harness.pair.call_count == 1
        assert harness.pair.call_args.args[2] == "SELECT 3"
        assert harness.outputs()["items_unknown"] == "1"

    @patch("requests.post")
    def test_reply_that_is_not_a_result_is_unconfirmed(self, mock_post):
        """A 200 from a proxy or captive portal (HTML, empty, or JSON without `blocked`)."""
        html = MagicMock(status_code=200)
        html.json.side_effect = ValueError("not JSON")
        no_verdict = MagicMock(status_code=201)
        no_verdict.json.return_value = {"id": 1}
        for resp in (html, no_verdict):
            mock_post.return_value = resp
            config = _make_config(api_key="k")
            assert _send_to_backend(config, {}) is None
            assert config.backend_error == "unconfirmed"

    @patch("requests.post")
    def test_reply_that_is_not_a_result_keeps_the_local_verdict(self, mock_post, harness):
        """The local threshold breach must not read as a pass when the reply has no verdict."""
        html = MagicMock(status_code=200)
        html.json.side_effect = ValueError("not JSON")
        mock_post.return_value = html
        harness.send.side_effect = _send_to_backend  # the real one, over the patched requests.post
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        assert run(_paid(_make_config(), absolute_usd=5.0)) == 0
        body = harness.post.call_args.args[3]
        assert "enforcement unconfirmed" in body and "could not be confirmed" in body
        out = harness.outputs()
        assert out["policy_exceeded"] == "true" and out["blocked"] == "false"

    def test_rate_limited_text_names_the_key_and_the_ip(self):
        from comment import status_note

        note = status_note("rate_limited")
        assert "for this API key" in note and "this runner's IP address" in note

    def test_new_no_changes_comment_does_not_mention_earlier_results(self, harness):
        """A created comment uses the clean wording; an existing one is replaced."""
        harness.find.return_value = []
        harness.send.return_value = {"blocked": False, "enforcement": "no_installation"}
        run(_make_config())
        call = harness.post.call_args
        assert "Earlier cost results" in call.args[3]  # replaces an earlier comment
        assert "No SQL cost changes detected" in call.kwargs["body_if_new"]
        assert "Earlier cost results" not in call.kwargs["body_if_new"]
        assert "No `diffcost/cost-check` status was posted" in call.kwargs["body_if_new"]

    def test_backend_reasons_get_the_editions_label(self, harness):
        """The backend's verdict replaces the local one; its $ figures are on-demand equivalents."""
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        harness.send.return_value = {"blocked": True, "enforcement": "active",
                                     "block_reasons": ["Worst query +$1.00/run exceeds $0.50/run limit"]}
        config = _make_config(bq_pricing_model="editions")
        run(config)
        assert "limit (on-demand equivalent)" in harness.post.call_args.args[3]

    def test_sub_cent_threshold_reason_keeps_its_digits(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 1.0008)
        harness.send.return_value = {"blocked": True, "block_reasons": ["x"], "enforcement": "active"}
        run(_paid(_make_config(), absolute_usd=0.0005))
        body = harness.post.call_args.args[3]
        assert "+$0.0008/run exceeds the $0.0005/run per-query threshold" in body

    def test_threshold_reason_shows_cents_like_the_table(self, harness):
        """The table and the policy line must show the same figure ($14.82, not $14.8158)."""
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(0.0632, 14.879)
        harness.send.return_value = {"blocked": True, "block_reasons": ["x"], "enforcement": "active"}
        run(_paid(_make_config(), absolute_usd=1.0))
        body = harness.post.call_args.args[3]
        assert "+$14.82/run exceeds the $1.00/run per-query threshold" in body

    def test_threshold_reason_keeps_digits_just_over_the_limit(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.004)
        harness.send.return_value = {"blocked": True, "block_reasons": ["x"], "enforcement": "active"}
        run(_paid(_make_config(), absolute_usd=1.0))
        body = harness.post.call_args.args[3]
        assert "+$1.004/run exceeds the $1.00/run per-query threshold" in body

    def test_threshold_reason_under_a_dollar_matches_the_table(self, harness):
        """The policy line shows the table's figure: +$0.010, never +$0.0095."""
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(0.0632, 0.0728)
        harness.send.return_value = {"blocked": True, "block_reasons": ["x"], "enforcement": "active"}
        run(_paid(_make_config(), absolute_usd=0.005))
        body = harness.post.call_args.args[3]
        assert "+$0.010/run exceeds the $0.005/run per-query threshold" in body
        assert "+$0.010 (" in body  # the table's Cost Change cell

    def test_usd_limit_format(self):
        from comment import format_usd_limit

        assert format_usd_limit(5) == "$5.00"
        assert format_usd_limit(0.005) == "$0.005"
        assert format_usd_limit(0.0005) == "$0.0005"
        assert format_usd_limit(0.00004) == "<$0.0001"
        assert format_usd_limit(0) == "$0.00"
        assert format_usd_limit(-0.25) == "-$0.25"


def _slack_response(status=200, text="ok"):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


class TestMergeQueue:
    """`merge_group` runs analyze the queue commit, the backend
    posts the status on it, and the report goes to the job summary only."""

    def _summary(self, monkeypatch, tmp_path):
        path = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
        return path

    def test_queue_run_sends_the_flag_and_posts_no_comment(self, harness, monkeypatch, tmp_path, capsys):
        summary = self._summary(monkeypatch, tmp_path)
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        harness.send.return_value = {"blocked": False, "enforcement": "active"}
        assert run(_make_config(merge_group=True, pr_number=77, commit_sha="c" * 40)) == 0
        payload = harness.send.call_args.args[1]
        assert payload["merge_group"] is True
        assert payload["pr_number"] == 77
        assert payload["commit_sha"] == "c" * 40
        assert payload["pr_author"] == ""
        harness.post.assert_not_called()
        text = summary.read_text()
        assert text.startswith("🔀 **Merge queue run** for [#77](https://github.com/org/repo/pull/77).")
        assert "DiffCost" in text
        assert "Merge queue run for #77" in capsys.readouterr().out

    def test_pr_run_payload_flag_is_false(self):
        assert _build_payload(_make_config(), summarize([]), [])["merge_group"] is False

    def test_queue_run_without_sql_changes_still_gets_a_status(self, harness, monkeypatch, tmp_path):
        summary = self._summary(monkeypatch, tmp_path)
        harness.find.return_value = []
        harness.send.return_value = {"blocked": False, "enforcement": "active"}
        assert run(_make_config(merge_group=True, pr_number=77, comment_on_clean=True)) == 0
        assert harness.send.call_args.args[1]["files_analyzed"] == 0
        assert harness.send.call_args.args[1]["merge_group"] is True
        harness.post.assert_not_called()
        assert "Merge queue run" in summary.read_text()

    def test_unreadable_queue_branch_fails_instead_of_skipping(self, harness, monkeypatch, tmp_path, capsys):
        summary = self._summary(monkeypatch, tmp_path)
        config = _make_config(merge_group=True, pr_number=0, config_error="Could not find the pull request number")
        assert run(config) == 0
        out = capsys.readouterr().out
        assert "Not running in a pull request context" not in out
        assert "DiffCost could not run: Could not find the pull request number" in out
        harness.post.assert_not_called()
        harness.send.assert_not_called()
        text = summary.read_text()
        assert text.startswith("🔀 **Merge queue run** for a pull request.")
        assert "could not analyze" in text
        assert harness.outputs()["analysis_status"] == "error"
        config = _make_config(merge_group=True, pr_number=0, config_error="x", fail_on_error=True)
        assert run(config) == 1

    def test_queue_run_git_failure_posts_no_comment(self, harness):
        harness.find.side_effect = GitError("The merge queue's base commit `ffff` is not in the checkout.")
        assert run(_make_config(merge_group=True, pr_number=77)) == 0
        harness.post.assert_not_called()
        harness.send.assert_not_called()

    def test_other_events_are_still_skipped(self, harness, capsys):
        assert run(_make_config(pr_number=0)) == 0
        assert "Not running in a pull request context" in capsys.readouterr().out
        harness.find.assert_not_called()

    @patch("requests.post")
    def test_yml_slack_says_merge_queue(self, mock_post, harness):
        mock_post.return_value = _slack_response()
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 3.0)
        harness.send.return_value = {"blocked": True, "block_reasons": ["x"], "enforcement": "active",
                                     "threshold_source": "yml"}
        run(_paid(_make_config(merge_group=True, pr_number=77, slack_webhook="https://hooks.slack.com/x"),
                  absolute_usd=1.0))
        assert "org/repo#77> (merge queue) exceeds" in mock_post.call_args.kwargs["json"]["text"]


class TestSlackFailures:
    """A failed Slack alert is reported in the log and the comment."""

    WEBHOOK = "https://hooks.slack.com/services/T000/B000/secret-token"

    def _run(self, harness, backend=None, config=None):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 3.0)
        harness.send.return_value = backend or {"blocked": True, "block_reasons": ["x"], "enforcement": "active",
                                                "threshold_source": "yml"}
        run(config or _paid(_make_config(slack_webhook=self.WEBHOOK), absolute_usd=1.0))
        return harness.post.call_args.args[3]

    @patch("requests.post")
    def test_revoked_webhook_is_reported(self, mock_post, harness, capsys):
        mock_post.return_value = _slack_response(404, "no_service\n")
        body = self._run(harness)
        out = capsys.readouterr().out
        assert ("::warning::The Slack alert for the .diffcost.yml webhook failed: Slack returned HTTP 404: "
                "no_service. What to check: https://diffcost.dev/docs/troubleshooting#slack-alerts-dont-arrive") in out
        assert "**Slack alert failed** (.diffcost.yml webhook): Slack returned HTTP 404: no\\_service." in body
        assert "secret-token" not in out and "secret-token" not in body

    @patch("requests.post")
    def test_timeout_and_connection_errors(self, mock_post, harness):
        import requests

        mock_post.side_effect = requests.Timeout("https://hooks.slack.com/services/T000/B000/secret-token")
        assert "(.diffcost.yml webhook): timed out." in self._run(harness)
        mock_post.side_effect = requests.ConnectionError(self.WEBHOOK)
        body = self._run(harness)
        assert "request failed (ConnectionError)" in body
        assert "secret-token" not in body

    @patch("requests.post")
    def test_long_slack_body_is_cut(self, mock_post, harness):
        mock_post.return_value = _slack_response(500, "x" * 500)
        assert f"Slack returned HTTP 500: {'x' * 60}." in self._run(harness)

    @patch("requests.post")
    def test_success_adds_nothing(self, mock_post, harness, capsys):
        mock_post.return_value = _slack_response()
        body = self._run(harness)
        assert "Slack alert failed" not in body
        assert "Slack alert" not in capsys.readouterr().out

    @patch("requests.post")
    def test_slack_is_sent_before_the_comment(self, mock_post, harness):
        order = []
        mock_post.side_effect = lambda *a, **k: order.append("slack") or _slack_response()
        harness.post.side_effect = lambda *a, **k: order.append("comment")
        self._run(harness)
        assert order == ["slack", "comment"]

    def test_backend_slack_error_is_reported(self, harness, capsys):
        backend = {"blocked": True, "block_reasons": ["x"], "enforcement": "active", "threshold_source": "org",
                   "slack_error": "Slack returned HTTP 404: no_service\n::error::injected"}
        body = self._run(harness, backend, _paid(_make_config(), absolute_usd=1.0))
        out = capsys.readouterr().out
        assert "**Slack alert failed** (dashboard threshold webhook): Slack returned HTTP 404: no\\_service" in body
        assert ("::warning::The Slack alert for the dashboard threshold webhook failed: Slack returned HTTP 404: "
                "no_service ::error::injected.") in out
        assert "\n::error::injected" not in out  # one log line: no injected workflow command

    def test_no_backend_slack_error_adds_nothing(self, harness):
        backend = {"blocked": True, "block_reasons": ["x"], "enforcement": "active", "threshold_source": "org",
                   "slack_error": None}
        assert "Slack alert failed" not in self._run(harness, backend, _paid(_make_config(), absolute_usd=1.0))


class TestNoApiKey:
    """An empty api_key is said plainly, with its likely cause."""

    def _run(self, harness, **cfg):
        harness.get_plan.return_value = "free"
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        run(_make_config(api_key="", **cfg))
        return harness.post.call_args.args[3]

    def test_notice_and_footer_name_the_secret(self, harness, capsys):
        body = self._run(harness)
        out = capsys.readouterr().out
        assert ("::notice::No DiffCost API key (the api_key input is empty), so no diffcost/cost-check status is "
                "posted. Check that the `DIFFCOST_API_KEY` secret exists and the workflow passes it as `api_key`.") in out
        assert "::warning::No DiffCost API key" not in out
        assert "🔑 **No DiffCost API key**, so this run can't block merges." in body
        assert "Block expensive PRs" not in body
        harness.send.assert_not_called()

    def test_fork_pr_cause(self, harness, capsys):
        body = self._run(harness, fork_pr=True)
        assert "Pull requests from forks don't get repository secrets." in capsys.readouterr().out
        assert "The `api_key` input is empty: pull requests from forks don't get repository secrets." in body

    def test_dependabot_cause(self, harness):
        body = self._run(harness, dependabot_pr=True)
        assert "Dependabot pull requests only get Dependabot secrets: add `DIFFCOST_API_KEY` there too." in body

    def test_free_plan_with_a_key_keeps_the_upsell(self, harness, capsys):
        harness.get_plan.return_value = "free"
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        run(_make_config())
        body = harness.post.call_args.args[3]
        assert "block expensive PRs" in body
        assert "No DiffCost API key" not in body
        assert "::notice::" not in capsys.readouterr().out

    def test_no_changes_comment_and_summary(self, harness, monkeypatch, tmp_path):
        path = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
        harness.find.return_value = []
        run(_make_config(api_key="", comment_on_clean=True))
        assert "No DiffCost API key" in harness.post.call_args.args[3]
        assert "No DiffCost API key" in path.read_text()


class TestSummaryAndSlackSecrecy:
    """The job summary on a queue run, and a Slack failure reason that never holds the webhook URL."""

    WEBHOOK = "https://hooks.slack.com/services/T000/B000/secret-token"

    def _blocked(self, harness, config=None):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 3.0)
        harness.send.return_value = {"blocked": True, "block_reasons": ["x"], "enforcement": "active",
                                     "threshold_source": "yml"}
        run(config or _paid(_make_config(slack_webhook=self.WEBHOOK), absolute_usd=1.0))
        return harness.post.call_args.args[3]

    def test_no_key_line_in_a_merge_queue_job_summary(self, harness, monkeypatch, tmp_path):
        path = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
        harness.get_plan.return_value = "free"
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        run(_make_config(api_key="", merge_group=True, pr_number=77))
        harness.post.assert_not_called()
        text = path.read_text()
        assert text.startswith("🔀 **Merge queue run**")
        assert "🔑 **No DiffCost API key**" in text

    @patch("requests.post")
    def test_proxy_body_echoing_the_webhook_is_scrubbed(self, mock_post, harness, capsys):
        mock_post.return_value = _slack_response(
            403, f"Blocked by proxy: POST {self.WEBHOOK} denied; see http://proxy.corp/x"
        )
        body = self._blocked(harness)
        out = capsys.readouterr().out
        for text in (out, body):
            assert "secret-token" not in text
            assert "hooks.slack.com" not in text
            assert "proxy.corp" not in text
        assert "Slack returned HTTP 403: Blocked by proxy: POST <url> denied; see <url>" in out

    @patch("requests.post")
    def test_webhook_without_scheme_match_is_still_removed(self, mock_post, harness, capsys):
        """The configured value is removed even where it isn't matched as a URL."""
        mock_post.return_value = _slack_response(500, f"err:{self.WEBHOOK}")
        body = self._blocked(harness)
        assert "secret-token" not in capsys.readouterr().out and "secret-token" not in body

    @patch("requests.post")
    def test_exceptions_never_leak_the_url(self, mock_post, harness, capsys):
        import requests

        for exc in (requests.Timeout(self.WEBHOOK), requests.ConnectionError(self.WEBHOOK), ValueError(self.WEBHOOK)):
            mock_post.side_effect = exc
            body = self._blocked(harness)
            out = capsys.readouterr().out
            assert "secret-token" not in out and "secret-token" not in body
            expected = "timed out" if isinstance(exc, requests.Timeout) else f"request failed ({type(exc).__name__})"
            assert f"(.diffcost.yml webhook): {expected}." in body

    @patch("requests.post")
    def test_200_without_ok_is_a_failure(self, mock_post, harness):
        mock_post.return_value = _slack_response(200, "<html>captive portal</html>")
        assert "Slack returned HTTP 200: \\<html\\>captive portal\\</html\\>." in self._blocked(harness)
        mock_post.return_value = _slack_response(200, "ok\n")
        assert "Slack alert failed" not in self._blocked(harness)


class TestBillingStateAndPolicySource:
    """/v1/config's billing state and the backend's policy source."""

    def test_subscription_inactive_from_config(self, harness):
        harness.get_plan.side_effect = lambda c: __import__("main")._apply_server_config(
            c, {"plan": "free", "features": [], "subscription_inactive": True}
        )
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        harness.send.return_value = {"blocked": False, "block_reasons": [], "enforcement": "plan"}
        run(_make_config())
        body = harness.post.call_args.args[3]
        assert "**Your Stripe subscription isn't active.**" in body
        assert "DiffCost Free" not in body

    def test_missing_or_non_boolean_flag_is_not_inactive(self):
        from main import _apply_server_config

        for data in ({"plan": "free"}, {"plan": "free", "subscription_inactive": "yes"}):
            config = _make_config()
            _apply_server_config(config, data)
            assert config.subscription_inactive is False

    def test_backend_threshold_source_reaches_the_comment(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = {
            "blocked": True, "block_reasons": ["Worst query +$19.00/run exceeds $5.00/run limit"],
            "enforcement": "active", "threshold_source": "org", "threshold_absolute_usd": 5.0,
        }
        run(_paid(_make_config(), absolute_usd=5.0))
        assert "The policy that applied: the org-wide dashboard threshold (`*`)." in harness.post.call_args.args[3]

    def test_paid_run_without_a_threshold(self, harness):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 2.0)
        harness.send.return_value = {"blocked": False, "block_reasons": [], "enforcement": "active",
                                     "threshold_source": "none"}
        run(_paid(_make_config()))
        assert "Merge blocking enabled, but no threshold applies to this repository" in harness.post.call_args.args[3]

    def test_local_yml_check_names_the_file(self, harness):
        """Backend unreachable: the local check of .diffcost.yml is the verdict shown."""
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = None
        run(_paid(_make_config(), absolute_usd=5.0))
        body = harness.post.call_args.args[3]
        assert "Cost policy exceeded — not enforced" in body
        assert "The policy that applied: `.diffcost.yml` on the base branch." in body

    def test_yml_checked_on_the_runner_when_config_unavailable(self, harness):
        """The key was rejected at /v1/config and again at POST: the file was only checked locally."""
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = None
        config = _make_config(backend_error="rejected")
        config.thresholds = ThresholdConfig(absolute_usd=5.0)
        run(config)
        body = harness.post.call_args.args[3]
        assert "Cost policy exceeded — not enforced" in body
        assert "`.diffcost.yml` on the base branch, checked on the runner" in body

    def test_suspended_app(self, harness, capsys):
        harness.find.return_value = [_item("a")]
        harness.pair.return_value = _pair(1.0, 20.0)
        harness.send.return_value = {
            "blocked": True, "block_reasons": ["Worst query +$19.00/run exceeds $5.00/run limit"],
            "enforcement": "suspended", "threshold_source": "org",
        }
        run(_paid(_make_config(), absolute_usd=5.0))
        body = harness.post.call_args.args[3]
        assert "Cost policy exceeded — not enforced" in body
        assert "(the DiffCost GitHub App is suspended in this organization)" in body
        assert "unsuspend it in the organization's **Settings → GitHub Apps → DiffCost**" in body
        assert harness.outputs()["blocked"] == "false" and harness.outputs()["policy_exceeded"] == "true"
        assert "suspended in this organization" in capsys.readouterr().out
