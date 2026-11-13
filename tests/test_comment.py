"""Tests for PR comment formatting."""

from unittest.mock import patch, MagicMock

import pytest

from comment import (
    COMMENT_MARKER,
    MAX_COMMENT_CHARS,
    UPPER_BOUND_NOTE,
    classify_cost,
    format_comment,
    post_or_update_comment,
    truncate_body,
    truncate_summary,
)


class TestFormatCommentWithBQCreds:
    def test_cost_regression(self):
        results = [
            {
                "name": "fct_daily_revenue",
                "path": "models/fct_daily_revenue.sql",
                "dry_run": {
                    "cost_before": 2.30,
                    "cost_after": 10.62,
                    "bytes_before": 368_000_000_000,
                    "bytes_after": 1_699_200_000_000,
                    "error": None,
                },
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=2.30,
            total_after=10.62,
            total_delta=8.32,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        assert COMMENT_MARKER in body
        assert "fct_daily_revenue" in body
        assert "$2.30" in body
        assert "$10.62" in body
        assert "increases query costs" in body
        assert "DiffCost" in body
        assert "Data Scanned per run:" in body
        assert "/run" in body
        assert "/mo at 30 runs/mo" in body  # monthly projection is secondary text

    def test_cost_optimization(self):
        results = [
            {
                "name": "stg_orders",
                "path": "models/stg_orders.sql",
                "dry_run": {
                    "cost_before": 5.00,
                    "cost_after": 1.00,
                    "error": None,
                },
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=5.00,
            total_after=1.00,
            total_delta=-4.00,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        assert "reduces" in body.lower()

    def test_no_cost_change(self):
        results = [
            {
                "name": "dim_users",
                "path": "models/dim_users.sql",
                "dry_run": {
                    "cost_before": 0.50,
                    "cost_after": 0.50,
                    "error": None,
                },
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=0.50,
            total_after=0.50,
            total_delta=0.0,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        assert "No significant cost change" in body

    def test_blocked_pr(self):
        results = [
            {
                "name": "fct_revenue",
                "path": "models/fct_revenue.sql",
                "dry_run": {
                    "cost_before": 1.0,
                    "cost_after": 20.0,
                    "error": None,
                },
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=1.0,
            total_after=20.0,
            total_delta=19.0,
            has_bq_creds=True,
            blocked=True,
            block_reasons=["`fct_revenue`: cost increase $19.00 exceeds $5.00 threshold"],
            plan="team",
        )
        assert "Merge blocked" in body
        assert "$5.00 threshold" in body

    def test_multiple_results(self):
        results = [
            {
                "name": "model_a",
                "path": "a.sql",
                "dry_run": {"cost_before": 1.0, "cost_after": 5.0, "error": None},
                "structural_findings": [],
            },
            {
                "name": "model_b",
                "path": "b.sql",
                "dry_run": {"cost_before": 0.5, "cost_after": 0.5, "error": None},
                "structural_findings": [],
            },
        ]
        body = format_comment(
            results=results,
            total_before=1.5,
            total_after=5.5,
            total_delta=4.0,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        assert "TOTAL" in body
        assert "model_a" in body
        assert "model_b" in body

    def test_dry_run_error_shown(self):
        results = [
            {
                "name": "broken_query",
                "path": "broken.sql",
                "dry_run": {"error": "Table not found: broken_table", "cost_before": None, "cost_after": None},
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=0.0,
            total_after=0.0,
            total_delta=0.0,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        assert "Table not found" in body


class TestFormatCommentWithoutBQCreds:
    def test_structural_only_report(self):
        from analysis.structural import Finding

        results = [
            {
                "name": "daily_revenue",
                "path": "queries/daily_revenue.sql",
                "dry_run": None,
                "structural_findings": [
                    Finding("WHERE clause on `order_date` REMOVED", "high", "Partition filter removal"),
                ],
            }
        ]
        body = format_comment(
            results=results,
            total_before=0.0,
            total_after=0.0,
            total_delta=0.0,
            has_bq_creds=False,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        assert "SQL Change Analysis" in body
        assert "No BigQuery credentials" in body
        assert "daily_revenue" in body
        assert "Add BigQuery" in body  # setup CTA
        assert "diffcost.dev/docs/setup" not in body  # that route does not exist

    def test_no_findings(self):
        results = [
            {
                "name": "safe_query",
                "path": "queries/safe.sql",
                "dry_run": None,
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=0.0,
            total_after=0.0,
            total_delta=0.0,
            has_bq_creds=False,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        assert "No structural cost risks" in body


class TestFormatCTAs:
    def test_free_plan_shows_upgrade(self):
        results = [
            {
                "name": "q",
                "path": "q.sql",
                "dry_run": {"cost_before": 1.0, "cost_after": 2.0, "error": None},
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=1.0,
            total_after=2.0,
            total_delta=1.0,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        # A keyed Free run is a former payer: upgrade on Billing, never the trial on /pricing
        assert (
            "📊 **DiffCost Free:** merge blocking is off. [Upgrade](https://diffcost.dev/dashboard/billing) "
            "to Starter or Team to block expensive PRs."
        ) in body
        assert "https://diffcost.dev/pricing" not in body

    def test_pro_plan_no_upgrade_cta(self):
        results = [
            {
                "name": "q",
                "path": "q.sql",
                "dry_run": {"cost_before": 1.0, "cost_after": 2.0, "error": None},
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=1.0,
            total_after=2.0,
            total_delta=1.0,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="team",
        )
        # Paid (Team) users shouldn't see an upgrade CTA
        lines_with_pro = [line for line in body.split("\n") if "Pro" in line and "pricing" in line.lower()]
        assert len(lines_with_pro) == 0

    def test_trial_countdown_display(self):
        results = [
            {
                "name": "q",
                "path": "q.sql",
                "dry_run": {"cost_before": 1.0, "cost_after": 2.0, "error": None},
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=1.0,
            total_after=2.0,
            total_delta=1.0,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="trial",
            trial_days_remaining=3,
        )
        assert (
            "⏳ **DiffCost Team trial:** 3 days remaining. [Subscribe](https://diffcost.dev/dashboard/billing) "
            "to keep merge blocking after the trial (Starter $49/mo, Team $249/mo)."
        ) in body
        assert "Upgrade to Starter" not in body

    def test_editions_pricing_mode(self):
        results = [
            {
                "name": "fct_orders",
                "path": "models/fct_orders.sql",
                "dry_run": {
                    "cost_before": 1.0,
                    "cost_after": 3.0,
                    "bytes_before": 1_000_000_000,
                    "bytes_after": 3_000_000_000,
                    "error": None,
                },
                "structural_findings": [],
            }
        ]
        body = format_comment(
            results=results,
            total_before=1.0,
            total_after=3.0,
            total_delta=2.0,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="free",
            bq_pricing_model="editions",
        )
        assert "[Editions - Slot Based]" in body
        assert "3.0x scan" in body

    def test_unpartitioned_high_scan_warning_callout(self):
        from analysis.structural import Finding

        results = [
            {
                "name": "fct_daily_revenue",
                "path": "models/fct_daily_revenue.sql",
                "dry_run": {
                    "cost_before": 2.0,
                    "cost_after": 10.0,
                    "bytes_before": 320_000_000_000,
                    "bytes_after": 1_600_000_000_000,
                    "error": None,
                },
                "structural_findings": [
                    Finding(
                        description="WHERE clause on `order_date` REMOVED",
                        risk="high",
                        detail="Removing a partition filter causes a full table scan. This can dramatically increase query cost.",
                    )
                ],
            }
        ]
        body = format_comment(
            results=results,
            total_before=2.0,
            total_after=10.0,
            total_delta=8.0,
            has_bq_creds=True,
            blocked=False,
            block_reasons=[],
            plan="free",
        )
        assert "⚠️ **High Scan Warning:** Query scans full table without partition filter." in body


class TestPostOrUpdateComment:
    @patch("comment.requests")
    def test_creates_new_comment(self, mock_requests):
        # Mock listing comments (empty)
        list_response = MagicMock()
        list_response.status_code = 200
        list_response.json.return_value = []
        list_response.links = {}

        # Mock creating comment
        create_response = MagicMock()
        create_response.status_code = 201
        create_response.json.return_value = {"html_url": "https://github.com/owner/repo/pull/1#issuecomment-1"}

        mock_requests.get.return_value = list_response
        mock_requests.post.return_value = create_response

        result = post_or_update_comment("owner/repo", 1, "token", "body text")

        assert result == "https://github.com/owner/repo/pull/1#issuecomment-1"
        mock_requests.post.assert_called_once()

    @patch("comment.requests")
    def test_updates_existing_comment(self, mock_requests):
        # Mock listing comments (has existing DiffCost comment)
        list_response = MagicMock()
        list_response.status_code = 200
        list_response.json.return_value = [
            {
                "id": 123,
                "url": "https://api.github.com/repos/o/r/issues/comments/123",
                "body": f"old report {COMMENT_MARKER}",
                "user": {"login": "github-actions[bot]", "type": "Bot"},
            }
        ]
        list_response.links = {}

        # Mock updating comment
        update_response = MagicMock()
        update_response.status_code = 200
        update_response.json.return_value = {"html_url": "https://github.com/owner/repo/pull/1#issuecomment-123"}

        mock_requests.get.return_value = list_response
        mock_requests.patch.return_value = update_response

        result = post_or_update_comment("owner/repo", 1, "token", f"new report {COMMENT_MARKER}")

        assert result == "https://github.com/owner/repo/pull/1#issuecomment-123"
        mock_requests.patch.assert_called_once()
        mock_requests.post.assert_not_called()

    @patch("comment.requests")
    def test_handles_permission_error(self, mock_requests):
        list_response = MagicMock()
        list_response.status_code = 403

        mock_requests.get.return_value = list_response

        result = post_or_update_comment("owner/repo", 1, "token", "body")

        assert result is None

    def test_returns_none_when_token_empty(self):
        result = post_or_update_comment("owner/repo", 1, "", "body")
        assert result is None


class TestEscapingAndLimits:
    def _cost_result(self, name, before=1.0, after=2.0):
        return {
            "name": name,
            "path": f"{name}.sql",
            "dry_run": {"cost_before": before, "cost_after": after, "bytes_before": 10, "bytes_after": 20, "error": None},
            "structural_findings": [],
        }

    def test_model_name_cannot_break_out_of_table(self):
        evil = "x`\n\n✅ No significant cost change detected.\n<details><summary></summary>|"
        body = format_comment(
            results=[self._cost_result(evil)],
            total_before=1.0, total_after=2.0, total_delta=1.0,
            has_bq_creds=True, blocked=False, block_reasons=[], plan="free",
        )
        table_rows = [line for line in body.split("\n") if line.startswith("| ``")]
        assert len(table_rows) == 1
        # newline collapsed, pipe escaped, backticks wrapped in a double-backtick span
        assert "\\|" in table_rows[0]
        # The injected text stays inside the code span: no forged verdict line, no raw HTML line
        lines = body.split("\n")
        assert not any(line.strip() == "✅ No significant cost change detected." for line in lines)
        assert not any(line.startswith("<details><summary></summary>") for line in lines)

    def test_error_text_is_escaped(self):
        result = {
            "name": "q",
            "path": "q.sql",
            "dry_run": {"error": "bad | thing <script>", "cost_before": None, "cost_after": None},
            "structural_findings": [],
        }
        body = format_comment(
            results=[result], total_before=0, total_after=0, total_delta=0,
            has_bq_creds=True, blocked=False, block_reasons=[], plan="free",
        )
        assert "<script>" not in body
        assert "bad \\| thing" in body

    def test_body_truncated_under_github_limit(self):
        """The posted comment is cut to GitHub's limit and points to the job summary, which
        keeps a longer version (format_comment itself doesn't truncate)."""
        results = [self._cost_result(f"model_{i:05d}_" + "x" * 80) for i in range(1500)]
        full = format_comment(
            results=results, total_before=1500.0, total_after=3000.0, total_delta=1500.0,
            has_bq_creds=True, blocked=False, block_reasons=[], plan="free",
        )
        assert len(full) > MAX_COMMENT_CHARS and "model_01499_" in full
        body = truncate_body(full)
        assert len(body) <= MAX_COMMENT_CHARS
        assert "Report truncated" in body and "longer version is in the job summary" in body
        assert "workflow log" not in body
        assert body.startswith(COMMENT_MARKER)

    @patch("comment.requests")
    def test_posted_comment_is_truncated(self, mock_requests):
        mock_requests.get.return_value = MagicMock(status_code=200, json=lambda: [], links={})
        mock_requests.post.return_value = MagicMock(status_code=201, content=b"{}", json=lambda: {})
        post_or_update_comment("o/r", 1, "token", COMMENT_MARKER + "x\n" * 40_000)
        assert mock_requests.post.call_args.args[0].endswith("/repos/o/r/issues/1/comments")
        sent = mock_requests.post.call_args.kwargs["json"]["body"]
        assert len(sent) <= MAX_COMMENT_CHARS and "job summary" in sent

    def test_truncation_keeps_utf8_valid_and_closes_details(self):
        """A byte cut inside a multi-byte character must not crash or corrupt the summary, and a
        cut inside the collapsed root-cause section must not hide the note."""
        out = truncate_summary("é" * 100, limit_bytes=80)  # no newline, odd byte cut
        assert len(out.encode("utf-8")) <= 80 and "job summary size limit" in out
        body = "x\n<details>\n<summary>Root cause</summary>\n\n" + "- finding\n" * 20_000
        cut = truncate_body(body)
        assert len(cut) <= MAX_COMMENT_CHARS
        assert cut.index("</details>") < cut.index("Report truncated")

    def test_no_baseline_model_is_unknown_without_credentials(self):
        """Without a dry run (no credentials), an existing dbt model with no baseline must not
        read as "No structural changes"."""
        rec = classify_cost(None, has_production=None, has_pr=True, baseline_unknown=True)
        assert rec["status"] == "unknown" and "no production baseline" in rec["reason"]
        result = {"name": "fct_orders", "path": "models/fct_orders.sql", "dry_run": None,
                  "structural_findings": [], "cost": rec}
        body = format_comment(
            results=[result], total_before=0, total_after=0, total_delta=0,
            has_bq_creds=False, blocked=False, block_reasons=[], plan="free",
        )
        assert "Cost unavailable: no production baseline" in body
        assert "No structural changes" not in body

    def test_unknown_items_shown_as_cost_unavailable(self):
        result = {
            "name": "fct_new_join",
            "path": "m.sql",
            "dry_run": {"cost_before": None, "cost_after": None, "unmaterialized_upstream": True, "error": None},
            "structural_findings": [],
        }
        body = format_comment(
            results=[result], total_before=0, total_after=0, total_delta=0,
            has_bq_creds=True, blocked=False, block_reasons=[], plan="free", items_unknown=1,
        )
        assert "Cost unavailable" in body
        assert "Removed" not in body
        assert "reduces" not in body.lower()
        assert "1 query could not be costed" in body

    def test_worst_single_query_line(self):
        results = [self._cost_result("a", 1.0, 5.0), self._cost_result("b", 5.0, 1.5)]
        body = format_comment(
            results=results, total_before=6.0, total_after=6.5, total_delta=0.5,
            has_bq_creds=True, blocked=False, block_reasons=[], plan="free",
            worst_query={"name": "a", "delta": 4.0, "pct": 300.0},
        )
        assert "Largest single-query increase: `a` **+$4.00/run** (↑300%)" in body


class TestCommentPosting:
    def _list_resp(self, comments, next_url=None):
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = comments
        r.links = {"next": {"url": next_url}} if next_url else {}
        return r

    @patch("comment.requests")
    def test_paginates_to_find_existing_comment(self, mock_requests):
        filler = [{"id": i, "body": "hi", "user": {"type": "User"}} for i in range(100)]
        ours = {"id": 999, "url": "https://api.github.com/c/999", "body": COMMENT_MARKER, "user": {"type": "Bot"}}
        mock_requests.get.side_effect = [self._list_resp(filler, next_url="https://api.github.com/page2"),
                                         self._list_resp([ours])]
        patch_resp = MagicMock(status_code=200, content=b"x")
        patch_resp.json.return_value = {"html_url": "https://github.com/o/r/pull/1#issuecomment-999"}
        mock_requests.patch.return_value = patch_resp

        url = post_or_update_comment("o/r", 1, "t", f"{COMMENT_MARKER} new")

        assert url.endswith("999")
        assert mock_requests.get.call_args_list[0].kwargs["params"] == {"per_page": 100}
        mock_requests.post.assert_not_called()

    @patch("comment.requests")
    def test_ignores_marker_in_human_comment(self, mock_requests):
        human = {"id": 5, "url": "https://api.github.com/c/5", "body": f"quoting {COMMENT_MARKER}", "user": {"type": "User"}}
        mock_requests.get.return_value = self._list_resp([human])
        post_resp = MagicMock(status_code=201, content=b"x")
        post_resp.json.return_value = {"html_url": "https://github.com/o/r/pull/1#issuecomment-6"}
        mock_requests.post.return_value = post_resp

        post_or_update_comment("o/r", 1, "t", "body")

        mock_requests.patch.assert_not_called()
        mock_requests.post.assert_called_once()

    @patch("comment.requests")
    def test_create_false_only_updates(self, mock_requests):
        mock_requests.get.return_value = self._list_resp([])
        assert post_or_update_comment("o/r", 1, "t", "body", create=False) is None
        mock_requests.post.assert_not_called()

    @patch("comment.requests")
    def test_honors_github_api_url(self, mock_requests, monkeypatch):
        monkeypatch.setenv("GITHUB_API_URL", "https://ghe.example.com/api/v3")
        mock_requests.get.return_value = self._list_resp([])
        mock_requests.post.return_value = MagicMock(status_code=201, content=b"", text="")
        post_or_update_comment("o/r", 1, "t", "body")
        assert mock_requests.get.call_args.args[0].startswith("https://ghe.example.com/api/v3/repos/o/r/")


class TestVerdictAndEscaping:
    def _row(self, name="q", before=1.0, after=2.0, findings=None, **dr):
        dry = {"cost_before": before, "cost_after": after, "bytes_before": 10, "bytes_after": 20, "error": None}
        dry.update(dr)
        return {"name": name, "path": f"{name}.sql", "dry_run": dry, "structural_findings": findings or []}

    def _body(self, results, **kw):
        args = dict(total_before=1.0, total_after=2.0, total_delta=1.0, has_bq_creds=True,
                    blocked=False, block_reasons=[], plan="team")
        args.update(kw)
        return format_comment(results=results, **args)

    def test_code_fence_is_longer_than_any_backtick_run(self):
        from comment import md_code
        assert md_code("a``b") == "``` a``b ```"
        assert md_code("plain") == "`plain`"

    def test_mentions_and_urls_are_neutralised(self):
        from comment import md_text
        out = md_text("ping @octocat at https://evil.example and www.evil.example")
        assert "@octocat" not in out and "https://" not in out and "www.evil" not in out

    def test_shorten_never_cuts_mid_word(self):
        from comment import shorten
        text = "This is the most common cause of BQ cost explosions."
        assert shorten(text, 46) == "This is the most common cause of BQ cost…"

    def test_not_enforced_never_says_merge_blocked(self):
        body = self._body([self._row()], blocked=True, block_reasons=["q: +$1.00/run exceeds"],
                          not_enforced_reason="the DiffCost GitHub App is not installed for this repository")
        assert "Merge blocked" not in body
        assert "Cost policy exceeded — not enforced" in body
        assert "GitHub App is not installed" in body

    def test_enforced_says_merge_blocked(self):
        body = self._body([self._row()], blocked=True, block_reasons=["q: +$1.00/run exceeds"])
        assert "🚫 **Merge blocked**" in body

    def test_editions_with_nothing_costed_is_not_a_green_check(self):
        row = self._row(error="Access Denied")
        body = self._body([row], bq_pricing_model="editions", total_before=0, total_after=0, total_delta=0)
        assert "No query in this PR could be costed" in body
        assert "No scan volume change" not in body

    def test_offsetting_savings_headline_names_the_worst_query(self):
        body = self._body(
            [self._row("a", 1.0, 11.0), self._row("b", 12.0, 2.0)],
            total_before=13.0, total_after=13.0, total_delta=0.0,
            worst_query={"name": "a", "delta": 10.0, "pct": 1000.0},
        )
        assert "No significant cost change" not in body
        assert "+$10.00/run" in body and "offset" in body

    def test_not_costed_row_and_note(self):
        row = self._row(after=None, pr_depends_on_new_objects=True)
        row["cost"] = {"status": "not_costed", "cost_before": 1.0, "cost_after": None, "delta": None,
                       "pct": None, "reason": "depends on objects this PR creates", "blocking": False}
        body = self._body([row], items_not_costed=1, total_delta=0.0)
        assert "Not costed: depends on objects this PR creates" in body
        assert "does not block" in body

    def test_policy_warning_shown(self):
        body = self._body([self._row()], policy_warnings=[".diffcost.yml on main is not valid YAML"])
        assert "**Cost policy:**" in body

    def test_unusable_credentials_are_not_called_missing(self):
        body = self._body([self._row()], has_bq_creds=False, bq_error="RefreshError: invalid_grant")
        assert "could not be used" in body
        assert "No BigQuery credentials configured" not in body

    def test_trial_ended_line(self):
        body = self._body([self._row()], plan="free", trial_ended=True)
        assert "Your Team trial has ended" in body
        assert "Merge blocking, Slack alerts and cost trends are off" in body
        assert "to turn them back on" in body and "[upgrade](" in body

    def test_team_trial_countdown(self):
        body = self._body([self._row()], plan="team", trial_days_remaining=3)
        assert "Team trial:** 3 days remaining" in body

    def test_high_scan_banner_suppressed_when_dry_run_shows_no_increase(self):
        from analysis.structural import Finding
        f = Finding("Partition filter removed", "high", "Removing a partition filter causes a full table scan.")
        same = self._row(bytes_before=100, bytes_after=100, findings=[f])
        assert "High Scan Warning" not in self._body([same])
        more = self._row(bytes_before=100, bytes_after=1000, findings=[f])
        assert "High Scan Warning" in self._body([more])


class TestCommentIdentity:
    def _list(self, comments):
        r = MagicMock(status_code=200)
        r.json.return_value = comments
        r.links = {}
        return r

    @patch("comment.requests")
    def test_prefers_github_actions_bot_comment(self, mock_requests):
        other = {"id": 1, "url": "https://api.github.com/c/1", "body": COMMENT_MARKER,
                 "user": {"type": "Bot", "login": "other-app[bot]"}}
        ours = {"id": 2, "url": "https://api.github.com/c/2", "body": COMMENT_MARKER,
                "user": {"type": "Bot", "login": "github-actions[bot]"}}
        mock_requests.get.return_value = self._list([other, ours])
        mock_requests.patch.return_value = MagicMock(status_code=200, content=b"")
        post_or_update_comment("o/r", 1, "t", "body")
        assert mock_requests.patch.call_args.args[0].endswith("/c/2")

    @patch("comment.requests")
    def test_forbidden_edit_falls_back_to_new_comment(self, mock_requests):
        other = {"id": 1, "url": "https://api.github.com/c/1", "body": COMMENT_MARKER,
                 "user": {"type": "Bot", "login": "other-app[bot]"}}
        mock_requests.get.return_value = self._list([other])
        mock_requests.patch.return_value = MagicMock(status_code=403, content=b"", text="")
        created = MagicMock(status_code=201, content=b"x")
        created.json.return_value = {"html_url": "https://github.com/o/r/pull/1#issuecomment-9"}
        mock_requests.post.return_value = created
        assert post_or_update_comment("o/r", 1, "t", "body").endswith("9")

    @patch("comment.requests")
    def test_app_token_skips_the_default_bots_comment_it_cannot_edit(self, mock_requests):
        """After a switch from the default token to a GitHub App token: github-actions[bot]'s
        comment can't be edited, so the App's own comment is updated (no new one per run)."""
        default = {"id": 1, "url": "https://api.github.com/c/1", "body": COMMENT_MARKER,
                   "user": {"type": "Bot", "login": "github-actions[bot]"}}
        app = {"id": 2, "url": "https://api.github.com/c/2", "body": COMMENT_MARKER,
               "user": {"type": "Bot", "login": "my-app[bot]"}}
        mock_requests.get.return_value = self._list([app, default])
        ok = MagicMock(status_code=200, content=b"x")
        ok.json.return_value = {"html_url": "https://github.com/o/r/pull/1#issuecomment-2"}
        mock_requests.patch.side_effect = [MagicMock(status_code=403, content=b"", text=""), ok]
        assert post_or_update_comment("o/r", 1, "t", "body").endswith("2")
        assert [c.args[0][-4:] for c in mock_requests.patch.call_args_list] == ["/c/1", "/c/2"]
        mock_requests.post.assert_not_called()

    @patch("comment.requests")
    def test_personal_access_token_updates_its_own_comment(self, mock_requests):
        mine = {"id": 3, "url": "https://api.github.com/c/3", "body": COMMENT_MARKER,
                "user": {"type": "User", "login": "Octo-Dev"}}
        human = {"id": 4, "url": "https://api.github.com/c/4", "body": f"quoting {COMMENT_MARKER}",
                 "user": {"type": "User", "login": "someone-else"}}
        me = MagicMock(status_code=200)
        me.json.return_value = {"login": "octo-dev"}
        mock_requests.get.side_effect = lambda url, **kw: me if url.endswith("/user") else self._list([human, mine])
        ok = MagicMock(status_code=200, content=b"x")
        ok.json.return_value = {"html_url": "https://github.com/o/r/pull/1#issuecomment-3"}
        mock_requests.patch.return_value = ok
        assert post_or_update_comment("o/r", 1, "t", "body").endswith("3")
        assert mock_requests.patch.call_count == 1 and mock_requests.patch.call_args.args[0].endswith("/c/3")
        mock_requests.post.assert_not_called()

    @patch("comment.requests")
    def test_installation_token_never_edits_user_comments(self, mock_requests):
        """GET /user is 403 for the default token: a user comment with the marker is left alone."""
        human = {"id": 4, "url": "https://api.github.com/c/4", "body": COMMENT_MARKER,
                 "user": {"type": "User", "login": "octo-dev"}}
        forbidden = MagicMock(status_code=403)
        mock_requests.get.side_effect = lambda url, **kw: forbidden if url.endswith("/user") else self._list([human])
        created = MagicMock(status_code=201, content=b"x")
        created.json.return_value = {"html_url": "https://github.com/o/r/pull/1#issuecomment-9"}
        mock_requests.post.return_value = created
        assert post_or_update_comment("o/r", 1, "t", "body").endswith("9")
        mock_requests.patch.assert_not_called()

    @patch("comment.requests")
    def test_permission_hint_on_post_403(self, mock_requests, capsys):
        mock_requests.get.return_value = self._list([])
        mock_requests.post.return_value = MagicMock(status_code=403, content=b"", text="Resource not accessible")
        assert post_or_update_comment("o/r", 1, "t", "body") is None
        assert "pull-requests: write" in capsys.readouterr().out


class TestUnknownRowsAndNotes:
    def _unknown(self, name="stg_orders", reason="the model file changed but the PR manifest was not recompiled",
                 findings=None):
        return {"name": name, "path": f"models/{name}.sql", "dry_run": None, "structural_findings": findings or [],
                "cost": {"status": "unknown", "cost_before": None, "cost_after": None, "delta": None,
                         "pct": None, "reason": reason, "blocking": False}}

    def _body(self, results, **kw):
        args = dict(total_before=0.0, total_after=0.0, total_delta=0.0, has_bq_creds=True,
                    blocked=False, block_reasons=[], plan="team")
        args.update(kw)
        return format_comment(results=results, **args)

    def test_unknown_item_without_dry_run_gets_a_row(self):
        """The docs promise these rows: they must never be dropped from the table."""
        body = self._body([self._unknown()], items_unknown=1)
        assert "| `stg_orders` | — | — | — | ⚠️ Cost unavailable: the model file changed" in body

    def test_structural_findings_shown_for_items_without_dry_run(self):
        from analysis.structural import Finding

        f = Finding("SELECT * added", "high", "SELECT * scans all columns.")
        body = self._body([self._unknown(findings=[f])], items_unknown=1)
        assert "Root cause analysis" in body and "SELECT \\* added" in body

    def test_structural_only_report_lists_unknown_reasons(self):
        body = self._body([self._unknown(reason="the file is a symlink in this PR")], has_bq_creds=False)
        assert "Cost unavailable: the file is a symlink in this PR" in body
        assert "No structural cost risks detected" not in body

    def test_status_note_and_paid_footer(self):
        from comment import status_note

        note = status_note("no_installation")
        body = self._body([self._unknown()], status_note_line=note, merge_blocking_on=False)
        assert "No `diffcost/cost-check` status was posted" in body
        assert "Install the DiffCost GitHub App on this repository" in body
        # A re-run that posts nothing leaves the earlier run's status, maybe a passing one
        assert "A status an earlier run posted on this commit still applies, passing or failing." in body
        assert "can't merge until it is" not in body
        assert "🛡️ **DiffCost Team** [Manage subscription]" in body
        assert "Merge blocking enabled" not in body
        assert "Merge blocking enabled" in self._body([self._unknown()])

    def test_no_status_notes_never_say_the_pr_cannot_merge_outright(self):
        """An earlier run on the same commit (a re-run, a reopened PR) may have posted a status
        that still applies, so "can't merge until it is" would be untrue if that one passed."""
        from comment import format_failure_comment, status_note

        unconfirmed = status_note("timeout")
        assert "this commit has no status yet" in unconfirmed
        assert "earlier run on this commit may be out of date" in unconfirmed
        failure = format_failure_comment("boom", no_status=True)
        assert "can't merge until DiffCost runs successfully" in failure
        assert "earlier run posted on this commit still applies" in failure
        assert "earlier run" not in format_failure_comment("boom")

    def test_service_problem_hides_free_upsell(self):
        body = self._body([self._unknown()], plan="free", service_problem=True)
        assert "DiffCost Free" not in body
        assert "DiffCost Free" in self._body([self._unknown()], plan="free")

    def test_no_changes_comment_carries_the_note(self):
        from comment import format_clean_comment, format_no_changes_comment, status_note

        note = status_note("failed")
        assert note in format_no_changes_comment(note)
        assert note in format_clean_comment(note)
        assert "status was posted" not in format_no_changes_comment()

    @patch("comment.requests")
    def test_permission_hint_mentions_fork_prs(self, mock_requests, capsys):
        r = MagicMock(status_code=403)
        mock_requests.get.return_value = r
        assert post_or_update_comment("o/r", 1, "t", "body") is None
        out = capsys.readouterr().out
        assert "pull-requests: write" in out and "forks" in out and "job summary" in out


class TestBodyIfNew:
    @patch("comment.requests")
    def test_body_if_new_used_only_when_creating(self, mock_requests):
        from comment import post_or_update_comment

        listing = MagicMock(status_code=200, links={})
        listing.json.return_value = []
        created = MagicMock(status_code=201, content=b"{}")
        created.json.return_value = {"html_url": "u"}
        mock_requests.get.return_value = listing
        mock_requests.post.return_value = created
        post_or_update_comment("o/r", 1, "t", "old wording", create=True, body_if_new="new wording")
        assert mock_requests.post.call_args.kwargs["json"]["body"] == "new wording"


class TestRenderingOnRealRuns:
    """How messages, BigQuery errors and dollar figures render in the comment on GitHub."""

    def _row(self, name="q", before=1.0, after=2.0, **dr):
        dry = {"cost_before": before, "cost_after": after, "bytes_before": 10, "bytes_after": 20, "error": None}
        dry.update(dr)
        return {"name": name, "path": f"{name}.sql", "dry_run": dry, "structural_findings": []}

    def _body(self, results, **kw):
        args = dict(total_before=1.0, total_after=2.0, total_delta=1.0, has_bq_creds=True,
                    blocked=False, block_reasons=[], plan="team")
        args.update(kw)
        return format_comment(results=results, **args)

    def test_failure_message_keeps_its_code_spans(self):
        from comment import format_failure_comment
        body = format_failure_comment(
            "The pull request's base commit `0f23ea1bb9cd` is not in the checkout. "
            "Set `fetch-depth: 0` on `actions/checkout`."
        )
        assert "\\`" not in body
        assert "base commit `0f23ea1bb9cd` is not" in body
        assert "Set `fetch-depth: 0` on `actions/checkout`." in body

    def test_rich_text_escapes_everything_outside_code_spans(self):
        from comment import md_rich
        out = md_rich("Could not diff against `main`: **boom** @octocat https://evil.example <b>x</b>")
        assert out.startswith("Could not diff against `main`: ")
        assert "**boom**" not in out and "@octocat" not in out and "https://" not in out and "<b>" not in out

    def test_rich_text_backtick_in_a_value_cannot_break_out(self):
        from comment import md_rich
        # An untrusted ref closes DiffCost's span early: the rest is still escaped text
        ref = "x` **bold** @octocat [l](http://e) `y"
        out = md_rich(f"Could not diff against `{ref}`: failed")
        assert "**bold**" not in out and "@octocat" not in out and "[l](" not in out
        assert out.startswith("Could not diff against `x` ")

    def test_rich_text_touching_spans_are_escaped_whole(self):
        from comment import md_rich, md_text
        # "`a`" + "```` b``` ````" would merge into a 5-backtick run and leave `a` unclosed,
        # so "**@octocat**" would render as bold text and a mention
        ref = "**@octocat**``b``` "
        message = f"Could not diff against `{ref}`"
        assert md_rich(message) == md_text(message)
        assert "@octocat" not in md_rich(message)

    def test_rich_text_in_tables_escapes_pipes_inside_spans(self):
        from comment import md_rich
        assert "|" not in md_rich("`a|b` c|d").replace("\\|", "")

    def test_policy_warning_keeps_its_code_spans(self):
        body = self._body([self._row()], policy_warnings=[".diffcost.yml: `thresholds` must be a mapping."])
        assert "`thresholds` must be a mapping" in body and "\\`thresholds" not in body

    def test_whole_bigquery_error_fits_in_its_row(self):
        reason = ("PR dry-run failed: Access Denied: Table <redacted>: User does not have permission to "
                  "query table <redacted>, or perhaps it does not exist.")
        body = self._body([self._row(error=reason)])
        assert "or perhaps it does not exist" in body

    def test_sub_cent_cost_is_not_shown_as_zero(self):
        from comment import _format_cost
        assert _format_cost(0) == "$0.00"
        assert _format_cost(0.00045) == "<$0.001"
        assert _format_cost(-0.00045) == "-<$0.001"
        assert _format_cost(0.071) == "$0.071"
        body = self._body([self._row(name="old", before=0.00045, after=None, bytes_after=None)],
                          total_before=0.00045, total_after=0.0, total_delta=-0.00045)
        assert "(was <$0.001)" in body and "(was $0.00)" not in body

    def test_not_enforced_reason_not_repeated_when_status_note_says_it(self):
        from comment import NOT_ENFORCED_REASONS, status_note
        reason = NOT_ENFORCED_REASONS["unreachable"]
        body = self._body([self._row()], blocked=True, block_reasons=["q: +$14.82/run exceeds"],
                          not_enforced_reason=reason, status_note_line=status_note("unreachable"))
        assert "⚠️ **Cost policy exceeded — not enforced**:" in body
        assert body.count("could not be reached") == 1

    def test_not_enforced_reason_kept_without_status_note(self):
        # Free plan: the (success) status was posted, so there is no status note to give the reason
        body = self._body([self._row()], blocked=True, block_reasons=["q: +$14.82/run exceeds"],
                          not_enforced_reason="merge blocking is off on the Free plan")
        assert "not enforced** (merge blocking is off on the Free plan):" in body

    def test_usd_change_shows_cents_unless_that_hides_the_excess(self):
        from comment import format_usd_change
        assert format_usd_change(14.8158, 1.0) == "$14.82"
        assert format_usd_change(14.8158) == "$14.82"
        assert format_usd_change(1234.5) == "$1,234.50"
        assert format_usd_change(1.004, 1.0) == "$1.004"      # "$1.00 exceeds $1.00" would be wrong
        assert format_usd_change(5.003, 5.0025) == "$5.003"

    def test_usd_change_under_a_dollar_matches_the_table(self):
        """The reason and the status show the table's figure: +$0.010, never +$0.0095."""
        from comment import _format_cost, format_usd_change
        for value in (0.0096, 0.0712, 0.336, 0.0004):
            assert format_usd_change(value) == _format_cost(value)
        assert format_usd_change(0.0096, 0.005) == "$0.010"
        assert format_usd_change(0.0712, 0.05) == "$0.071"
        # rounding would hide the excess: keep the digits that show it
        assert format_usd_change(0.0052, 0.005) == "$0.0052"
        assert format_usd_change(0.0008, 0.0005) == "$0.0008"   # not "<$0.001 exceeds $0.0005"
        assert format_usd_change(0.00003, 0.0) == "<$0.001"     # above $0, as the table shows it
        assert format_usd_change(0.0, None) == "$0.00"

    def test_usd_change_never_reads_as_equal_to_its_limit(self):
        """`+$5.00 exceeds $5.00` would be wrong: add decimals (up to 8) until the figure reads above."""
        from comment import format_usd_change
        assert format_usd_change(5.00004, 5.0) == "$5.00004"
        assert format_usd_change(0.00053, 0.0005) == "$0.00053"
        assert format_usd_change(1.00000001, 1.0) == "$1.00000001"


class TestNoKeyAndSlackLines:
    """The no-API-key footer and failed Slack alerts."""

    def _body(self, **kwargs):
        args = dict(results=[], total_before=0.0, total_after=0.0, total_delta=0.0, has_bq_creds=True,
                    blocked=False, block_reasons=[], plan="free")
        args.update(kwargs)
        return format_comment(**args)

    def test_no_key_causes(self):
        from comment import no_api_key_cause

        assert "forks don't get repository secrets" in no_api_key_cause(True, False)
        assert "only get Dependabot secrets" in no_api_key_cause(False, True)
        assert "`DIFFCOST_API_KEY` secret exists" in no_api_key_cause(False, False)
        assert "forks" in no_api_key_cause(True, True)  # a fork PR gets no secrets at all

    def test_no_key_line_replaces_the_free_upsell(self):
        from comment import no_api_key_cause

        body = self._body(no_key_cause=no_api_key_cause(False, False))
        assert "🔑 **No DiffCost API key**, so this run can't block merges." in body
        assert "The `api_key` input is empty: check that the `DIFFCOST_API_KEY` secret exists" in body
        assert "(https://diffcost.dev/docs/troubleshooting#no-diffcost-api-key)" in body
        assert "DiffCost Free" not in body

    def test_free_plan_with_a_key_keeps_the_upsell(self):
        body = self._body()
        assert "block expensive PRs" in body
        assert "No DiffCost API key" not in body

    def test_no_key_line_on_the_clean_and_no_changes_comments(self):
        from comment import format_clean_comment, format_no_changes_comment

        assert "No DiffCost API key" in format_clean_comment("", "a cause")
        assert "No DiffCost API key" in format_no_changes_comment("", "a cause")
        assert "No DiffCost API key" not in format_clean_comment("")

    def test_slack_failure_lines_are_escaped_and_near_the_top(self):
        body = self._body(blocked=True, block_reasons=["x"], slack_failures=[
            (".diffcost.yml webhook", "Slack returned HTTP 404: no_service"),
            ("dashboard threshold webhook", "<b>@team</b> https://x.example"),
        ])
        assert ("⚠️ **Slack alert failed** (.diffcost.yml webhook): Slack returned HTTP 404: no\\_service. "
                "[What to check](https://diffcost.dev/docs/troubleshooting#slack-alerts-dont-arrive)") in body
        assert "**Slack alert failed** (dashboard threshold webhook): \\<b\\>@⁠team" in body
        assert "https://x.example" not in body  # untrusted text never becomes a link
        assert body.index("Slack alert failed") < body.index("| ")  # before the cost table


def test_format_pct_change_matches_the_backend_rule():
    """`+200% exceeds the 200% threshold` would be wrong: show the decimals that read above the
    limit, the same way the backend's `pct_change` does for the commit status and Slack."""
    from comment import format_pct_change
    assert format_pct_change(1900.0, 50.0) == "1900"
    assert format_pct_change(200.4, 200.0) == "200.4"
    assert format_pct_change(200.04, 200.0) == "200.04"
    assert format_pct_change(12.6, 12.5) == "13"
    assert format_pct_change(33.333) == "33"
    assert format_pct_change(200.00001, 200.0) == "200.00001"  # never "200.0000 exceeds 200"


class TestPlanAndPolicyLines:
    """Plan lines per plan state, the policy source, `suspended`."""

    def _body(self, **kwargs):
        results = [{
            "name": "q", "path": "q.sql", "structural_findings": [],
            "dry_run": {"cost_before": 1.0, "cost_after": 2.0, "error": None},
        }]
        args = dict(results=results, total_before=1.0, total_after=2.0, total_delta=1.0, has_bq_creds=True,
                    blocked=False, block_reasons=[], plan="free")
        args.update(kwargs)
        return format_comment(**args)

    def test_unpaid_stripe_says_update_the_payment_method(self):
        body = self._body(subscription_inactive=True)
        assert (
            "⚠️ **Your Stripe subscription isn't active.** Merge blocking, Slack alerts and cost trends are off; "
            "[update the payment method](https://diffcost.dev/dashboard/billing) to turn them back on."
        ) in body
        assert "DiffCost Free" not in body and "Upgrade" not in body

    def test_paid_plan_without_a_threshold_says_so(self):
        body = self._body(plan="team", threshold_source="none")
        assert (
            "🛡️ **DiffCost Team** — Merge blocking enabled, but no threshold applies to this repository "
            "([set one](https://diffcost.dev/docs/thresholds#where-to-set-thresholds)). "
            "[Manage subscription](https://diffcost.dev/dashboard/billing)"
        ) in body
        assert "but no threshold applies" not in self._body(plan="team", threshold_source="org")
        # No status posted: the footer never claims merge blocking either way
        assert "Merge blocking enabled" not in self._body(
            plan="starter", threshold_source="none", merge_blocking_on=False
        )

    @pytest.mark.parametrize("source, text", [
        ("repo", "The policy that applied: this repository's dashboard threshold."),
        ("org", "The policy that applied: the org-wide dashboard threshold (`*`)."),
        ("yml", "The policy that applied: `.diffcost.yml` on the base branch."),
    ])
    def test_blocked_verdict_names_the_policy(self, source, text):
        body = self._body(plan="team", blocked=True, block_reasons=["q: +$1.00/run exceeds"], threshold_source=source)
        assert text in body
        assert body.index("🚫 **Merge blocked**") < body.index(text) < body.index("| Query / Model")

    def test_policy_line_is_its_own_paragraph(self):
        """Not a lazy continuation of the last reason bullet (CommonMark)."""
        body = self._body(plan="team", blocked=True, block_reasons=["a", "b"], threshold_source="org")
        assert "- b\n\nThe policy that applied:" in body

    def test_trial_without_a_threshold_says_so(self):
        hint = ("Merge blocking is on, but no threshold applies to this repository "
                "([set one](https://diffcost.dev/docs/thresholds#where-to-set-thresholds)).")
        body = self._body(plan="team", trial_days_remaining=9, threshold_source="none")
        assert f"⏳ **DiffCost Team trial:** 9 days remaining. {hint} [Subscribe]" in body
        assert hint not in self._body(plan="team", trial_days_remaining=9, threshold_source="org")
        assert hint not in self._body(plan="team", trial_days_remaining=9, threshold_source="none",
                                      merge_blocking_on=False)

    def test_runner_checked_yml_says_so(self):
        body = self._body(plan="team", blocked=True, block_reasons=["x"], threshold_source="yml_local")
        assert ("The policy that applied: `.diffcost.yml` on the base branch, checked on the runner "
                "(the dashboard settings weren't fetched).") in body

    def test_unknown_source_names_nothing(self):
        for source in ("", "none", "plan"):
            body = self._body(plan="team", blocked=True, block_reasons=["x"], threshold_source=source)
            assert "The policy that applied" not in body
        assert "The policy that applied" not in self._body(plan="team", threshold_source="org")

    def test_suspended_app_note(self):
        from comment import NOT_ENFORCED_REASONS, status_note

        assert NOT_ENFORCED_REASONS["suspended"] == "the DiffCost GitHub App is suspended in this organization"
        note = status_note("suspended")
        assert "**No `diffcost/cost-check` status was posted** (the DiffCost GitHub App is suspended" in note
        assert (
            "A GitHub organization owner can unsuspend it in the organization's **Settings → GitHub Apps → DiffCost**, "
            "then re-run this job."
        ) in note

    def test_reasons_dont_repeat_the_note(self):
        from comment import NOT_ENFORCED_REASONS, status_note

        for key, reason in NOT_ENFORCED_REASONS.items():
            assert "no commit status was posted" not in reason, key
            assert "could not be confirmed" not in reason, key
        assert "(the DiffCost API key was rejected)." in status_note("rejected")
        assert "(the DiffCost service did not answer in time)." in status_note("timeout")


class TestAccuracyMarkers:
    """BigQuery's own accuracy flag: "≤" on upper-bound figures, with a note saying why."""

    def _row(self, name="q", before=1.0, after=2.0, findings=None, **dr):
        dry = {"cost_before": before, "cost_after": after, "bytes_before": 10 * 2**30, "bytes_after": 20 * 2**30,
               "error": None}
        dry.update(dr)
        return {"name": name, "path": f"{name}.sql", "dry_run": dry, "structural_findings": findings or []}

    def _body(self, results, **kw):
        args = dict(total_before=1.0, total_after=2.0, total_delta=1.0, has_bq_creds=True,
                    blocked=False, block_reasons=[], plan="team")
        args.update(kw)
        return format_comment(results=results, **args)

    def test_upper_bound_figures_get_a_marker_and_a_note(self):
        body = self._body([self._row(upper_bound_before=True, upper_bound_after=True)])
        assert "| `q` | ≤$1.00 | ≤$2.00 |" in body
        assert UPPER_BOUND_NOTE in body

    def test_the_note_says_the_change_is_no_bound(self):
        body = self._body([self._row(upper_bound_after=True)])
        assert "| `q` | $1.00 | ≤$2.00 |" in body
        assert "A Cost Change isn't a bound" in body and "the real increase can be larger" in body
        assert "analytics" not in body  # the note names no tables

    def test_new_and_removed_figures_keep_the_marker(self):
        new = self._row("n", before=None, upper_bound_after=True, bytes_before=None)
        new["cost"] = classify_cost(new["dry_run"], has_production=False, has_pr=True)
        removed = self._row("r", after=None, upper_bound_before=True, bytes_after=None)
        removed["cost"] = classify_cost(removed["dry_run"], has_production=True, has_pr=False)
        body = self._body([new, removed])
        assert "| `n` | — | ≤$2.00 | +≤20.00 GiB | **New** (+≤$2.00) |" in body
        assert "| `r` | ≤$1.00 | — | -≤10.00 GiB | **Removed** (was ≤$1.00) |" in body

    def test_precise_figures_have_no_marker(self):
        body = self._body([self._row(upper_bound_before=False, upper_bound_after=False)])
        assert "≤" not in body

    def test_tiny_figures_keep_their_less_than_sign(self):
        body = self._body([self._row(before=0.001, after=0.002, upper_bound_after=True)])
        assert "≤<" not in body

    def test_totals_and_editions_bytes_are_marked(self):
        rows = [self._row("a", upper_bound_after=True), self._row("b")]
        body = self._body(rows, total_before=2.0, total_after=4.0, total_delta=2.0)
        assert "| **TOTAL** | **$2.00** | **≤$4.00** |" in body
        editions = self._body([self._row(upper_bound_after=True)], bq_pricing_model="editions")
        assert "≤20.00 GiB" in editions

    def test_high_scan_warning_names_the_query_and_confirmed_partition_tables(self):
        from analysis.structural import Finding

        f = Finding("Partition filter removed", "high", "event_date is the partition column of analytics.events.",
                    partition_tables=("analytics.events (4.00 TiB table, partitioned daily)",))
        body = self._body([self._row(findings=[f])])
        assert ("> ⚠️ **High Scan Warning:** `q` no longer filters on the partition column of "
                "analytics.events (4.00 TiB table, partitioned daily), so it reads every partition.") in body
        assert "Query scans full table without partition filter" not in body

    def test_high_scan_warning_names_several_queries(self):
        from analysis.structural import Finding

        def finding():
            return Finding("Partition filter removed", "high", "event_date is the partition column of analytics.events.",
                           partition_tables=("analytics.events (partitioned daily)",))
        body = self._body([self._row("a", findings=[finding()]), self._row("b", findings=[finding()])],
                          total_before=2.0, total_after=4.0, total_delta=2.0)
        assert ("`a`, `b` no longer filter on the partition column of analytics.events (partitioned daily), "
                "so they read every partition.") in body

    def test_high_scan_warning_stays_generic_for_a_guess(self):
        from analysis.structural import Finding

        f = Finding("Partition filter removed", "high", "Removing a partition filter causes a full table scan.")
        assert "Query scans full table without partition filter." in self._body([self._row(findings=[f])])

    def test_no_high_scan_warning_when_another_partition_filter_remains(self):
        """The PR still filters on the partition column: more partitions, not a full scan."""
        from analysis.structural import Finding

        confirmed = Finding("Partition filter removed", "high", "event_date is the partition column of analytics.events.",
                            partition_tables=("analytics.events (4.00 TiB, daily)",), partition_filter_kept=True)
        guessed = Finding("Partition filter removed", "high", "The query still has another filter on this column.",
                          partition_filter_kept=True)
        body = self._body([self._row(findings=[confirmed]), self._row("r", findings=[guessed])])
        assert "High Scan Warning" not in body
        assert "Partition filter removed" in body  # still listed in the root cause analysis

    def test_not_costed_row_keeps_the_mark_on_its_base_figure(self):
        """The base figure of a Not costed row is shown but not summed: no mark on the totals."""
        cost = classify_cost({"cost_before": 1.0, "bytes_before": 2**30, "error": None, "pr_depends_on_new_objects": True},
                             has_production=True, has_pr=True)
        row = {"name": "m", "path": "models/m.sql", "structural_findings": [], "cost": cost,
               "dry_run": {"cost_before": 1.0, "bytes_before": 2**30, "error": None, "pr_depends_on_new_objects": True,
                           "upper_bound_before": True}}
        body = self._body([row, self._row("a"), self._row("b")], total_before=2.0, total_after=4.0, total_delta=2.0,
                          items_not_costed=1)
        assert "| `m` | ≤$1.00 | — | — | ⏳ Not costed" in body
        assert "| **TOTAL** | **$2.00** | **$4.00** |" in body
        assert UPPER_BOUND_NOTE in body

    def test_exempt_hint_lists_files_bigquery_cannot_estimate(self):
        row = {"name": "orders", "path": "models/orders.sql", "structural_findings": [],
               "exempt_entry": "models/orders.sql",
               "dry_run": {"error": None, "unknown_reason": "the PR version reads a table with row-level security"},
               "cost": classify_cost({"error": None, "unknown_reason": "the PR version reads a table with row-level security"})}
        body = self._body([row], items_unknown=1)
        assert "Cost unavailable: the PR version reads a table with row-level security" in body
        assert "BigQuery can't estimate some queries before they run" in body
        assert "or an estimate of unknown accuracy" in body and "in a separate PR" in body
        assert '`ignored_paths: ["models/orders.sql"]`' in body

    def test_exempt_hint_lists_every_file(self):
        reason = "the PR version reads a table with row-level security"
        rows = [
            {"name": f"m{i}", "path": f"models/m{i}.sql", "structural_findings": [], "exempt_entry": f"models/m{i}.sql",
             "dry_run": {"error": None, "unknown_reason": reason}, "cost": classify_cost({"error": None, "unknown_reason": reason})}
            for i in range(5)
        ]
        body = self._body(rows, items_unknown=5)
        assert "ignored_paths: [" + ", ".join(f'"models/m{i}.sql"' for i in range(5)) + "]" in body
        assert "more files" not in body


class TestQueryParameterLines:
    def test_priced_with_line_escapes_values_and_caps_the_list(self):
        from comment import query_parameters_line
        line = query_parameters_line([("s", '"a`b"')] + [(f"p{i}", str(i)) for i in range(11)])
        assert line.startswith('ℹ️ Plain SQL files were priced with `` @s = "a`b" ``, `@p0 = 0`, ')
        assert line.endswith("`@p8 = 8` and 2 more (`query_parameters` in `.diffcost.yml`).")

    def test_hint_example_uses_supported_types_only(self):
        from comment import _parameters_hint
        hint = _parameters_hint([
            {"unbound_parameters": [("ids", "ARRAY"), ("when", "TIMESTAMP")]},
            {"unbound_parameters": [("WHEN", ""), ("label", "")]},
        ])
        assert "(`@ids ARRAY`, `@when TIMESTAMP`, `@label`)" in hint
        assert "for example: `query_parameters: {when: {type: TIMESTAMP, days_ago: 1}}`" in hint

    def test_hint_without_a_known_type_has_no_example(self):
        from comment import _parameters_hint
        hint = _parameters_hint([{"unbound_parameters": [("label", "")]}])
        assert "(`@label`)" in hint and "for example" not in hint
        assert _parameters_hint([{"unbound_parameters": []}, {}]) == ""

    def test_hint_with_only_parameters_that_cant_be_set(self):
        from comment import _parameters_hint
        hint = _parameters_hint([{"unbound_parameters": [("?", "INT64")], "parameter_exempt_entry": "q/a.sql"}])
        assert hint.startswith("> ℹ️ **Some queries use query parameters that have no value**, so they count")
        assert "Give each" not in hint and '`ignored_paths: ["q/a.sql"]`' in hint
