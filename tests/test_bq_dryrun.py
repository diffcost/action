"""Tests for BigQuery dry-run wrapper (unit tests with mocked client)."""

import pytest
from unittest.mock import MagicMock, patch

from analysis.accuracy import assess
from analysis.bq_dryrun import dry_run, dry_run_pair, _bytes_to_cost


TIB = 2**40
GIB = 2**30


class TestBytesToCost:
    """BigQuery bills on-demand queries per TiB (2**40 bytes), $6.25/TiB."""

    def test_one_tib(self):
        assert _bytes_to_cost(TIB, 6.25) == pytest.approx(6.25)

    def test_one_decimal_tb_is_less_than_one_tib(self):
        # 1e12 bytes is 0.909 TiB — the old 1e12 divisor overstated cost by ~10%
        assert _bytes_to_cost(1_000_000_000_000, 6.25) == pytest.approx(6.25 * 1e12 / TIB)

    def test_one_gib(self):
        assert _bytes_to_cost(GIB, 6.25) == pytest.approx(6.25 / 1024)

    def test_zero_bytes(self):
        assert _bytes_to_cost(0, 6.25) == 0.0

    def test_custom_rate(self):
        assert _bytes_to_cost(TIB, 5.00) == pytest.approx(5.00)


class TestDryRun:
    def test_successful_dry_run(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.total_bytes_processed = 5_000_000_000  # 5 GB
        mock_job.total_bytes_billed = 5_000_000_000
        mock_table = MagicMock()
        mock_table.project = "my-project"
        mock_table.dataset_id = "my_dataset"
        mock_table.table_id = "orders"
        mock_job.referenced_tables = [mock_table]
        mock_client.query.return_value = mock_job

        result = dry_run(mock_client, "SELECT * FROM orders")

        assert result["bytes_processed"] == 5_000_000_000
        assert result["error"] is None
        assert "my-project.my_dataset.orders" in result["referenced_tables"]
        mock_client.query.assert_called_once()

    def test_dry_run_error(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("Syntax error: Unexpected identifier at [1:15]")

        result = dry_run(mock_client, "SELECT * FROM nonexistent_table")

        assert result["error"] is not None
        assert result["bytes_processed"] == 0

    def test_dry_run_unmaterialized_upstream(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("Not found: Table my_project.my_dataset.new_table was not found")

        result = dry_run(mock_client, "SELECT * FROM new_table")

        assert result["unmaterialized_upstream"] is True
        assert result["bytes_processed"] is None
        assert result["error"] is None

    def test_dataset_not_found_is_unmaterialized(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("404 Not found: Dataset my-project:ci_schema was not found")
        assert dry_run(mock_client, "SELECT 1 FROM ci_schema.t")["unmaterialized_upstream"] is True

    def test_unrecognized_column_is_a_real_error(self):
        """A column typo must not be treated as an unmaterialized relation (it would hide the cost)."""
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception(
            "400 POST https://bigquery.googleapis.com/bigquery/v2/projects/p/jobs: Unrecognized name: amout at [1:8]"
        )
        result = dry_run(mock_client, "SELECT amout FROM t")
        assert result["unmaterialized_upstream"] is False
        assert result["error"] == "Unrecognized name: amout at [1:8]"

    def test_generic_404_is_a_real_error(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("404 Not found: Job my-project:US.abc")
        assert dry_run(mock_client, "SELECT 1")["unmaterialized_upstream"] is False

    def test_error_redacts_qualified_names(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("Access Denied: Table secret-proj:finance.payroll: no permission")
        result = dry_run(mock_client, "SELECT 1")
        assert "payroll" not in result["error"]
        assert "finance" not in result["error"]

    def test_dry_run_sets_timeout_and_retry_deadline(self):
        mock_client = MagicMock()
        mock_client.query.return_value = MagicMock(total_bytes_processed=1, total_bytes_billed=1, referenced_tables=[])
        dry_run(mock_client, "SELECT 1")
        kwargs = mock_client.query.call_args.kwargs
        assert kwargs["timeout"] > 0
        assert kwargs["retry"] is not None

    def test_empty_sql(self):
        mock_client = MagicMock()
        result = dry_run(mock_client, "")
        assert result["error"] == "Empty SQL"
        mock_client.query.assert_not_called()

    def test_whitespace_only_sql(self):
        mock_client = MagicMock()
        result = dry_run(mock_client, "   \n\t  ")
        assert result["error"] == "Empty SQL"


class TestDryRunPair:
    def _make_client(self, prod_bytes: int, pr_bytes: int):
        """Create a mock client that returns different bytes for sequential calls."""
        mock_client = MagicMock()
        calls = iter([prod_bytes, pr_bytes])

        def mock_query(sql, job_config=None, **kwargs):
            job = MagicMock()
            job.total_bytes_processed = next(calls)
            job.total_bytes_billed = job.total_bytes_processed
            job.referenced_tables = []
            return job

        mock_client.query.side_effect = mock_query
        return mock_client

    def test_cost_regression(self):
        client = self._make_client(GIB, 10 * GIB)  # 1 GiB → 10 GiB
        result = dry_run_pair(client, "SELECT a FROM t", "SELECT * FROM t")

        assert result["cost_before"] == pytest.approx(6.25 / 1024)
        assert result["cost_after"] == pytest.approx(62.5 / 1024)
        assert result["error"] is None

    def test_cost_optimization(self):
        client = self._make_client(10_000_000_000, 1_000_000_000)  # 10 GB → 1 GB
        result = dry_run_pair(client, "SELECT * FROM t", "SELECT id FROM t")

        assert result["cost_after"] == pytest.approx(result["cost_before"] / 10)

    def test_no_change(self):
        client = self._make_client(5_000_000_000, 5_000_000_000)
        result = dry_run_pair(client, "SELECT id FROM t", "SELECT id FROM t")

        assert result["cost_before"] is not None and result["cost_after"] == result["cost_before"]

    def test_new_query_no_production(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.total_bytes_processed = 5_000_000_000
        mock_job.total_bytes_billed = 5_000_000_000
        mock_job.referenced_tables = []
        mock_client.query.return_value = mock_job

        result = dry_run_pair(mock_client, None, "SELECT * FROM new_table")

        assert result["cost_before"] is None
        assert result["cost_after"] == pytest.approx(5_000_000_000 / TIB * 6.25)

    def test_deleted_query_no_pr(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.total_bytes_processed = 5_000_000_000
        mock_job.total_bytes_billed = 5_000_000_000
        mock_job.referenced_tables = []
        mock_client.query.return_value = mock_job

        result = dry_run_pair(mock_client, "SELECT * FROM old_table", None)

        assert result["cost_before"] == pytest.approx(5_000_000_000 / TIB * 6.25)
        assert result["cost_after"] is None

    def test_production_dry_run_fails(self):
        mock_client = MagicMock()

        call_count = [0]

        def mock_query(sql, job_config=None, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Permission denied")
            job = MagicMock()
            job.total_bytes_processed = 5_000_000_000
            job.total_bytes_billed = 5_000_000_000
            job.referenced_tables = []
            return job

        mock_client.query.side_effect = mock_query

        result = dry_run_pair(mock_client, "SELECT old FROM t", "SELECT new FROM t")

        assert "Production dry-run failed" in (result["error"] or "")
        assert result["cost_after"] is not None


class TestCreateBqClient:
    def test_no_credentials_and_no_adc_env(self, monkeypatch):
        from analysis.bq_dryrun import create_bq_client
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        mock_config = MagicMock()
        mock_config.gcp_credentials = ""
        mock_config.gcp_project = ""
        assert create_bq_client(mock_config) is None

    def test_adc_env_variable_fallback(self, monkeypatch):
        from analysis.bq_dryrun import create_bq_client
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/fake-adc-key.json")
        mock_config = MagicMock()
        mock_config.gcp_credentials = ""
        mock_config.gcp_project = "test-project"

        with patch("google.cloud.bigquery.Client") as mock_client:
            client = create_bq_client(mock_config)
            assert client is not None
            mock_client.assert_called_once_with(project="test-project")



class TestContainerCredentialsPath:
    def test_host_path_remapped_to_workspace_mount(self, tmp_path, monkeypatch):
        from analysis.bq_dryrun import _container_credentials_path

        (tmp_path / "gha-creds-abc.json").write_text("{}")
        monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
        host_path = "/home/runner/work/repo/repo/gha-creds-abc.json"
        assert _container_credentials_path(host_path) == str(tmp_path / "gha-creds-abc.json")

    def test_existing_path_unchanged(self, tmp_path):
        from analysis.bq_dryrun import _container_credentials_path

        p = tmp_path / "key.json"
        p.write_text("{}")
        assert _container_credentials_path(str(p)) == str(p)


class TestPrSideMissingDependency:
    """A PR-side "Unrecognized name" / "Not found: Table" usually means the query reads an
    object another model in the same PR creates (prod doesn't have it yet)."""

    def _client(self, prod_bytes, pr_error):
        client = MagicMock()

        def query(sql, job_config=None, **kwargs):
            if sql == "PR":
                raise Exception(pr_error)
            job = MagicMock()
            job.total_bytes_processed = prod_bytes
            job.total_bytes_billed = prod_bytes
            job.referenced_tables = []
            return job

        client.query.side_effect = query
        return client

    def test_unrecognized_name_on_pr_side_with_base_ok(self):
        r = dry_run_pair(self._client(GIB, "400 Unrecognized name: discount_pct at [3:5]"), "BASE", "PR")
        assert r["pr_depends_on_new_objects"] is True
        assert r["error"] is None
        assert r["cost_before"] == pytest.approx(6.25 / 1024)

    def test_table_not_found_on_pr_side_of_new_model(self):
        r = dry_run_pair(self._client(GIB, "404 Not found: Table p:ds.stg_new was not found"), None, "PR")
        assert r["pr_depends_on_new_objects"] is True
        assert r["unmaterialized_upstream"] is False

    def test_unrecognized_name_on_pr_side_of_new_model_keeps_the_error(self):
        """A new model with a column typo: the caller decides (from what changed upstream in
        this PR) whether it is exempt, and shows the error when it isn't."""
        r = dry_run_pair(self._client(
            GIB, "400 POST https://bigquery.googleapis.com/bigquery/v2/projects/p/jobs: Unrecognized name: col_typo at [1:8]"
        ), None, "PR")
        assert r["pr_depends_on_new_objects"] is True
        assert r["pr_error"] == "Unrecognized name: col_typo at [1:8]"

    def test_not_found_keeps_the_missing_object_in_memory_only(self):
        r = dry_run_pair(self._client(GIB, "404 Not found: Table my-project:ds.stg_new was not found in location US"),
                         "BASE", "PR")
        assert r["pr_depends_on_new_objects"] is True
        assert r["pr_missing_object"] == {"kind": "table", "name": "my-project:ds.stg_new"}
        assert r["pr_error"] is None and r["error"] is None  # nothing that could reach the comment

    def test_not_found_without_a_readable_name(self):
        r = dry_run_pair(self._client(GIB, "404 Not found: Dataset"), None, "PR")
        assert r["pr_depends_on_new_objects"] is True
        assert r["pr_missing_object"] is None

    def test_other_pr_errors_stay_errors(self):
        r = dry_run_pair(self._client(GIB, "403 Access Denied: Table p:ds.t"), "BASE", "PR")
        assert r["pr_depends_on_new_objects"] is False
        assert "Access Denied" in r["error"]

    def test_base_side_failure_is_not_a_pr_dependency(self):
        client = MagicMock()
        client.query.side_effect = Exception("400 Unrecognized name: x")
        r = dry_run_pair(client, "BASE", "PR")
        assert r["pr_depends_on_new_objects"] is False
        assert r["error"]


class TestClientErrorRecorded:
    def test_bad_credentials_recorded_on_config(self):
        from analysis.bq_dryrun import create_bq_client

        config = MagicMock(gcp_credentials='{"type": "service_account"}', gcp_project="")
        config.bq_client_error = ""
        assert create_bq_client(config) is None
        assert config.bq_client_error

    def test_non_json_credentials_are_never_echoed(self, capsys, monkeypatch):
        """A base64-encoded key must never be echoed into the PR comment."""
        from analysis.bq_dryrun import create_bq_client

        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        secret = "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAiYWNtZS1wcm9kIgp9Cg=="
        config = MagicMock(gcp_credentials=secret, gcp_project="")
        config.bq_client_error = ""
        assert create_bq_client(config) is None
        assert "raw JSON key" in config.bq_client_error
        assert "base64" in config.bq_client_error
        assert secret[:20] not in config.bq_client_error
        assert secret[:20] not in capsys.readouterr().out

    def test_malformed_json_credentials_are_not_echoed(self, capsys, monkeypatch):
        from analysis.bq_dryrun import create_bq_client

        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        config = MagicMock(gcp_credentials='{"private_key": "-----BEGIN', gcp_project="")
        config.bq_client_error = ""
        assert create_bq_client(config) is None
        assert "BEGIN" not in config.bq_client_error
        assert "BEGIN" not in capsys.readouterr().out


class TestEstimateAccuracy:
    """BigQuery's accuracy flag and row-level security, read from the job's raw statistics."""

    def _job(self, num_bytes, accuracy=None, rls=False, statement_type="SELECT"):
        job = MagicMock(total_bytes_processed=None if rls else num_bytes, total_bytes_billed=None,
                        referenced_tables=[], statement_type=statement_type)
        stats = {"query": {"totalBytesProcessedAccuracy": accuracy} if accuracy else {}}
        if rls:
            stats["rowLevelSecurityStatistics"] = {"rowLevelSecurityApplied": True}
        job._properties = {"statistics": stats}
        return job

    def test_query_cache_is_off(self):
        client = MagicMock()
        client.query.return_value = self._job(1)
        dry_run(client, "SELECT 1")
        assert client.query.call_args.kwargs["job_config"].use_query_cache is False

    def test_accuracy_and_statement_type(self):
        client = MagicMock()
        client.query.return_value = self._job(5, "UPPER_BOUND", statement_type="SCRIPT")
        result = dry_run(client, "SELECT 1")
        assert result["accuracy"] == "UPPER_BOUND" and result["statement_type"] == "SCRIPT"
        assert result["row_level_security"] is False

    def test_mocked_job_without_statistics(self):
        client = MagicMock()
        client.query.return_value = MagicMock(total_bytes_processed=1, total_bytes_billed=1, referenced_tables=[])
        result = dry_run(client, "SELECT 1")
        assert result["accuracy"] is None and result["statement_type"] is None
        assert result["row_level_security"] is False

    def test_row_level_security_has_no_bytes(self):
        """Never $0.00: BigQuery returns no estimate for tables with row access policies."""
        client = MagicMock()
        client.query.return_value = self._job(None, rls=True)
        result = dry_run(client, "SELECT 1")
        assert result["row_level_security"] is True and result["bytes_processed"] is None

    def test_pair_reports_row_level_security(self):
        client = MagicMock()
        client.query.side_effect = [self._job(None, rls=True), self._job(None, rls=True)]
        result = dry_run_pair(client, "SELECT 1", "SELECT 2")
        assert result["rls_before"] is True and result["rls_after"] is True
        assert result["cost_before"] is None and result["cost_after"] is None
        assess(result, {})
        assert result["unknown_reason"].startswith("both versions read a table with row-level security")

    def test_pair_keeps_accuracy_per_side(self):
        client = MagicMock()
        client.query.side_effect = [self._job(1, "PRECISE"), self._job(2, "LOWER_BOUND")]
        result = dry_run_pair(client, "SELECT 1", "SELECT 2")
        assert result["accuracy_before"] == "PRECISE" and result["accuracy_after"] == "LOWER_BOUND"
        assert result["unknown_reason"] == ""
