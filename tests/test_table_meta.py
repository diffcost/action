"""Table metadata (tables.get), dry-run accuracy, and how both feed the analysis of one item."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.cloud import bigquery

from adapters import ChangedSQL
from analysis.accuracy import assess
from adapters.sql_files import _matches_glob, exempt_entry
from analysis.table_meta import META_CALL_TIMEOUT_SECONDS, TableMeta, TableMetaCache, fetch_table_meta, parse_table
from comment import classify_cost, format_comment
from config import ActionConfig
from main import DRY_RUN_BUDGET_SECONDS, _analyze_item, _build_payload, _table_info, summarize

TIB = 2**40
GIB = 2**30


def _table(table_id, type_="TABLE", partition=None, range_field=None, clustering=None, num_bytes=None):
    """A real google-cloud-bigquery Table built from a tables.get response."""
    project, dataset, table = table_id.split(".")
    resource = {
        "tableReference": {"projectId": project, "datasetId": dataset, "tableId": table},
        "type": type_,
        "location": "US",
        "schema": {"fields": [{"name": c, "type": "STRING"} for c in ("user_id", "event_date", "created_at", "a")]},
    }
    if partition:
        kind, field = partition
        resource["timePartitioning"] = {"type": kind, **({"field": field} if field else {})}
    if range_field:
        resource["rangePartitioning"] = {"field": range_field, "range": {"start": "0", "end": "100", "interval": "10"}}
    if clustering:
        resource["clustering"] = {"fields": clustering}
    if num_bytes is not None:
        resource["numBytes"] = str(num_bytes)
    return bigquery.Table.from_api_repr(resource)


class FakeJob:
    def __init__(self, num_bytes=0, tables=(), accuracy="PRECISE", statement_type="SELECT", rls=False):
        self.total_bytes_processed = None if rls else num_bytes
        self.total_bytes_billed = self.total_bytes_processed
        self.statement_type = statement_type
        self.referenced_tables = [
            SimpleNamespace(project=p, dataset_id=d, table_id=t) for p, d, t in (x.split(".") for x in tables)
        ]
        # accuracy=None: BigQuery left the field out
        stats = {"query": {} if rls or accuracy is None else {"totalBytesProcessedAccuracy": accuracy}}
        if rls:
            stats["rowLevelSecurityStatistics"] = {"rowLevelSecurityApplied": True}
        self._properties = {"statistics": stats}


class FakeClient:
    """`jobs`: SQL -> FakeJob, or an exception to raise; `tables`: table id -> Table, an
    exception to raise, or a callable."""

    def __init__(self, jobs, tables=None):
        self.jobs, self.tables = jobs, tables or {}
        self.get_calls: list[str] = []
        self.get_kwargs: list[dict] = []
        self.job_configs: list = []

    def query(self, sql, job_config=None, timeout=None, retry=None):
        self.job_configs.append(job_config)
        job = self.jobs[sql]
        if isinstance(job, Exception):
            raise job
        return job

    def get_table(self, table_id, timeout=None, retry=None):
        self.get_calls.append(table_id)
        self.get_kwargs.append({"timeout": timeout, "retry": retry})
        table = self.tables[table_id]
        if isinstance(table, Exception):
            raise table
        return table() if callable(table) else table


def _config() -> ActionConfig:
    return ActionConfig(
        sql_paths="queries/**/*.sql", manifest_path="", gcp_credentials="", api_key="", config_path=".diffcost.yml",
        base_ref="main", comment_on_clean=False, gcp_project="p", cost_per_tb=6.25,
    )


def _analyze(client, before, after, name="q", path=None, adapter="sql_files", cache=None, started_at=None):
    item = ChangedSQL(name=name, path=path or f"queries/{name}.sql", production_sql=before, pr_sql=after, adapter=adapter)
    started = time.monotonic() if started_at is None else started_at
    return _analyze_item(item, client, _config(), started, cache or TableMetaCache(client))


def _body(results, s, **kw):
    return format_comment(results=results, total_before=s["total_before"], total_after=s["total_after"],
                          total_delta=s["total_delta"], has_bq_creds=True, blocked=False, block_reasons=[], plan="team",
                          worst_query=s["worst"], **kw)


EVENTS = "p.analytics.events"
PRUNED = "SELECT user_id FROM `p.analytics.events` WHERE event_date >= '2024-01-01'"
FULL = "SELECT user_id FROM `p.analytics.events`"
# As google-api-core raises it: the status and request URL before BigQuery's reason
UNRECOGNIZED = ("400 POST https://bigquery.googleapis.com/bigquery/v2/projects/p/jobs?prettyPrint=false: "
                "Unrecognized name: user_id at [1:8]")


class TestParseTable:
    def test_day_partitioned_and_clustered(self):
        m = parse_table(EVENTS, _table(EVENTS, partition=("DAY", "event_date"), clustering=["user_id"],
                                       num_bytes=4 * TIB))
        assert m.partition_column == "event_date" and m.partition_kind == "DAY"
        assert m.partition_columns == frozenset({"event_date"})
        assert m.clustering_fields == ("user_id",)
        assert m.num_bytes == 4 * TIB and m.stored
        assert m.short_name == "analytics.events"
        assert m.label == "analytics.events (4.00 TiB table, partitioned daily)"

    def test_ingestion_time_partitioning(self):
        m = parse_table(EVENTS, _table(EVENTS, partition=("DAY", None), num_bytes=2 * GIB))
        assert m.ingestion_time
        assert m.partition_columns == frozenset({"_partitiontime", "_partitiondate"})
        assert m.label == "analytics.events (2.00 GiB table, partitioned by ingestion time, daily)"

    def test_integer_range_partitioning(self):
        m = parse_table(EVENTS, _table(EVENTS, range_field="customer_bucket"))
        assert m.partition_kind == "RANGE" and m.partition_columns == frozenset({"customer_bucket"})
        assert m.label == "analytics.events (partitioned by integer range)"

    def test_label_without_partitioning_or_size(self):
        assert parse_table(EVENTS, _table(EVENTS)).label == "analytics.events"

    def test_external_and_view_are_not_stored(self):
        assert parse_table("p.raw.sheet", _table("p.raw.sheet", type_="EXTERNAL")).external
        assert not parse_table("p.raw.sheet", _table("p.raw.sheet", type_="EXTERNAL")).stored
        assert not parse_table("p.d.v", _table("p.d.v", type_="VIEW")).stored

    def test_unexpected_values_are_ignored(self):
        """A mocked or unusual response never yields a wrong partition column."""
        m = parse_table(EVENTS, MagicMock())
        assert m == TableMeta(table_id=EVENTS)


class TestFetch:
    def test_reads_in_parallel_and_skips_failures(self):
        client = FakeClient({}, {
            EVENTS: _table(EVENTS, partition=("DAY", "event_date")),
            "p.d.gone": Exception("404 Not found"),
        })
        found = fetch_table_meta(client, [EVENTS, "p.d.gone", EVENTS])
        assert list(found) == [EVENTS]
        assert sorted(client.get_calls) == [EVENTS, "p.d.gone"]

    def test_slow_reads_are_cut_off(self):
        def slow():
            time.sleep(1.5)
            return _table(EVENTS)

        client = FakeClient({}, {EVENTS: slow})
        started = time.monotonic()
        assert fetch_table_meta(client, [EVENTS], timeout=0.2) == {}
        assert time.monotonic() - started < 1.0

    def test_cache_reads_each_table_once_per_run(self):
        client = FakeClient({}, {EVENTS: _table(EVENTS), "p.d.gone": Exception("boom")})
        cache = TableMetaCache(client)
        cache.get([EVENTS, "p.d.gone"])
        assert list(cache.get([EVENTS, "p.d.gone"])) == [EVENTS]
        assert sorted(client.get_calls) == [EVENTS, "p.d.gone"]
        assert cache.counts == (1, 2)

    def test_each_read_has_a_timeout_and_no_retry(self):
        """A read that hangs stops at its own timeout instead of retrying past the run."""
        client = FakeClient({}, {EVENTS: _table(EVENTS)})
        fetch_table_meta(client, [EVENTS])
        assert client.get_kwargs == [{"timeout": META_CALL_TIMEOUT_SECONDS, "retry": None}]
        assert META_CALL_TIMEOUT_SECONDS == 5

    def test_cache_stops_reading_when_the_budget_is_spent(self):
        client = FakeClient({}, {EVENTS: _table(EVENTS)})
        cache = TableMetaCache(client, budget_seconds=0)
        assert cache.get([EVENTS]) == {} and client.get_calls == []


class TestAssess:
    def _meta(self, table_id, **kw):
        return parse_table(table_id, _table(table_id, **kw))

    def test_external_lower_bound_is_unknown_and_named(self):
        dr = {"accuracy_after": "LOWER_BOUND", "tables_after": ["p.raw.sheet"]}
        assess(dr, {"p.raw.sheet": self._meta("p.raw.sheet", type_="EXTERNAL")})
        assert dr["unknown_reason"].startswith("the PR version reads the external table raw.sheet")
        assert "lower bound" in dr["unknown_reason"]

    def test_script_lower_bound(self):
        dr = {"accuracy_before": "LOWER_BOUND", "accuracy_after": "LOWER_BOUND",
              "statement_type_before": "SCRIPT", "statement_type_after": "SCRIPT"}
        assess(dr, {})
        assert "multi-statement script" in dr["unknown_reason"] and "both versions" in dr["unknown_reason"]

    def test_other_lower_bound(self):
        dr = {"accuracy_before": "LOWER_BOUND"}
        assess(dr, {})
        assert dr["unknown_reason"] == "the dry run of the base version gives only a lower bound"

    def test_row_level_security_reason_comes_first(self):
        dr = {"rls_before": True, "accuracy_after": "LOWER_BOUND", "bytes_after": 7, "cost_after": 0.2}
        assess(dr, {})
        assert dr["unknown_reason"] == (
            "the base version reads a table with row-level security, "
            "and BigQuery dry runs give no byte estimate for those tables"
        )

    def test_missing_accuracy_is_trusted(self):
        dr = {"bytes_before": 5, "cost_before": 0.1, "bytes_after": 7, "cost_after": 0.2, "tables_after": [EVENTS]}
        assess(dr, {EVENTS: self._meta(EVENTS)})
        assert not dr.get("unknown_reason") and dr["bytes_after"] == 7

    def test_missing_accuracy_on_an_external_table_is_a_lower_bound(self):
        dr = {"accuracy_before": "PRECISE", "bytes_before": 5, "cost_before": 0.1, "tables_before": [EVENTS],
              "bytes_after": 0, "cost_after": 0.0, "tables_after": [EVENTS, "p.raw.sheet"]}
        assess(dr, {EVENTS: self._meta(EVENTS), "p.raw.sheet": self._meta("p.raw.sheet", type_="EXTERNAL")})
        assert dr["unknown_reason"].startswith("the PR version reads the external table raw.sheet")
        assert dr["bytes_after"] is None and dr["cost_after"] is None and dr["bytes_before"] == 5

    def test_a_lower_bound_figure_is_dropped(self):
        """Only the version BigQuery can't estimate loses its figures."""
        dr = {"accuracy_before": "LOWER_BOUND", "accuracy_after": "PRECISE", "bytes_before": 5, "cost_before": 0.1,
              "bytes_after": 7, "cost_after": 0.2}
        assess(dr, {})
        assert dr["bytes_before"] is None and dr["cost_before"] is None
        assert dr["bytes_after"] == 7 and dr["cost_after"] == 0.2

    def test_an_upper_bound_figure_is_kept(self):
        dr = {"accuracy_before": "UPPER_BOUND", "accuracy_after": "UPPER_BOUND", "bytes_before": 5, "cost_before": 0.1,
              "bytes_after": 7, "cost_after": 0.2}
        assess(dr, {})
        assert (dr["bytes_before"], dr["cost_before"], dr["bytes_after"], dr["cost_after"]) == (5, 0.1, 7, 0.2)

    def test_upper_bound_is_marked_per_version(self):
        dr = {"accuracy_before": "PRECISE", "accuracy_after": "UPPER_BOUND"}
        assess(dr, {})
        assert dr["upper_bound_after"] and not dr["upper_bound_before"]
        assert not dr.get("unknown_reason")


class TestAnalyzeItem:
    def test_dry_runs_skip_the_query_cache(self):
        client = FakeClient({PRUNED: FakeJob(GIB, [EVENTS]), FULL: FakeJob(TIB, [EVENTS])},
                            {EVENTS: _table(EVENTS)})
        _analyze(client, PRUNED, FULL)
        assert client.job_configs and all(c.use_query_cache is False for c in client.job_configs)

    def test_removed_filter_on_the_real_partition_column_is_named(self):
        client = FakeClient({PRUNED: FakeJob(GIB, [EVENTS]), FULL: FakeJob(4 * TIB, [EVENTS])},
                            {EVENTS: _table(EVENTS, partition=("DAY", "event_date"), num_bytes=4 * TIB)})
        r = _analyze(client, PRUNED, FULL)
        high = [f for f in r["structural_findings"] if f.risk == "high"]
        assert high and high[0].description == "Partition filter removed"
        assert high[0].partition_tables == ("analytics.events (4.00 TiB table, partitioned daily)",)
        assert "event_date is the partition column of analytics.events" in high[0].detail
        body = format_comment(results=[r], total_before=0.0, total_after=1.0, total_delta=1.0, has_bq_creds=True,
                              blocked=False, block_reasons=[], plan="team")
        assert ("`q` no longer filters on the partition column of analytics.events "
                "(4.00 TiB table, partitioned daily), so it reads every partition.") in body

    def test_a_date_like_column_the_metadata_does_not_name_keeps_the_heuristic(self):
        before = "SELECT user_id FROM `p.analytics.events` WHERE created_at > '2024-01-01'"
        client = FakeClient({before: FakeJob(TIB, [EVENTS]), FULL: FakeJob(TIB, [EVENTS])},
                            {EVENTS: _table(EVENTS, partition=("DAY", "event_date"))})
        r = _analyze(client, before, FULL)
        partition = [f for f in r["structural_findings"] if f.description == "Partition filter removed"]
        assert len(partition) == 1 and partition[0].risk == "high" and not partition[0].partition_tables

    def test_partition_columns_come_from_the_base_versions_tables(self):
        """A removed filter comes from the base version: a table only the PR reads can't explain it."""
        sessions = "p.analytics.sessions"
        meta = {EVENTS: parse_table(EVENTS, _table(EVENTS, partition=("DAY", "event_date"), clustering=["user_id"])),
                sessions: parse_table(sessions, _table(sessions, partition=("DAY", "dt"), clustering=["sid"]))}
        info = _table_info({"tables_before": [EVENTS], "tables_after": [EVENTS, sessions]}, meta)
        assert info == {"partition_columns": {"event_date": ["analytics.events (partitioned daily)"]},
                        "cluster_columns": {"user_id": ["analytics.events"]}}

    def test_integer_range_partition_column_is_caught(self):
        before = "SELECT user_id FROM `p.analytics.events` WHERE customer_bucket = 3"
        client = FakeClient({before: FakeJob(GIB, [EVENTS]), FULL: FakeJob(TIB, [EVENTS])},
                            {EVENTS: _table(EVENTS, range_field="customer_bucket")})
        r = _analyze(client, before, FULL)
        assert any(f.partition_tables for f in r["structural_findings"])

    def test_without_metadata_the_name_heuristic_stays(self):
        before = "SELECT user_id FROM `p.analytics.events` WHERE created_at > '2024-01-01'"
        client = FakeClient({before: FakeJob(GIB, [EVENTS]), FULL: FakeJob(TIB, [EVENTS])},
                            {EVENTS: Exception("403 Access Denied")})
        r = _analyze(client, before, FULL)
        assert any(f.description == "Partition filter removed" and not f.partition_tables
                   for f in r["structural_findings"])

    def test_a_view_keeps_the_name_heuristic(self):
        """A view's partitioning lives in its base tables, which tables.get doesn't describe."""
        view = "p.analytics.events_v"
        before = "SELECT user_id FROM `p.analytics.events_v` WHERE created_at > '2024-01-01'"
        after = "SELECT user_id FROM `p.analytics.events_v`"
        client = FakeClient({before: FakeJob(GIB, [view]), after: FakeJob(TIB, [view])},
                            {view: _table(view, type_="VIEW")})
        r = _analyze(client, before, after)
        assert any(f.description == "Partition filter removed" for f in r["structural_findings"])

    def test_a_view_over_a_partitioned_table_keeps_the_name_heuristic(self):
        """The view renames the base table's partition column (event_date AS day): a removed
        filter on `day` stays a removed partition filter."""
        before = "SELECT user_id FROM `p.analytics.events_v` WHERE day = '2024-01-05'"
        after = "SELECT user_id FROM `p.analytics.events_v`"
        client = FakeClient(
            {before: FakeJob(GIB, [EVENTS]), after: FakeJob(TIB, [EVENTS])},
            {EVENTS: _table(EVENTS, partition=("DAY", "event_date"))},
        )
        r = _analyze(client, before, after)
        assert any(f.description == "Partition filter removed" and f.risk == "high"
                   for f in r["structural_findings"])

    def test_external_table_lower_bound_is_not_costed_and_can_be_exempted(self):
        before = "SELECT a FROM `p.raw.orders`"
        after = "SELECT a FROM `p.raw.orders` JOIN `p.raw.sheet` USING (a)"
        client = FakeClient(
            {before: FakeJob(GIB, ["p.raw.orders"]),
             after: FakeJob(GIB, ["p.raw.orders", "p.raw.sheet"], accuracy="LOWER_BOUND")},
            {"p.raw.orders": _table("p.raw.orders"), "p.raw.sheet": _table("p.raw.sheet", type_="EXTERNAL")},
        )
        r = _analyze(client, before, after, path="queries/[x]orders.sql")
        assert r["cost"]["status"] == "unknown"
        assert "external table raw.sheet" in r["cost"]["reason"]
        assert r["exempt_entry"] == "queries/[[]x]orders.sql"  # glob characters escaped
        assert summarize([r])["items_unknown"] == 1
        body = format_comment(results=[r], total_before=0.0, total_after=0.0, total_delta=0.0, has_bq_creds=True,
                              blocked=False, block_reasons=[], plan="team", items_unknown=1)
        assert "Cost unavailable: the PR version reads the external table raw.sheet" in body
        assert 'ignored_paths: ["queries/[[]x]orders.sql"]' in body

    def test_row_level_security_is_not_costed(self):
        """BigQuery gives no byte estimate for such tables; never shown as $0.00."""
        client = FakeClient({PRUNED: FakeJob(GIB, [EVENTS]), FULL: FakeJob(tables=[EVENTS], rls=True)},
                            {EVENTS: _table(EVENTS)})
        r = _analyze(client, PRUNED, FULL)
        assert r["cost"]["status"] == "unknown"
        assert "row-level security" in r["cost"]["reason"] and r["cost"]["reason"].startswith("the PR version")
        assert r["exempt_entry"] == "queries/q.sql"

    def test_script_lower_bound_is_not_costed(self):
        before = "CREATE TEMP TABLE t AS SELECT 1 AS a; SELECT * FROM t"
        after = "CREATE TEMP TABLE t AS SELECT 2 AS a; SELECT * FROM t"
        client = FakeClient({before: FakeJob(0, [], "LOWER_BOUND", "SCRIPT"),
                             after: FakeJob(0, [], "LOWER_BOUND", "SCRIPT")})
        r = _analyze(client, before, after)
        assert r["cost"]["status"] == "unknown" and "multi-statement script" in r["cost"]["reason"]

    def test_clustered_upper_bound_is_costed_with_a_marker(self):
        client = FakeClient(
            {PRUNED: FakeJob(GIB, [EVENTS], "UPPER_BOUND"), FULL: FakeJob(TIB, [EVENTS], "UPPER_BOUND")},
            {EVENTS: _table(EVENTS, partition=("DAY", "event_date"), clustering=["user_id"])},
        )
        r = _analyze(client, PRUNED, FULL)
        assert r["cost"]["status"] == "changed"
        assert r["dry_run"]["upper_bound_before"] and r["dry_run"]["upper_bound_after"]
        assert "exempt_entry" not in r

    def test_cte_alias_of_the_partition_column_keeps_the_heuristic(self):
        before = ("WITH e AS (SELECT event_date AS dt, user_id FROM `p.analytics.events`) "
                  "SELECT user_id FROM e WHERE dt >= '2024-01-01'")
        after = "WITH e AS (SELECT event_date AS dt, user_id FROM `p.analytics.events`) SELECT user_id FROM e"
        client = FakeClient({before: FakeJob(GIB, [EVENTS]), after: FakeJob(TIB, [EVENTS])},
                            {EVENTS: _table(EVENTS, partition=("DAY", "event_date"))})
        r = _analyze(client, before, after)
        assert any(f.description == "Partition filter removed" and f.risk == "high"
                   for f in r["structural_findings"])

    def test_missing_accuracy_with_an_external_table_is_unknown_cost(self):
        before = "SELECT a FROM `p.raw.orders`"
        after = "SELECT a FROM `p.raw.orders` JOIN `p.raw.sheet` USING (a)"
        client = FakeClient(
            {before: FakeJob(GIB, ["p.raw.orders"], accuracy=None),
             after: FakeJob(GIB, ["p.raw.orders", "p.raw.sheet"], accuracy=None)},
            {"p.raw.orders": _table("p.raw.orders"), "p.raw.sheet": _table("p.raw.sheet", type_="EXTERNAL")},
        )
        r = _analyze(client, before, after)
        assert r["cost"]["status"] == "unknown" and "external table raw.sheet" in r["cost"]["reason"]
        assert r["exempt_entry"] == "queries/q.sql"

    def test_missing_accuracy_without_an_external_table_is_costed(self):
        client = FakeClient({PRUNED: FakeJob(GIB, [EVENTS], accuracy=None), FULL: FakeJob(TIB, [EVENTS], accuracy=None)},
                            {EVENTS: _table(EVENTS)})
        assert _analyze(client, PRUNED, FULL)["cost"]["status"] == "changed"

    def test_narrowed_partition_filter_is_not_a_full_scan(self):
        """One of two conditions on the partition column is removed: the finding stays high,
        but neither it nor a High Scan Warning says the query reads every partition."""
        before = PRUNED + " AND event_date < '2024-02-01'"
        client = FakeClient({before: FakeJob(GIB, [EVENTS]), PRUNED: FakeJob(20 * GIB, [EVENTS])},
                            {EVENTS: _table(EVENTS, partition=("DAY", "event_date"), num_bytes=4 * TIB)})
        r = _analyze(client, before, PRUNED)
        high = [f for f in r["structural_findings"] if f.description == "Partition filter removed"]
        assert high and high[0].risk == "high" and high[0].partition_filter_kept
        body = format_comment(results=[r], total_before=0.0, total_after=1.0, total_delta=1.0, has_bq_creds=True,
                              blocked=False, block_reasons=[], plan="team")
        assert "every partition" not in body and "High Scan Warning" not in body
        assert "still has another filter on event\\_date" in body

    def test_date_like_clustering_column_with_an_unreadable_table_keeps_the_heuristic(self):
        """`created_at` clusters events, but orders (no metadata) could be partitioned by it."""
        orders = "p.analytics.orders"
        before = ("SELECT e.a FROM `p.analytics.events` e JOIN `p.analytics.orders` o USING (a) "
                  "WHERE created_at > '2024-01-01' AND e.a = 1")
        after = "SELECT e.a FROM `p.analytics.events` e JOIN `p.analytics.orders` o USING (a) WHERE e.a = 1"
        client = FakeClient(
            {before: FakeJob(GIB, [EVENTS, orders]), after: FakeJob(TIB, [EVENTS, orders])},
            {EVENTS: _table(EVENTS, clustering=["created_at"]), orders: Exception("403 Access Denied")},
        )
        r = _analyze(client, before, after)
        assert any(f.description == "Partition filter removed" and f.risk == "high" for f in r["structural_findings"])
        assert not any(f.description == "Cluster filter removed" for f in r["structural_findings"])

    def test_removed_cluster_filter_is_named(self):
        before = "SELECT a FROM `p.analytics.events` WHERE user_id = '42'"
        after = "SELECT a FROM `p.analytics.events`"
        client = FakeClient(
            {before: FakeJob(TIB, [EVENTS], "UPPER_BOUND"), after: FakeJob(TIB, [EVENTS], "UPPER_BOUND")},
            {EVENTS: _table(EVENTS, partition=("DAY", "event_date"), clustering=["user_id"])},
        )
        r = _analyze(client, before, after)
        assert any(f.description == "Cluster filter removed" for f in r["structural_findings"])

    def test_unknown_accuracy_is_not_costed(self):
        client = FakeClient({PRUNED: FakeJob(GIB, [EVENTS]), FULL: FakeJob(TIB, [EVENTS], accuracy="UNKNOWN")},
                            {EVENTS: _table(EVENTS)})
        r = _analyze(client, PRUNED, FULL)
        assert r["cost"]["status"] == "unknown"
        assert r["cost"]["reason"] == "BigQuery doesn't know how accurate the dry run of the PR version is"

    def test_missing_byte_estimate_is_never_zero(self):
        job = FakeJob(TIB, [EVENTS])
        job.total_bytes_processed = None
        client = FakeClient({PRUNED: FakeJob(GIB, [EVENTS]), FULL: job}, {EVENTS: _table(EVENTS)})
        r = _analyze(client, PRUNED, FULL)
        assert r["cost"]["status"] == "unknown" and r["cost"]["reason"] == "dry run returned no byte estimate"

    def test_deleting_a_query_bigquery_cannot_estimate_does_not_block(self):
        """A deletion can only lower the cost: it stays Removed, with no figure."""
        client = FakeClient({FULL: FakeJob(tables=[EVENTS], rls=True)}, {EVENTS: _table(EVENTS)})
        r = _analyze(client, FULL, None)
        assert r["cost"]["status"] == "removed" and r["cost"]["cost_before"] is None
        assert "exempt_entry" not in r
        s = summarize([r])
        assert s["items_unknown"] == 0 and s["known"] == 0 and s["worst"] is None
        body = format_comment(results=[r], total_before=0.0, total_after=0.0, total_delta=0.0, has_bq_creds=True,
                              blocked=False, block_reasons=[], plan="team")
        assert ("| `q` | — | — | — | **Removed** (not estimated: the base version reads a table with row-level "
                "security, and BigQuery dry runs give no byte estimate for those tables; doesn't block) |") in body
        assert ("ℹ️ 1 removed query BigQuery couldn't estimate is left out of the totals; removing a query never "
                "blocks.") in body
        assert "No significant cost change" not in body and "could be costed" not in body
        payload = _build_payload(_config(), s, [r])
        assert payload["queries"] == [] and payload["total_cost_before"] is None and payload["total_cost_delta"] is None

    def test_deleting_a_query_whose_base_dry_run_fails_does_not_block(self):
        """The old version no longer runs (a column it read was dropped): the row is
        Removed, with BigQuery's error in place of a figure, and nothing counts as unknown."""
        from main import _build_payload

        client = FakeClient(
            {FULL: Exception(UNRECOGNIZED),
             PRUNED: FakeJob(GIB, [EVENTS]), FULL + " LIMIT 1": FakeJob(2 * GIB, [EVENTS])},
            {EVENTS: _table(EVENTS)},
        )
        deleted = _analyze(client, FULL, None, name="old")
        grew = _analyze(client, PRUNED, FULL + " LIMIT 1", name="grew")
        assert deleted["cost"]["status"] == "removed" and deleted["cost"]["delta"] == 0.0
        assert deleted["cost"]["cost_before"] is None and "exempt_entry" not in deleted
        s = summarize([deleted, grew])
        assert s["items_unknown"] == 0 and s["known"] == 1 and s["worst"]["name"] == "grew"
        assert (s["total_bytes_before"], s["total_bytes_after"]) == (GIB, 2 * GIB)
        payload = _build_payload(_config(), s, [deleted, grew])
        assert payload["items_unknown"] == 0
        assert all(q["cost_after"] is not None for q in payload["queries"])  # only grew's figures are sent
        row = ("| `old` | — | — | — | **Removed** (not estimated: Production dry-run failed: Unrecognized name: "
               "user\\_id at \\[1:8\\]; doesn't block)")
        for kw, suffix in (({}, " |"), ({"bq_pricing_model": "editions"}, " [Editions - Slot Based] |")):
            body = _body([deleted, grew], s, **kw)
            assert row + suffix in body
            assert "| **TOTAL** |" not in body  # one costed query

    def test_deleting_a_query_whose_table_is_gone_does_not_block(self):
        client = FakeClient({FULL: Exception("404 Not found: Table p:analytics.events was not found in location US")})
        r = _analyze(client, FULL, None)
        assert r["cost"]["status"] == "removed" and r["cost"]["cost_before"] is None
        assert r["cost"]["reason"] == "the base version reads a table that does not exist"
        assert summarize([r])["items_unknown"] == 0

    def test_deleting_a_query_with_no_byte_estimate_does_not_block(self):
        job = FakeJob(TIB, [EVENTS])
        job.total_bytes_processed = None
        r = _analyze(FakeClient({FULL: job}, {EVENTS: _table(EVENTS)}), FULL, None)
        assert r["cost"]["status"] == "removed" and r["cost"]["reason"] == "dry run returned no byte estimate"

    def test_a_changed_query_whose_base_dry_run_fails_is_still_unknown(self):
        """Only a deletion is exempt: the PR version of a changed query can cost anything."""
        client = FakeClient({FULL: Exception(UNRECOGNIZED), PRUNED: FakeJob(GIB, [EVENTS])},
                            {EVENTS: _table(EVENTS)})
        r = _analyze(client, FULL, PRUNED)
        assert r["cost"]["status"] == "error" and r["cost"]["reason"].startswith("Production dry-run failed")
        assert summarize([r])["items_unknown"] == 1

    def test_deleting_a_lower_bound_query_shows_and_sends_no_figure(self):
        """The base version's lower bound (an external table) isn't a cost: it must not
        appear in the row, offset another query's increase in the totals, or be sent."""
        from main import _build_payload

        ext, query = "p.raw.sheet", "SELECT a FROM `p.raw.sheet`"
        client = FakeClient(
            {query: FakeJob(500 * GIB, [ext], accuracy="LOWER_BOUND"),
             PRUNED: FakeJob(GIB, [EVENTS]), FULL: FakeJob(100 * GIB, [EVENTS])},
            {EVENTS: _table(EVENTS), ext: _table(ext, type_="EXTERNAL")},
        )
        deleted = _analyze(client, query, None, name="ext")
        grew = _analyze(client, PRUNED, FULL, name="grew")
        assert deleted["cost"]["status"] == "removed" and deleted["cost"]["cost_before"] is None
        assert deleted["dry_run"]["bytes_before"] is None and "exempt_entry" not in deleted
        s = summarize([deleted, grew])
        assert s["items_unknown"] == 0
        assert (s["total_bytes_before"], s["total_bytes_after"]) == (GIB, 100 * GIB)
        payload = _build_payload(_config(), s, [deleted, grew])
        assert payload["total_bytes_before"] == GIB
        assert len(payload["queries"]) == 1 and payload["queries"][0]["bytes_before"] == GIB
        for kw in ({}, {"bq_pricing_model": "editions"}):
            body = _body([deleted, grew], s, **kw)
            assert "| `ext` | — | — | — | **Removed** (not estimated:" in body
            assert "**Data Scanned per run:** 1.00 GiB → 100.00 GiB (+99.00 GiB)" in body

    def test_row_level_security_base_with_a_pr_side_missing_column(self):
        """The row-level-security reason (and its hint) wins over "PR dry-run failed"."""
        after = "SELECT new_col FROM `p.analytics.events`"

        class Client(FakeClient):
            def query(self, sql, job_config=None, timeout=None, retry=None):
                if sql == after:
                    raise Exception("400 Unrecognized name: new_col at [1:8]")
                return super().query(sql, job_config, timeout, retry)

        client = Client({FULL: FakeJob(tables=[EVENTS], rls=True)}, {EVENTS: _table(EVENTS)})
        r = _analyze(client, FULL, after)
        assert r["cost"]["status"] == "unknown" and "row-level security" in r["cost"]["reason"]
        assert r["exempt_entry"] == "queries/q.sql"

    def test_deleting_a_query_after_the_dry_run_budget_is_spent_does_not_block(self):
        client = FakeClient({})
        r = _analyze(client, FULL, None, started_at=time.monotonic() - DRY_RUN_BUDGET_SECONDS - 1)
        assert r["cost"]["status"] == "removed" and r["cost"]["reason"] == "skipped: dry-run time budget exceeded"
        assert summarize([r])["items_unknown"] == 0

    def test_deleting_a_query_without_bigquery_credentials_is_unknown(self):
        """No client, no dry run: nothing says the deletion is all the PR does to that cost."""
        item = ChangedSQL(name="old", path="queries/old.sql", production_sql=FULL, pr_sql=None, adapter="sql_files")
        r = _analyze_item(item, None, _config(), time.monotonic())
        assert r["cost"]["status"] == "structural"
        assert summarize([r])["items_unknown"] == 1

    def test_exempt_entry_matches_exactly_the_file(self):
        for path in ("reports/q1,q2.sql", "queries/[x]orders.sql", "models/a*b?.sql", " lead.sql", "trail .sql ",
                     "\ttab.sql", "nbsp.sql\u00a0"):
            entry = exempt_entry(path)
            assert _matches_glob(path, entry), path
            assert not _matches_glob("reports/other.sql", entry)
        entry = exempt_entry("reports/q1,q2.sql")
        assert entry == "reports/q1[,]q2.sql"
        for other in ("reports/q1/q2.sql", "reports/q1_q2.sql", "reports/q1"):
            assert not _matches_glob(other, entry), other
        assert exempt_entry(" q.sql") == "[ ]q.sql" and not _matches_glob("q.sql", exempt_entry(" q.sql"))
        assert exempt_entry("\tq.sql") == "[\t]q.sql" and not _matches_glob("q.sql", exempt_entry("\tq.sql"))

    def test_a_literal_comma_in_a_glob_list(self):
        assert _matches_glob("a,b.sql", "x.sql, a[,]b.sql")
        assert _matches_glob("x.sql", "x.sql, a[,]b.sql")
        assert not _matches_glob("a", "x.sql, a[,]b.sql")

    def test_real_query_job_statistics(self):
        """Pins the raw statistics layout DiffCost reads (the client has no properties for it)."""
        from analysis.bq_dryrun import dry_run

        resource = {
            "jobReference": {"projectId": "p", "jobId": "j", "location": "US"},
            "configuration": {"query": {"query": "SELECT 1"}, "dryRun": True},
            "statistics": {
                "rowLevelSecurityStatistics": {"rowLevelSecurityApplied": True},
                "query": {"statementType": "SELECT", "totalBytesProcessedAccuracy": "UPPER_BOUND",
                          "referencedTables": [{"projectId": "p", "datasetId": "analytics", "tableId": "events"}]},
            },
        }
        client = MagicMock()
        client.query.return_value = bigquery.QueryJob.from_api_repr(resource, client)
        result = dry_run(client, "SELECT 1")
        assert result["row_level_security"] is True and result["bytes_processed"] is None
        assert result["accuracy"] == "UPPER_BOUND" and result["statement_type"] == "SELECT"
        assert result["referenced_tables"] == [EVENTS]

    def test_metadata_is_read_once_for_tables_several_items_share(self):
        client = FakeClient({PRUNED: FakeJob(GIB, [EVENTS]), FULL: FakeJob(TIB, [EVENTS])},
                            {EVENTS: _table(EVENTS)})
        cache = TableMetaCache(client)
        _analyze(client, PRUNED, FULL, name="a", cache=cache)
        _analyze(client, PRUNED, FULL, name="b", cache=cache)
        assert client.get_calls == [EVENTS]


def test_classify_cost_treats_unknown_reason_as_unknown():
    rec = classify_cost({"cost_before": 1.0, "cost_after": 0.0, "bytes_before": 1, "bytes_after": 0,
                         "error": None, "unknown_reason": "the dry run of the PR version gives only a lower bound"})
    assert rec["status"] == "unknown" and rec["reason"].startswith("the dry run of the PR version")
