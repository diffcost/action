"""Tests for structural SQL analysis."""

import pytest

from analysis.structural import analyze_structural_changes


class TestPartitionFilterRemoval:
    def test_removed_date_filter(self):
        prod = "SELECT * FROM orders WHERE order_date >= '2024-01-01'"
        pr = "SELECT * FROM orders"
        findings = analyze_structural_changes(prod, pr)
        assert any("Partition filter removed" in f.description or "WHERE" in f.description for f in findings)
        assert any(f.risk == "high" for f in findings)

    def test_removed_timestamp_filter(self):
        prod = "SELECT id FROM events WHERE event_timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)"
        pr = "SELECT id FROM events"
        findings = analyze_structural_changes(prod, pr)
        high_findings = [f for f in findings if f.risk == "high"]
        assert len(high_findings) >= 1

    def test_removed_partitiontime_filter(self):
        prod = "SELECT * FROM `project.dataset.table` WHERE _PARTITIONTIME >= '2024-01-01'"
        pr = "SELECT * FROM `project.dataset.table`"
        findings = analyze_structural_changes(prod, pr)
        assert any(f.risk == "high" for f in findings)

    def test_kept_filter_no_finding(self):
        prod = "SELECT * FROM orders WHERE order_date >= '2024-01-01'"
        pr = "SELECT * FROM orders WHERE order_date >= '2024-06-01'"
        findings = analyze_structural_changes(prod, pr)
        # Filter changed but not removed — should not be high risk
        high_findings = [f for f in findings if f.risk == "high"]
        assert len(high_findings) == 0

    def test_non_partition_substring_match_not_high_risk(self):
        """Removed predicate containing 'date' in substring (e.g. 'candidate') is not partition filter."""
        prod = "SELECT * FROM applications WHERE status = 'candidate'"
        pr = "SELECT * FROM applications"
        findings = analyze_structural_changes(prod, pr)
        # Should be medium risk (WHERE clause removed), NOT high risk (Partition filter removed)
        assert any(f.risk == "medium" and "WHERE clause removed" in f.description for f in findings)
        assert not any("Partition filter removed" in f.description for f in findings)


class TestSelectStar:
    def test_select_star_added(self):
        prod = "SELECT id, name, email FROM users"
        pr = "SELECT * FROM users"
        findings = analyze_structural_changes(prod, pr)
        assert any("SELECT *" in f.description for f in findings)
        assert any(f.risk == "high" for f in findings)

    def test_select_star_already_present(self):
        prod = "SELECT * FROM users WHERE active = true"
        pr = "SELECT * FROM users WHERE active = true AND role = 'admin'"
        findings = analyze_structural_changes(prod, pr)
        star_findings = [f for f in findings if "SELECT *" in f.description]
        assert len(star_findings) == 0

    def test_qualified_star_added(self):
        """sqlglot parses `o.*` as a Column whose name is a Star."""
        prod = "SELECT o.id, o.amount FROM orders o JOIN users u ON o.user_id = u.id"
        for pr in (
            "SELECT o.* FROM orders o JOIN users u ON o.user_id = u.id",
            "SELECT o.* EXCEPT (note) FROM orders o JOIN users u ON o.user_id = u.id",
        ):
            findings = analyze_structural_changes(prod, pr)
            assert any(f.description == "SELECT * added" and f.risk == "high" for f in findings), pr

    def test_bare_star_added_next_to_an_existing_qualified_star(self):
        joined = "FROM orders o JOIN users u ON o.user_id = u.id"
        findings = analyze_structural_changes(f"SELECT o.*, u.name {joined}", f"SELECT * {joined}")
        assert any(f.description == "SELECT * added" for f in findings)

    def test_star_for_another_table_added(self):
        joined = "FROM orders o JOIN users u ON o.user_id = u.id"
        findings = analyze_structural_changes(f"SELECT o.*, u.name {joined}", f"SELECT o.*, u.* {joined}")
        assert any(f.description == "SELECT * added" for f in findings)

    def test_qualified_star_inside_an_existing_bare_star_query(self):
        joined = "FROM orders o JOIN users u ON o.user_id = u.id"
        findings = analyze_structural_changes(f"SELECT * {joined}", f"SELECT o.* {joined}")
        assert not [f for f in findings if "SELECT *" in f.description]

    def test_qualified_star_already_present(self):
        prod = "SELECT o.* FROM orders o"
        pr = "SELECT o.* FROM orders o WHERE o.amount > 0"
        assert not [f for f in analyze_structural_changes(prod, pr) if "SELECT *" in f.description]

    def test_no_select_star(self):
        prod = "SELECT id, name FROM users"
        pr = "SELECT id, name, email FROM users"
        findings = analyze_structural_changes(prod, pr)
        star_findings = [f for f in findings if "SELECT *" in f.description]
        assert len(star_findings) == 0


class TestJoinChanges:
    def test_new_join_detected(self):
        prod = "SELECT o.id FROM orders o"
        pr = "SELECT o.id FROM orders o JOIN users u ON o.user_id = u.id"
        findings = analyze_structural_changes(prod, pr)
        assert any("JOIN" in f.description.upper() for f in findings)

    def test_cross_join_high_risk(self):
        prod = "SELECT a.id FROM table_a a"
        pr = "SELECT a.id FROM table_a a CROSS JOIN table_b b"
        findings = analyze_structural_changes(prod, pr)
        cross_findings = [f for f in findings if "CROSS" in f.description.upper()]
        assert len(cross_findings) >= 1
        assert all(f.risk == "high" for f in cross_findings)


class TestWhereClauseChanges:
    def test_all_where_removed(self):
        prod = "SELECT id FROM orders WHERE status = 'active' AND region = 'US'"
        pr = "SELECT id FROM orders"
        findings = analyze_structural_changes(prod, pr)
        assert any("WHERE" in f.description for f in findings)

    def test_partial_where_removed(self):
        prod = "SELECT id FROM orders WHERE status = 'active' AND region = 'US'"
        pr = "SELECT id FROM orders WHERE status = 'active'"
        findings = analyze_structural_changes(prod, pr)
        assert any("condition" in f.description.lower() or "WHERE" in f.description for f in findings)


class TestTableReferences:
    def test_new_table_reference(self):
        prod = "SELECT id FROM orders"
        pr = "SELECT o.id FROM orders o JOIN shipments s ON o.id = s.order_id"
        findings = analyze_structural_changes(prod, pr)
        assert any("table" in f.description.lower() or "JOIN" in f.description.upper() for f in findings)

    def test_cte_not_flagged_as_external_table_reference(self):
        """A CTE name in a FROM clause should not be reported as a new external table reference."""
        prod = "SELECT id FROM orders"
        pr = "WITH summary AS (SELECT id FROM orders) SELECT * FROM summary"
        findings = analyze_structural_changes(prod, pr)
        # Should detect new CTE, but NOT 'New table reference: summary'
        assert not any(f.description == "New table reference" and f.subject == "summary" for f in findings)


class TestSubqueryChanges:
    def test_new_cte_detected(self):
        prod = "SELECT id FROM orders"
        pr = "WITH active_orders AS (SELECT * FROM orders WHERE active = true) SELECT id FROM active_orders"
        findings = analyze_structural_changes(prod, pr)
        assert any("CTE" in f.description for f in findings)


class TestColumnExpansion:
    def test_significant_column_expansion(self):
        prod = "SELECT id, name FROM users"
        pr = "SELECT id, name, email, phone, address, city, state FROM users"
        findings = analyze_structural_changes(prod, pr)
        assert any("column" in f.description.lower() or "expanded" in f.description.lower() for f in findings)

    def test_minor_column_change_ignored(self):
        prod = "SELECT id, name FROM users"
        pr = "SELECT id, name, email FROM users"
        findings = analyze_structural_changes(prod, pr)
        col_findings = [f for f in findings if "column" in f.description.lower() and "expanded" in f.description.lower()]
        assert len(col_findings) == 0  # Only 1 column added, below threshold of 3


class TestEdgeCases:
    def test_unparseable_production_sql(self):
        prod = "NOT VALID SQL {{jinja}}"
        pr = "SELECT id FROM orders"
        findings = analyze_structural_changes(prod, pr)
        assert any("parse" in f.description.lower() for f in findings)

    def test_unparseable_pr_sql(self):
        prod = "SELECT id FROM orders"
        pr = "{{jinja_template}} NOT VALID {%block%} SQL {{ref('model')}}"
        findings = analyze_structural_changes(prod, pr)
        assert any("parse" in f.description.lower() for f in findings)

    def test_identical_sql(self):
        sql = "SELECT id, name FROM users WHERE active = true"
        findings = analyze_structural_changes(sql, sql)
        assert len(findings) == 0

    def test_empty_sql(self):
        findings = analyze_structural_changes("", "")
        # Should handle gracefully (parse error or empty findings)
        assert isinstance(findings, list)

    def test_complex_bq_sql(self):
        """Test with realistic BigQuery SQL."""
        prod = """
        SELECT
            DATE_TRUNC(order_date, MONTH) as month,
            COUNT(*) as order_count,
            SUM(amount) as total_amount
        FROM `project.dataset.orders`
        WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
        GROUP BY 1
        """
        pr = """
        SELECT
            DATE_TRUNC(order_date, MONTH) as month,
            COUNT(*) as order_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM `project.dataset.orders`
        GROUP BY 1
        """
        findings = analyze_structural_changes(prod, pr)
        # Should detect: WHERE clause removed (partition filter on order_date)
        assert any(f.risk == "high" for f in findings)


class TestFindingSorting:
    def test_high_risk_first(self):
        prod = "SELECT * FROM orders WHERE order_date > '2024-01-01'"
        pr = """
        SELECT *
        FROM orders
        CROSS JOIN users
        """
        findings = analyze_structural_changes(prod, pr)
        if len(findings) >= 2:
            risks = [f.risk for f in findings]
            assert risks == sorted(risks, key=lambda r: {"high": 0, "medium": 1, "low": 2}.get(r, 3))


class TestHeuristicsAndRobustness:
    def test_substring_date_is_not_partition(self):
        """`is_validated` and `candidate_id` contain 'date' but are not partition columns."""
        for col in ("is_validated", "candidate_id"):
            prod = f"SELECT * FROM t WHERE {col} = TRUE"
            findings = analyze_structural_changes(prod, "SELECT * FROM t")
            assert not any(f.description == "Partition filter removed" for f in findings), col

    def test_whole_word_date_tokens_are_partition(self):
        for col in ("event_date", "dt", "event_ts", "load_date"):
            prod = f"SELECT * FROM t WHERE {col} >= '2026-01-01'"
            findings = analyze_structural_changes(prod, "SELECT * FROM t")
            assert any(f.description == "Partition filter removed" for f in findings), col

    def test_partition_removal_not_duplicated_by_where_count(self):
        prod = "SELECT * FROM orders WHERE order_date >= '2024-01-01'"
        findings = analyze_structural_changes(prod, "SELECT * FROM orders")
        high = [f for f in findings if f.risk == "high"]
        assert len(high) == 1
        assert high[0].description == "Partition filter removed"
        assert high[0].subject.startswith("order_date")

    def test_tokenizer_error_does_not_raise(self):
        """sqlglot TokenError is not a ParseError subclass; it must not crash the run."""
        findings = analyze_structural_changes("SELECT 1", "SELECT 'abc FROM t")
        assert any("parse" in f.description.lower() for f in findings)

    def test_sql_kept_out_of_description(self):
        prod = "SELECT * FROM t WHERE status = 'x`y'"
        findings = analyze_structural_changes(prod, "SELECT * FROM t")
        removed = [f for f in findings if f.description == "WHERE clause removed"]
        assert removed and "`" not in removed[0].description


class TestWeakTokensAndDedupe:
    def test_weak_time_token_without_date_value_is_not_a_partition_filter(self):
        findings = analyze_structural_changes("SELECT 1 FROM t WHERE response_time_ms > 100", "SELECT 1 FROM t")
        assert not any(f.description == "Partition filter removed" for f in findings)
        assert any(f.description == "WHERE clause removed" for f in findings)

    def test_weak_token_with_date_value_is_a_partition_filter(self):
        findings = analyze_structural_changes(
            "SELECT 1 FROM t WHERE event_hour >= TIMESTAMP('2026-01-01')", "SELECT 1 FROM t"
        )
        assert any(f.description == "Partition filter removed" for f in findings)

    def test_new_join_is_not_also_reported_as_new_table_reference(self):
        findings = analyze_structural_changes(
            "SELECT a.id FROM proj.ds.a a", "SELECT a.id FROM proj.ds.a a LEFT JOIN proj.ds.b b ON a.id = b.id"
        )
        descriptions = [f.description for f in findings]
        assert "New LEFT JOIN" in descriptions
        assert "New table reference" not in descriptions


class TestSkipped:
    """`skipped` marks findings that say the analysis didn't run, so callers can't read
    "no risk" from them (the dbt Not costed exemption fails closed on those)."""

    def test_unparseable_sql_is_skipped(self):
        findings = analyze_structural_changes("SELECT id FROM t", "SELECT FROM WHERE (((")
        assert findings and all(f.skipped for f in findings)

    def test_failed_check_is_skipped(self, monkeypatch):
        import analysis.structural as structural

        monkeypatch.setattr(structural, "_run_checks", lambda a, b: 1 / 0)
        findings = analyze_structural_changes("SELECT id FROM t", "SELECT id FROM t WHERE x = 1")
        assert [f.description for f in findings] == ["Structural analysis failed"] and findings[0].skipped

    def test_real_findings_are_not_skipped(self):
        findings = analyze_structural_changes("SELECT id FROM t", "SELECT * FROM t")
        assert findings and not any(f.skipped for f in findings)


class TestPartitionColumnsFromMetadata:
    """With table metadata the partition column is a fact, not a guess from its name."""

    BEFORE = "SELECT a FROM t WHERE event_date >= '2024-01-01' AND created_at > '2024-01-01'"
    AFTER = "SELECT a FROM t"

    def test_confirmed_partition_column(self):
        findings = analyze_structural_changes(
            "SELECT a FROM t WHERE e.event_date >= '2024-01-01'", self.AFTER,
            partition_columns={"event_date": ["analytics.events (4.00 TiB table, partitioned daily)"]},
        )
        partition = [f for f in findings if f.description == "Partition filter removed"]
        assert len(partition) == 1 and partition[0].risk == "high"
        assert partition[0].partition_tables == ("analytics.events (4.00 TiB table, partitioned daily)",)
        assert partition[0].detail.startswith("event_date is the partition column of analytics.events")

    def test_an_unconfirmed_date_like_column_keeps_the_heuristic(self):
        """Metadata names event_date as the partition column; created_at is still a guess from
        its name, whatever the metadata says about the tables' columns."""
        findings = analyze_structural_changes(
            self.BEFORE, self.AFTER, partition_columns={"event_date": ["analytics.events (partitioned daily)"]},
        )
        by_subject = {f.subject: f for f in findings if f.subject}
        assert by_subject["event_date >= '2024-01-01'"].partition_tables
        created = by_subject["created_at > '2024-01-01'"]
        assert (created.description, created.risk) == ("Partition filter removed", "high")
        assert not created.partition_tables

    def test_an_alias_keeps_the_heuristic(self):
        """A CTE (or a dbt ephemeral model) renames the partition column: a filter on `dt`
        is still a partition filter."""
        before = ("WITH e AS (SELECT event_date AS dt, a FROM t) "
                  "SELECT a FROM e WHERE dt >= '2024-01-01' AND a = 1")
        after = "WITH e AS (SELECT event_date AS dt, a FROM t) SELECT a FROM e WHERE a = 1"
        findings = analyze_structural_changes(before, after, partition_columns={"event_date": ["d.t (partitioned daily)"]})
        assert any(f.description == "Partition filter removed" and f.risk == "high" for f in findings)

    def test_a_pseudo_column_keeps_the_heuristic(self):
        """_TABLE_SUFFIX on sharded tables is in no table's schema."""
        findings = analyze_structural_changes(
            "SELECT a FROM `p.d.events_*` WHERE _TABLE_SUFFIX BETWEEN '20240101' AND '20240107' AND a = 1",
            "SELECT a FROM `p.d.events_*` WHERE a = 1",
            partition_columns={},
        )
        assert any(f.description == "Partition filter removed" and f.risk == "high" for f in findings)

    def test_removed_filter_on_a_clustering_column(self):
        findings = analyze_structural_changes(
            "SELECT a FROM t WHERE user_id = 42", "SELECT a FROM t",
            cluster_columns={"user_id": ["analytics.events"]},
        )
        cluster = [f for f in findings if f.description == "Cluster filter removed"]
        assert cluster and cluster[0].risk == "medium" and cluster[0].subject == "user_id = 42"
        assert cluster[0].detail.startswith("user_id clusters analytics.events.")
        assert "real increase can be larger" in cluster[0].detail

    def test_oddly_named_partition_column(self):
        findings = analyze_structural_changes(
            "SELECT a FROM t WHERE bucket = 3", "SELECT a FROM t",
            partition_columns={"bucket": ["d.t (partitioned by integer range)"]},
        )
        assert any(f.partition_tables == ("d.t (partitioned by integer range)",) for f in findings)

    def test_ingestion_time_pseudo_column(self):
        label = "d.t (partitioned by ingestion time, daily)"
        findings = analyze_structural_changes(
            "SELECT a FROM t WHERE _PARTITIONTIME > TIMESTAMP('2024-01-01')", "SELECT a FROM t",
            partition_columns={"_partitiontime": [label], "_partitiondate": [label]},
        )
        assert any(f.partition_tables for f in findings)

    def test_a_date_like_clustering_column_keeps_the_partition_heuristic(self):
        """`created_at` clusters one table, but it can still partition another table: its
        removed filter counts as a removed partition filter."""
        before = "SELECT a FROM t JOIN u USING (a) WHERE created_at > '2024-01-01' AND a = 1"
        after = "SELECT a FROM t JOIN u USING (a) WHERE a = 1"
        findings = analyze_structural_changes(before, after, cluster_columns={"created_at": ["analytics.events"]})
        assert [(f.description, f.risk) for f in findings] == [("Partition filter removed", "high")]

    def test_another_filter_on_the_partition_column_remains(self):
        """One of two conditions on the partition column was removed: more partitions, not all."""
        findings = analyze_structural_changes(
            "SELECT a FROM t WHERE event_date >= '2024-01-01' AND event_date < '2024-02-01'",
            "SELECT a FROM t WHERE event_date >= '2024-01-01'",
            partition_columns={"event_date": ["analytics.events (4.00 TiB table, partitioned daily)"]},
        )
        partition = [f for f in findings if f.description == "Partition filter removed"]
        assert len(partition) == 1 and partition[0].risk == "high" and partition[0].partition_filter_kept
        assert "every partition" not in partition[0].detail
        assert "still has another filter on event_date" in partition[0].detail

    def test_another_filter_on_a_guessed_partition_column_remains(self):
        findings = analyze_structural_changes(
            "SELECT a FROM t WHERE event_date >= '2024-01-01' AND event_date < '2024-02-01'",
            "SELECT a FROM t WHERE event_date >= '2024-01-01'",
        )
        partition = [f for f in findings if f.description == "Partition filter removed"]
        assert len(partition) == 1 and partition[0].partition_filter_kept
        assert "full table scan" not in partition[0].detail

    def test_a_kept_filter_on_another_tables_column_of_the_same_name_does_not_count(self):
        """`a.dt` stays, `b.dt` goes: table b is read in full, so the finding still says so."""
        findings = analyze_structural_changes(
            "SELECT 1 FROM a JOIN b USING (id) WHERE a.dt = '2024-01-01' AND b.dt = '2024-01-01'",
            "SELECT 1 FROM a JOIN b USING (id) WHERE a.dt = '2024-01-01'",
            partition_columns={"dt": ["d.a (partitioned daily)", "d.b (partitioned daily)"]},
        )
        partition = [f for f in findings if f.description == "Partition filter removed"]
        assert len(partition) == 1 and not partition[0].partition_filter_kept
        assert partition[0].detail.endswith("Without this filter the query reads every partition.")

    def test_the_same_condition_kept_in_another_subquery_does_not_count(self):
        """Both CTEs had the lower bound; the PR drops the whole WHERE of one of them."""
        before = ("WITH x AS (SELECT a FROM t WHERE event_date >= '2024-01-01' AND event_date < '2024-02-01'), "
                  "y AS (SELECT a FROM u WHERE event_date >= '2024-01-01') SELECT * FROM x JOIN y USING (a)")
        after = ("WITH x AS (SELECT a FROM t), "
                 "y AS (SELECT a FROM u WHERE event_date >= '2024-01-01') SELECT * FROM x JOIN y USING (a)")
        for kwargs in ({"partition_columns": {"event_date": ["d.t (partitioned daily)"]}}, {}):
            findings = analyze_structural_changes(before, after, **kwargs)
            partition = [f for f in findings if f.description == "Partition filter removed"]
            assert partition and not any(f.partition_filter_kept for f in partition), kwargs

    def test_a_kept_filter_only_in_an_outer_query_does_not_count(self):
        before = ("SELECT a FROM (SELECT a, event_date FROM t WHERE event_date < '2024-02-01') "
                  "WHERE event_date >= '2024-01-01'")
        after = "SELECT a FROM (SELECT a, event_date FROM t) WHERE event_date >= '2024-01-01'"
        findings = analyze_structural_changes(before, after, partition_columns={"event_date": ["d.t (partitioned daily)"]})
        partition = [f for f in findings if f.description == "Partition filter removed"]
        assert len(partition) == 1 and not partition[0].partition_filter_kept

    def test_a_kept_filter_with_the_same_qualifier_counts(self):
        findings = analyze_structural_changes(
            "SELECT 1 FROM a JOIN b USING (id) WHERE b.dt >= '2024-01-01' AND b.dt < '2024-02-01' AND a.x = 1",
            "SELECT 1 FROM a JOIN b USING (id) WHERE b.dt >= '2024-01-01' AND a.x = 1",
            partition_columns={"dt": ["d.b (partitioned daily)"]},
        )
        partition = [f for f in findings if f.description == "Partition filter removed"]
        assert len(partition) == 1 and partition[0].partition_filter_kept

    def test_the_only_filter_on_the_partition_column_reads_every_partition(self):
        findings = analyze_structural_changes(
            "SELECT a FROM t WHERE event_date >= '2024-01-01' AND a = 1", "SELECT a FROM t WHERE a = 1",
            partition_columns={"event_date": ["analytics.events (partitioned daily)"]},
        )
        partition = [f for f in findings if f.description == "Partition filter removed"]
        assert not partition[0].partition_filter_kept
        assert partition[0].detail.endswith("Without this filter the query reads every partition.")


D = "event_date >= '2024-01-01'"
D2 = "event_date >= '2024-06-01'"
EV, OR, TH, US = "`p.d.events`", "`p.d.orders`", "`p.d.this`", "`p.d.users`"
PARTITIONED = {"event_date": ["p.d.events (4.00 TiB table, partitioned daily)"]}

# A removed filter is a removed partition filter: (behaviour, base SQL, PR SQL)
REMOVED_FILTERS = [
    ("dropped from one CTE while the same text stays in another table's CTE",
     f"WITH x AS (SELECT * FROM {EV} WHERE {D}), y AS (SELECT * FROM {OR} WHERE {D}) SELECT * FROM x JOIN y USING (id)",
     f"WITH x AS (SELECT * FROM {EV}), y AS (SELECT * FROM {OR} WHERE {D}) SELECT * FROM x JOIN y USING (id)"),
    ("dropped from one CTE while changed in another table's CTE",
     f"WITH x AS (SELECT * FROM {EV} WHERE {D}), y AS (SELECT * FROM {OR} WHERE {D}) SELECT * FROM x JOIN y USING (id)",
     f"WITH x AS (SELECT * FROM {EV}), y AS (SELECT * FROM {OR} WHERE {D2}) SELECT * FROM x JOIN y USING (id)"),
    ("two CTEs on one table: dropped in one, changed in the other",
     f"WITH s AS (SELECT * FROM {EV} WHERE {D} AND type = 'signup'), b AS (SELECT * FROM {EV} WHERE {D} AND type = 'buy') "
     "SELECT * FROM s JOIN b USING (id)",
     f"WITH s AS (SELECT * FROM {EV} WHERE type = 'signup'), b AS (SELECT * FROM {EV} WHERE {D2} AND type = 'buy') "
     "SELECT * FROM s JOIN b USING (id)"),
    ("UNION ALL branches on one table: dropped in one, changed in the other",
     f"SELECT id FROM {EV} WHERE {D} AND a = 1 UNION ALL SELECT id FROM {EV} WHERE {D} AND a = 2",
     f"SELECT id FROM {EV} WHERE a = 1 UNION ALL SELECT id FROM {EV} WHERE {D2} AND a = 2"),
    ("moved to the same column of another joined table",
     f"SELECT * FROM {EV} a JOIN {OR} b USING (id) WHERE a.{D}",
     f"SELECT * FROM {EV} a JOIN {OR} b USING (id) WHERE b.{D}"),
    ("replaced by IS NOT NULL",
     f"SELECT * FROM {EV} WHERE {D}",
     f"SELECT * FROM {EV} WHERE event_date IS NOT NULL"),
    ("dropped from a GA4-style query with two UNNEST joins",
     f"SELECT e.event_date FROM {EV} e, UNNEST(e.items) AS item, UNNEST(e.event_params) AS ep "
     f"WHERE e.{D} AND ep.key = 'k'",
     f"SELECT e.event_date FROM {EV} e, UNNEST(e.items) AS item, UNNEST(e.event_params) AS ep WHERE ep.key = 'k'"),
    ("two CTEs on one table: dropped in one, the other's rewritten as BETWEEN",
     f"WITH s AS (SELECT * FROM {EV} WHERE {D} AND type = 's'), t AS (SELECT * FROM {EV} WHERE {D} AND type = 't') "
     "SELECT * FROM s JOIN t USING (id)",
     f"WITH s AS (SELECT * FROM {EV} WHERE type = 's'), "
     f"t AS (SELECT * FROM {EV} WHERE event_date BETWEEN '2024-01-01' AND '2024-02-01' AND type = 't') "
     "SELECT * FROM s JOIN t USING (id)"),
    ("self-join: one side's filter dropped while the other side gets an upper bound",
     f"SELECT a.id FROM {EV} a JOIN {EV} b ON a.sid = b.sid WHERE a.{D} AND b.{D}",
     f"SELECT a.id FROM {EV} a JOIN {EV} b ON a.sid = b.sid WHERE a.{D} AND a.event_date < '2024-02-01'"),
    # Conditions are compared across the whole query, not per table: a filter that moves to
    # another table's subquery, or is dropped while another table's subquery gains one, isn't seen
    pytest.param(
        "moved from the events CTE to the orders CTE",
        f"WITH x AS (SELECT * FROM {EV} WHERE {D}), y AS (SELECT * FROM {OR}) SELECT * FROM x JOIN y USING (id)",
        f"WITH x AS (SELECT * FROM {EV}), y AS (SELECT * FROM {OR} WHERE {D}) SELECT * FROM x JOIN y USING (id)",
        marks=pytest.mark.xfail(strict=True, reason="known limit: no per-table comparison"),
    ),
    pytest.param(
        "dropped from the events CTE while a date comparison is added to the orders CTE",
        f"WITH x AS (SELECT * FROM {EV} WHERE {D}), y AS (SELECT * FROM {OR}) SELECT * FROM x JOIN y USING (id)",
        f"WITH x AS (SELECT * FROM {EV}), y AS (SELECT * FROM {OR} WHERE {D2}) SELECT * FROM x JOIN y USING (id)",
        marks=pytest.mark.xfail(strict=True, reason="known limit: no per-table comparison"),
    ),
    ("a surplus copy of the upper bound in another CTE doesn't replace the dropped lower bound",
     f"WITH e AS (SELECT * FROM {EV} WHERE {D} AND event_date < '2024-02-01'), o AS (SELECT * FROM {OR}) "
     "SELECT * FROM e JOIN o USING (id)",
     f"WITH e AS (SELECT * FROM {EV} WHERE event_date < '2024-02-01'), "
     f"o AS (SELECT * FROM {OR} WHERE event_date < '2024-02-01' AND status = 'x') SELECT * FROM e JOIN o USING (id)"),
    ("plain removal from a single scan",
     f"SELECT * FROM {EV} WHERE {D} AND a = 1",
     f"SELECT * FROM {EV} WHERE a = 1"),
    ("dropped while a join is added",
     f"SELECT a FROM {EV} WHERE {D}",
     f"SELECT a FROM {EV} JOIN {US} USING (a)"),
    ("lower bound dropped while the upper bound changes",
     f"SELECT * FROM {EV} WHERE {D} AND event_date < '2024-02-01'",
     f"SELECT * FROM {EV} WHERE event_date < '2024-03-01'"),
    ("dropped from an UPDATE",
     f"UPDATE {EV} SET a = 1 WHERE {D} AND a = 2",
     f"UPDATE {EV} SET a = 1 WHERE a = 2"),
    # A function of the column that BigQuery doesn't prune on, or a pruning one with a column
    # among its other arguments, doesn't replace a partition filter
    *[(f"replaced by {new}", f"SELECT * FROM {EV} WHERE {D} AND a = 1", f"SELECT * FROM {EV} WHERE {new} AND a = 1")
      for new in [
          "EXTRACT(MONTH FROM event_date) = 3",
          "FORMAT_DATE('%Y-%m', event_date) >= '2024-01'",
          "FORMAT_TIMESTAMP('%Y-%m', event_date) >= '2024-01'",
          "DATE_ADD(event_date, INTERVAL lag_days DAY) >= '2024-01-01'",
          "DATE_DIFF(event_date, first_seen_date, DAY) >= 0",
          "event_date + INTERVAL 1 DAY > CURRENT_DATE()",
      ]],
    *[(f"replaced by {new}", "SELECT * FROM `p.d.hits` WHERE event_ts >= '2024-01-01' AND a = 1",
       f"SELECT * FROM `p.d.hits` WHERE {new} AND a = 1")
      for new in [
          "TIMESTAMP_TRUNC(event_ts, DAY, user_tz) >= '2024-01-01'",
          "DATE(event_ts, user_tz) >= '2024-01-01'",
          "EXTRACT(HOUR FROM event_ts) = 20",
      ]],
    # Not on BigQuery's list of functions that prune
    *[(f"replaced by {new}", "SELECT * FROM `p.d.visits` WHERE event_dt >= '2024-01-01' AND a = 1",
       f"SELECT * FROM `p.d.visits` WHERE {new} AND a = 1")
      for new in [
          "DATETIME_ADD(event_dt, INTERVAL 1 DAY) >= '2024-01-02'",
          "FORMAT_DATETIME('%F', event_dt) >= '2024-01-01'",
      ]],
    *[(f"replaced by {new}", f"SELECT * FROM {EV} WHERE {D} AND a = 1", f"SELECT * FROM {EV} WHERE {new} AND a = 1")
      for new in ["EXTRACT(WEEK(MONDAY) FROM event_date) = 10", "EXTRACT(ISOYEAR FROM event_date) = 2024"]],
    ("a pruning EXTRACT(YEAR ...) replaced by EXTRACT(MONTH ...)",
     f"SELECT * FROM {EV} WHERE EXTRACT(YEAR FROM event_date) = 2024 AND a = 1",
     f"SELECT * FROM {EV} WHERE EXTRACT(MONTH FROM event_date) = 3 AND a = 1"),
    ("a pruning DATE_SUB comparison replaced by IS NOT NULL",
     f"SELECT * FROM {EV} WHERE DATE_SUB(event_date, INTERVAL 1 DAY) >= '2023-12-31'",
     f"SELECT * FROM {EV} WHERE event_date IS NOT NULL"),
]

# No high or medium finding: (behaviour, base SQL, PR SQL)
NO_REMOVED_FILTER = [
    ("range narrowed in place", f"SELECT * FROM {EV} WHERE {D}", f"SELECT * FROM {EV} WHERE {D2}"),
    ("LIKE edited",
     f"SELECT * FROM {EV} WHERE {D} AND name LIKE 'a%'", f"SELECT * FROM {EV} WHERE {D} AND name LIKE 'b%'"),
    ("!= edited", f"SELECT * FROM {EV} WHERE {D} AND x != 1", f"SELECT * FROM {EV} WHERE {D} AND x != 2"),
    ("IN-subquery edited",
     f"SELECT * FROM {EV} WHERE {D} AND id IN (SELECT id FROM `p.d.u` WHERE status = 'a')",
     f"SELECT * FROM {EV} WHERE {D} AND id IN (SELECT id FROM `p.d.u` WHERE status = 'b')"),
    ("incremental lookback edited",
     f"SELECT * FROM {EV} WHERE event_date > (SELECT MAX(event_date) FROM {TH})",
     f"SELECT * FROM {EV} WHERE event_date > (SELECT DATE_SUB(MAX(event_date), INTERVAL 3 DAY) FROM {TH})"),
    ("filter pushed into the CTE",
     f"WITH x AS (SELECT * FROM {EV}) SELECT * FROM x WHERE {D}", f"WITH x AS (SELECT * FROM {EV} WHERE {D}) SELECT * FROM x"),
    ("filter pulled up from the CTE",
     f"WITH x AS (SELECT * FROM {EV} WHERE {D}) SELECT * FROM x", f"WITH x AS (SELECT * FROM {EV}) SELECT * FROM x WHERE {D}"),
    ("filter pushed into a derived table",
     f"SELECT * FROM (SELECT * FROM {EV}) WHERE {D}", f"SELECT * FROM (SELECT * FROM {EV} WHERE {D})"),
    ("filter pulled up from a derived table",
     f"SELECT * FROM (SELECT * FROM {EV} WHERE {D})", f"SELECT * FROM (SELECT * FROM {EV}) WHERE {D}"),
    ("filter pushed into the CTE under another qualifier",
     f"WITH x AS (SELECT * FROM {EV}) SELECT * FROM x WHERE x.{D}",
     f"WITH x AS (SELECT * FROM {EV} e WHERE e.{D}) SELECT * FROM x"),
    ("filter pushed into a CTE that a join reads",
     f"SELECT * FROM {EV} JOIN {US} USING (id) WHERE {D}",
     f"WITH e AS (SELECT * FROM {EV} WHERE {D}) SELECT * FROM e JOIN {US} USING (id)"),
    ("filter kept in a renamed CTE",
     f"WITH x AS (SELECT a FROM {EV} WHERE {D}) SELECT * FROM x", f"WITH z AS (SELECT a FROM {EV} WHERE {D}) SELECT * FROM z"),
    ("filter moved out of a correlated subquery",
     f"SELECT * FROM {EV} e WHERE EXISTS (SELECT 1 FROM {US} u WHERE u.id = e.id AND e.{D})",
     f"SELECT * FROM {EV} e WHERE e.{D} AND EXISTS (SELECT 1 FROM {US} u WHERE u.id = e.id)"),
    ("EXISTS subquery edited",
     f"SELECT * FROM {EV} e WHERE EXISTS (SELECT 1 FROM {US} u WHERE u.id = e.id AND u.s = 'a')",
     f"SELECT * FROM {EV} e WHERE EXISTS (SELECT 1 FROM {US} u WHERE u.id = e.id AND u.s = 'b')"),
    ("IN UNNEST list edited",
     f"SELECT * FROM {EV} WHERE event_date IN UNNEST(@a)", f"SELECT * FROM {EV} WHERE event_date IN UNNEST(@b)"),
    ("UPDATE range edited",
     f"UPDATE {EV} SET a = 1 WHERE {D}", f"UPDATE {EV} SET a = 1 WHERE {D2}"),
    ("GA4-style query with two UNNEST joins: a parameter filter edited",
     f"SELECT e.event_date FROM {EV} e, UNNEST(e.items) AS item, UNNEST(e.event_params) AS ep "
     f"WHERE e.{D} AND ep.key = 'page_location'",
     f"SELECT e.event_date FROM {EV} e, UNNEST(e.items) AS item, UNNEST(e.event_params) AS ep "
     f"WHERE e.{D} AND ep.key = 'page_title'"),
    ("equality on another column replaced by IN UNNEST",
     f"SELECT * FROM {EV} WHERE {D} AND status = 'active'", f"SELECT * FROM {EV} WHERE {D} AND status IN UNNEST(@s)"),
    ("boolean comparison replaced by NOT",
     f"SELECT * FROM {EV} WHERE {D} AND is_deleted = FALSE", f"SELECT * FROM {EV} WHERE {D} AND NOT is_deleted"),
    ("unchanged query", f"SELECT * FROM {EV} WHERE {D}", f"SELECT * FROM {EV} WHERE {D}"),
    # Functions BigQuery still prunes partitions on when their other arguments are constant
    *[(f"replaced by {new}", f"SELECT * FROM {EV} WHERE {D} AND a = 1", f"SELECT * FROM {EV} WHERE {new} AND a = 1")
      for new in [
          "DATE_ADD(event_date, INTERVAL 1 DAY) >= '2024-01-02'",
          "DATE_SUB(event_date, INTERVAL 1 DAY) >= '2023-12-31'",
          "DATE_DIFF(CURRENT_DATE(), event_date, DAY) <= 30",
          "DATE_DIFF(event_date, '2024-01-01', DAY) >= 0",
          "EXTRACT(YEAR FROM event_date) IN (2024, 2025)",
          "DATE_ADD(event_date, INTERVAL @lag DAY) BETWEEN '2024-01-02' AND '2025-01-01'",
      ]],
    *[(f"replaced by {new}", "SELECT * FROM `p.d.hits` WHERE event_ts >= '2024-01-01' AND a = 1",
       f"SELECT * FROM `p.d.hits` WHERE {new} AND a = 1")
      for new in [
          "TIMESTAMP_ADD(event_ts, INTERVAL 1 HOUR) >= '2024-01-01'",
          "TIMESTAMP_SUB(event_ts, INTERVAL 1 HOUR) >= '2024-01-01'",
          "TIMESTAMP_DIFF(event_ts, TIMESTAMP '2024-01-01', HOUR) >= 0",
          "TIMESTAMP_TRUNC(event_ts, DAY, 'UTC') >= '2024-01-01'",
          "EXTRACT(DATE FROM event_ts) >= '2024-01-01'",
          "EXTRACT(YEAR FROM event_ts) >= 2024",
          "FORMAT_TIMESTAMP('%F', event_ts) >= '2024-01-01'",
          "FORMAT_TIMESTAMP('%Y-%m-%d', event_ts) >= '2024-01-01'",
          "FORMAT_TIMESTAMP('%Y%m%d', event_ts) >= '20240101'",
          "SAFE_CAST(event_ts AS DATE) >= '2024-01-01'",
      ]],
    ("replaced by DATETIME_DIFF(event_dt, DATETIME '2024-01-01', DAY) >= 0",
     "SELECT * FROM `p.d.visits` WHERE event_dt >= '2024-01-01' AND a = 1",
     "SELECT * FROM `p.d.visits` WHERE DATETIME_DIFF(event_dt, DATETIME '2024-01-01', DAY) >= 0 AND a = 1"),
    ("a pruning EXTRACT(YEAR ...) edited in place",
     f"SELECT * FROM {EV} WHERE EXTRACT(YEAR FROM event_date) = 2024",
     f"SELECT * FROM {EV} WHERE EXTRACT(YEAR FROM event_date) = 2025"),
]


def _ids(rows):
    return [(row.values if hasattr(row, "values") else row)[0] for row in rows]


@pytest.mark.parametrize("metadata", [True, False], ids=["with metadata", "column-name guess"])
class TestWhichFiltersCountAsRemoved:
    """WHERE conditions are compared across the whole query, one copy for one copy: a filter
    moved between a CTE and its reader is no change, one the PR has fewer times is removed, and
    an added condition on the same column stands in for one removed copy only (a partition
    filter only by another comparison that prunes)."""

    @pytest.mark.parametrize("label,before,after", REMOVED_FILTERS, ids=_ids(REMOVED_FILTERS))
    def test_removed_partition_filter(self, metadata, label, before, after):
        findings = analyze_structural_changes(before, after, **({"partition_columns": PARTITIONED} if metadata else {}))
        assert any(f.description == "Partition filter removed" and f.risk == "high" for f in findings)

    @pytest.mark.parametrize("label,before,after", NO_REMOVED_FILTER, ids=_ids(NO_REMOVED_FILTER))
    def test_no_removed_filter(self, metadata, label, before, after):
        findings = analyze_structural_changes(before, after, **({"partition_columns": PARTITIONED} if metadata else {}))
        assert not [(f.description, f.risk) for f in findings if f.risk in ("high", "medium")]

    def test_a_boolean_column_filter_swapped_for_another_column_is_removed(self, metadata):
        findings = analyze_structural_changes(
            f"SELECT * FROM {EV} WHERE {D} AND is_active",
            f"SELECT * FROM {EV} WHERE {D} AND is_verified",
            **({"partition_columns": PARTITIONED} if metadata else {}),
        )
        assert [(f.description, f.risk, f.subject) for f in findings] == [("WHERE clause removed", "medium", "is_active")]

    def test_a_swapped_table_with_the_same_filter_is_a_new_table_not_a_removed_filter(self, metadata):
        findings = analyze_structural_changes(
            f"SELECT * FROM `p.a.stg_events` WHERE {D}",
            f"SELECT * FROM `p.a.int_events` WHERE {D}",
            **({"partition_columns": PARTITIONED} if metadata else {}),
        )
        assert [f.description for f in findings] == ["New table reference"]


class TestKeptFilterMustPrune:
    """Only a comparison of the column with a value keeps partitions pruned: the column bare,
    or inside a function BigQuery prunes on with its other arguments constant."""

    PARTITIONS = {"event_date": ["d.events (partitioned daily)"]}

    def _kept(self, before, after):
        partition = [
            f for f in analyze_structural_changes(before, after, partition_columns=self.PARTITIONS)
            if f.description == "Partition filter removed"
        ]
        assert len(partition) == 1
        return partition[0].partition_filter_kept

    @pytest.mark.parametrize("kept", [
        "event_date IS NOT NULL",
        "(event_date < '2024-02-01' OR is_late)",
        "EXTRACT(DAYOFWEEK FROM event_date) = 2",
        "event_date = other_date",
        "event_date >= (SELECT MAX(d) FROM u)",
        "CAST(event_date AS STRING) < '2024-02-01'",
        "event_date IN UNNEST(@days)",
        "EXTRACT(MONTH FROM event_date) = 1",
        "FORMAT_DATE('%Y-%m', event_date) = '2024-01'",
        "FORMAT_TIMESTAMP('%Y-%m', event_date) = '2024-01'",
        "DATE_ADD(event_date, INTERVAL lag_days DAY) < '2024-02-01'",
        "DATE_DIFF(event_date, other_date, DAY) < 30",
    ])
    def test_a_condition_that_does_not_prune(self, kept):
        assert not self._kept(
            f"SELECT a FROM t WHERE event_date >= '2024-01-01' AND {kept}", f"SELECT a FROM t WHERE {kept}"
        )

    @pytest.mark.parametrize("kept", [
        "event_date < '2024-02-01'",
        "'2024-02-01' > event_date",
        "DATE(event_date) < '2024-02-01'",
        "event_date BETWEEN '2023-01-01' AND @end_date",
        "event_date IN ('2024-01-01', '2024-01-02')",
        "event_date < DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)",
        "CAST(event_date AS DATE) < '2024-02-01'",
        "TIMESTAMP_TRUNC(event_date, DAY) < TIMESTAMP('2024-02-01')",
        "EXTRACT(YEAR FROM event_date) = 2024",
        "DATE_ADD(event_date, INTERVAL 1 DAY) < '2024-02-02'",
        "DATE_DIFF(CURRENT_DATE(), event_date, DAY) <= 30",
        "FORMAT_TIMESTAMP('%F', event_date) < '2024-02-01'",
    ])
    def test_a_condition_that_prunes(self, kept):
        assert self._kept(
            f"SELECT a FROM t WHERE event_date >= '2024-01-01' AND {kept}", f"SELECT a FROM t WHERE {kept}"
        )

    def test_a_new_condition_that_does_not_prune_is_no_changed_filter(self):
        findings = analyze_structural_changes(
            "SELECT a FROM t WHERE event_date >= '2024-01-01'", "SELECT a FROM t WHERE event_date IS NOT NULL",
            partition_columns=self.PARTITIONS,
        )
        partition = [f for f in findings if f.description == "Partition filter removed"]
        assert len(partition) == 1 and not partition[0].partition_filter_kept


def _blocking(findings):
    """The findings that make a dbt Not costed model count as unknown, plus any skipped one."""
    return [(f.description, f.risk) for f in findings if f.risk in ("high", "medium") or f.skipped]


def _partition_removed(findings):
    return any(f.description == "Partition filter removed" and f.risk == "high" for f in findings)


class TestParenthesizedConditions:
    """`(a AND b)` is two conditions, like `a AND b`."""

    @pytest.mark.parametrize("metadata", [PARTITIONED, None], ids=["with metadata", "column-name guess"])
    def test_partition_filter_dropped_from_a_parenthesized_group_is_removed(self, metadata):
        for after in ("SELECT id FROM t WHERE status = 'paid'", "SELECT id FROM t WHERE status = 'active'"):
            findings = analyze_structural_changes(
                f"SELECT id FROM t WHERE ({D} AND status = 'active')", after, partition_columns=metadata,
            )
            assert _partition_removed(findings), after

    def test_nested_parentheses_are_split_too(self):
        findings = analyze_structural_changes(
            f"SELECT id FROM t WHERE (({D} AND a = 1)) AND x = 1", "SELECT id FROM t WHERE a = 1 AND x = 1"
        )
        assert _partition_removed(findings)

    def test_adding_or_removing_parentheses_is_no_change(self):
        flat, grouped = f"SELECT id FROM t WHERE a = 1 AND {D}", f"SELECT id FROM t WHERE (a = 1 AND {D})"
        assert analyze_structural_changes(flat, grouped) == []
        assert analyze_structural_changes(grouped, flat) == []

    def test_or_and_not_groups_stay_one_condition(self):
        import sqlglot
        from analysis.structural import _split_and_predicates

        for where, conditions in (
            (f"({D} OR is_late) AND a = 1", [f"({D} OR is_late)", "a = 1"]),
            (f"NOT ({D} AND a = 1)", [f"NOT ({D} AND a = 1)"]),
            (f"(({D} AND a = 1)) AND x = 1", [D, "a = 1", "x = 1"]),
        ):
            where_clause = sqlglot.parse_one(f"SELECT id FROM t WHERE {where}", dialect="bigquery").args["where"]
            assert [c.sql(dialect="bigquery") for c in _split_and_predicates(where_clause.this)] == conditions, where


class TestFewerConditionTexts:
    """A query that keeps some conditions is flagged per removed condition, never by a count."""

    def test_two_bounds_rewritten_as_between_is_no_finding(self):
        for column, low, high in (("event_date", "'2024-01-01'", "'2024-01-31'"), ("amount", "1", "5")):
            findings = analyze_structural_changes(
                f"SELECT id FROM t WHERE {column} >= {low} AND {column} <= {high}",
                f"SELECT id FROM t WHERE {column} BETWEEN {low} AND {high}",
            )
            assert not _blocking(findings), column

    def test_a_removed_condition_is_flagged_once(self):
        findings = analyze_structural_changes(
            "SELECT id FROM t WHERE a = 1 AND b = 2", "SELECT id FROM t WHERE a = 1"
        )
        assert _blocking(findings) == [("WHERE clause removed", "medium")]


class TestTablesNamedLikeACte:
    """A CTE is referenced by its bare name: a qualified table is a table."""

    def test_import_cte_repointed_to_another_dataset_is_a_new_table(self):
        findings = analyze_structural_changes(
            "WITH orders AS (SELECT id FROM p.stg.orders) SELECT id FROM orders",
            "WITH orders AS (SELECT id FROM p.raw.orders) SELECT id FROM orders",
        )
        assert any(f.description == "New table reference" and f.subject == "p.raw.orders" for f in findings)

    def test_import_cte_renamed_is_no_new_table(self):
        findings = analyze_structural_changes(
            "WITH orders AS (SELECT id FROM p.stg.orders) SELECT id FROM orders",
            "WITH src_orders AS (SELECT id FROM p.stg.orders) SELECT id FROM src_orders",
        )
        assert not _blocking(findings)

    def test_qualified_table_read_next_to_a_cte_of_the_same_name_is_a_new_table(self):
        findings = analyze_structural_changes(
            "WITH orders AS (SELECT 1 AS id) SELECT id FROM orders",
            "WITH orders AS (SELECT 1 AS id) SELECT id FROM orders WHERE id IN (SELECT id FROM raw.orders)",
        )
        assert any(f.description == "New table reference" and f.subject == "raw.orders" for f in findings)


class TestSeveralStatements:
    """Every statement of a file is compared, as one query."""

    def test_filter_dropped_from_a_later_statement_is_removed(self):
        for before, after in (
            (f"SELECT 1; SELECT id FROM raw.events WHERE {D}", "SELECT 1; SELECT id FROM raw.events"),
            ("DECLARE d DATE DEFAULT CURRENT_DATE(); SELECT id FROM raw.events WHERE event_date >= d;",
             "DECLARE d DATE DEFAULT CURRENT_DATE(); SELECT id FROM raw.events;"),
        ):
            assert _partition_removed(analyze_structural_changes(before, after)), before

    def test_select_star_added_after_a_temp_function(self):
        findings = analyze_structural_changes(
            "CREATE TEMP FUNCTION f(x INT64) AS (x + 1); SELECT f(id) FROM raw.events",
            "CREATE TEMP FUNCTION f(x INT64) AS (x + 1); SELECT * FROM raw.events",
        )
        assert any(f.description == "SELECT * added" for f in findings)

    def test_a_statement_added_in_front_changes_nothing(self):
        findings = analyze_structural_changes(
            f"SELECT id FROM raw.events WHERE {D}",
            f"DECLARE d DATE DEFAULT CURRENT_DATE(); SELECT id FROM raw.events WHERE {D}",
        )
        assert not _blocking(findings)

    def test_trailing_semicolons_and_comments_change_nothing(self):
        assert analyze_structural_changes("SELECT id FROM t WHERE x = 1;", "SELECT id FROM t WHERE x = 1") == []
        assert analyze_structural_changes(
            "SELECT id FROM t WHERE x = 1; -- end\n", "-- c\nSELECT id FROM t WHERE x = 1;;"
        ) == []

    def test_a_script_with_nothing_to_analyze_is_skipped(self):
        for before, after in (
            ("DECLARE d DATE DEFAULT CURRENT_DATE()", "DECLARE d DATE DEFAULT CURRENT_DATE(); CALL p.d.proc()"),
            ("SELECT id FROM t", "CALL p.d.proc()"),
        ):
            findings = analyze_structural_changes(before, after)
            assert findings and all(f.skipped for f in findings), after
            assert findings[0].detail.startswith("nothing to analyze"), after

    def test_unparseable_sql_says_so(self):
        findings = analyze_structural_changes("SELECT id FROM t", "SELECT FROM WHERE (((")
        assert findings[0].detail.startswith("sqlglot parse error")

    BLOCK = ("BEGIN\n  DECLARE n INT64 DEFAULT 1;\n  SELECT id FROM p.d.events WHERE event_date >= '2024-01-01';\n"
             "  SELECT id FROM p.d.users WHERE status = 'a';\nEND;")

    def test_the_first_statement_of_a_begin_block_is_read(self):
        """sqlglot keeps `BEGIN` with the text up to the first `;`: removing the DECLARE above
        a query must not hide the query."""
        assert analyze_structural_changes(self.BLOCK, self.BLOCK.replace("  DECLARE n INT64 DEFAULT 1;\n", "")) == []

    def test_a_filter_dropped_inside_a_begin_block_is_removed(self):
        before = self.BLOCK.replace("  DECLARE n INT64 DEFAULT 1;\n", "")
        after = before.replace(" WHERE event_date >= '2024-01-01'", "")
        assert _partition_removed(analyze_structural_changes(before, after))

    def test_statements_inside_if_and_exception_blocks_are_read(self):
        for before, after in (
            (f"IF a THEN SELECT id FROM p.d.x WHERE {D}; END IF;", "IF a THEN SELECT id FROM p.d.x; END IF;"),
            (f"BEGIN SELECT 1; EXCEPTION WHEN ERROR THEN SELECT id FROM p.d.x WHERE {D}; END;",
             "BEGIN SELECT 1; EXCEPTION WHEN ERROR THEN SELECT id FROM p.d.x; END;"),
        ):
            assert _partition_removed(analyze_structural_changes(before, after)), before


class TestDeepQueries:
    """Long UNION ALL and AND chains are deep trees: no check may recurse once per branch."""

    def test_a_long_union_all_is_analyzed(self):
        import sys

        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(1000)  # Python's default
        try:
            for projection in ("id", "*"):
                sql = " UNION ALL ".join(f"SELECT {projection} FROM p.d.t{i}" for i in range(1500))
                assert analyze_structural_changes(sql, sql) == [], projection
        finally:
            sys.setrecursionlimit(limit)

    def test_deeply_nested_subqueries_are_skipped_not_raised(self):
        """sqlglot's parser recurses once per nested subquery: past its depth the version is
        "could not parse" (skipped), never an exception."""
        n = 300
        before = "select * from (" * n + "select id from `p`.`raw`.`o`" + ")" * n
        after = "select * from (" * n + "select id, amount from `p`.`raw`.`o`" + ")" * n
        findings = analyze_structural_changes(before, after)
        assert findings and all(f.skipped for f in findings)

    def test_a_long_and_chain_is_analyzed(self):
        sql = "SELECT id FROM t WHERE " + " AND ".join(f"c{i} = {i}" for i in range(1500))
        assert analyze_structural_changes(sql, sql) == []
        findings = analyze_structural_changes(sql, sql.replace(" AND c7 = 7", ""))
        assert _blocking(findings) == [("WHERE clause removed", "medium")]


def _label(value):
    """Test ids from a (label, before, after) row: the label, and nothing for the SQL."""
    return value if not value.lstrip().lower().startswith(("select", "with")) else ""


class TestStarReachingTheResult:
    """`SELECT * added` is about the tables whose every column reaches the result through `*`."""

    def _star(self, before, after):
        findings = analyze_structural_changes(before, after)
        assert not any(f.skipped for f in findings)
        return any(f.description == "SELECT * added" for f in findings)

    def test_star_added_in_a_cte_that_an_outer_star_reads(self):
        assert self._star(
            "WITH base AS (SELECT id, dt FROM big_table) SELECT * FROM base",
            "WITH base AS (SELECT * FROM big_table) SELECT * FROM base",
        )

    def test_star_added_in_a_derived_table_that_an_outer_star_reads(self):
        assert self._star("SELECT * FROM (SELECT id FROM big) x", "SELECT * FROM (SELECT * FROM big) x")

    def test_star_over_a_cte_with_a_column_list_reads_no_more_columns(self):
        assert not self._star("SELECT id FROM orders", "WITH s AS (SELECT id FROM orders) SELECT * FROM s")

    def test_a_pass_through_cte_is_no_star_added(self):
        assert not self._star(
            "WITH j AS (SELECT * FROM a) SELECT * FROM j",
            "WITH j AS (SELECT * FROM a), final AS (SELECT * FROM j) SELECT * FROM final",
        )

    def test_star_added_in_one_union_branch(self):
        assert self._star("SELECT id FROM a UNION ALL SELECT id FROM b", "SELECT id FROM a UNION ALL SELECT * FROM b")

    def test_star_added_to_a_create_table_as_select(self):
        assert self._star("CREATE TABLE x AS SELECT id FROM t", "CREATE TABLE x AS SELECT * FROM t")

    def test_unnest_sources_and_recursive_ctes_never_fail_the_analysis(self):
        ga4 = "SELECT * FROM `p.d.events`, UNNEST(event_params) AS ep, UNNEST(items) AS it"
        assert not self._star(ga4, f"{ga4} WHERE ep.key = 'x'")
        recursive = "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT {} FROM r"
        self._star(recursive.format("n"), recursive.format("*"))

    def test_swapped_table_under_an_unchanged_star_is_a_new_table_not_a_star(self):
        findings = analyze_structural_changes("SELECT * FROM p.d.a", "SELECT * FROM p.d.b")
        assert [f.description for f in findings] == ["New table reference"]

    @pytest.mark.parametrize("label,before,after", [
        ("the row as a function argument",
         "WITH c AS (SELECT id, amount FROM p.d.orders) SELECT TO_JSON_STRING(c) AS j FROM c",
         "WITH c AS (SELECT * FROM p.d.orders) SELECT TO_JSON_STRING(c) AS j FROM c"),
        ("the row as a value", "WITH c AS (SELECT id FROM p.d.orders) SELECT c FROM c",
         "WITH c AS (SELECT * FROM p.d.orders) SELECT c FROM c"),
        ("the row of a table read through its name", "SELECT id FROM p.d.orders",
         "SELECT TO_JSON_STRING(orders) AS j FROM p.d.orders"),
        ("dbt_utils.deduplicate: the row aggregated, then the struct starred",
         "SELECT id, amount FROM p.d.orders QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY updated_at DESC) = 1",
         "SELECT unique.* FROM (SELECT ARRAY_AGG(original ORDER BY updated_at DESC LIMIT 1)[OFFSET(0)] unique "
         "FROM p.d.orders original GROUP BY id)"),
        ("a column named like the CTE is its row (BigQuery prefers the source's name)",
         "with source as (select * from p.d.raw), renamed as (select id from source) select * from renamed",
         "with source as (select * from p.d.raw), renamed as (select id, source from source) select * from renamed"),
    ], ids=_label)
    def test_a_whole_row_read_counts_like_a_star(self, label, before, after):
        assert self._star(before, after), label

    @pytest.mark.parametrize("label,before,after", [
        ("CTE join refactor",
         "with orders as (select * from `p`.`d`.`orders`), customers as (select * from `p`.`d`.`customers`), "
         "final as (select orders.*, customers.name from orders left join customers using (customer_id)) select * from final",
         "with orders as (select * from `p`.`d`.`orders`), customers as (select customer_id, name from `p`.`d`.`customers`), "
         "final as (select o.*, c.name, o.discount from orders o left join customers c using (customer_id)) select * from final"),
        ("import CTE with QUALIFY", "select * from `p`.`d`.`orders`",
         "with orders as (select * from `p`.`d`.`orders`), final as (select * except (legacy) from orders "
         "qualify row_number() over (partition by id order by ts desc) = 1) select * from final"),
        ("UNION through CTEs", "select * from `p`.`d`.`a` union all select * from `p`.`d`.`b`",
         "with a as (select * from `p`.`d`.`a`), b as (select * from `p`.`d`.`b`) select * from a union all select * from b"),
        ("import CTE read by named columns", "select o.id, o.amount from `p`.`d`.`orders` o",
         "with orders as (select * from `p`.`d`.`orders`) select id, amount, discount from orders"),
        ("parenthesized join", "SELECT * FROM (`p`.`d`.`a` JOIN `p`.`d`.`b` USING (id))",
         "SELECT * FROM `p`.`d`.`a` JOIN `p`.`d`.`b` USING (id)"),
        ("dbt_utils.deduplicate unchanged",
         "SELECT unique.* FROM (SELECT ARRAY_AGG(original ORDER BY u DESC LIMIT 1)[OFFSET(0)] unique FROM p.d.o original GROUP BY id)",
         "SELECT unique.* FROM (SELECT ARRAY_AGG(original ORDER BY u DESC LIMIT 1)[OFFSET(0)] unique FROM p.d.o original GROUP BY id)"),
    ], ids=_label)
    def test_refactors_that_read_the_same_columns_have_no_risk(self, label, before, after):
        assert not _blocking(analyze_structural_changes(before, after)), label

    @pytest.mark.parametrize("label,before,after", [
        ("a star over a CTE that lists more columns than the result took",
         "WITH s AS (SELECT id, name, email, payload FROM p.d.users) SELECT id, name FROM s",
         "WITH s AS (SELECT id, name, email, payload FROM p.d.users) SELECT * FROM s"),
        ("a star over a derived table that lists more columns than the result took",
         "SELECT id FROM (SELECT id, name, payload FROM p.d.users)",
         "SELECT * FROM (SELECT id, name, payload FROM p.d.users)"),
        ("a star next to a named column over a wider CTE",
         "WITH s AS (SELECT id, name, email FROM p.d.users) SELECT id FROM s",
         "WITH s AS (SELECT id, name, email FROM p.d.users) SELECT *, 1 AS x FROM s"),
        ("named columns of a join become a star over a wider CTE",
         "SELECT a.id, b.x FROM p.d.a a JOIN p.d.b b USING (id)",
         "WITH j AS (SELECT a.id, b.x, b.y FROM p.d.a a JOIN p.d.b b USING (id)) SELECT * FROM j"),
        ("a star inside an ARRAY subquery of the SELECT list",
         "SELECT o.id, ARRAY(SELECT AS STRUCT i.sku FROM p.d.items i WHERE i.order_id = o.id) AS items FROM p.d.orders o",
         "SELECT o.id, ARRAY(SELECT AS STRUCT * FROM p.d.items i WHERE i.order_id = o.id) AS items FROM p.d.orders o"),
        ("a star over a table function", "SELECT id FROM p.d.tvf(1)", "SELECT * FROM p.d.tvf(1)"),
        ("a star over an unqualified table function", "SELECT id FROM tvf(1)", "SELECT * FROM tvf(1)"),
        ("a second table function of the same dataset starred",
         "SELECT a.* FROM p.d.tvf_a(1) a JOIN p.d.tvf_b(1) b USING (id)",
         "SELECT a.*, b.* FROM p.d.tvf_a(1) a JOIN p.d.tvf_b(1) b USING (id)"),
        ("a list with a struct's fields, read through a star", "SELECT event_date FROM p.ga4.events",
         "WITH x AS (SELECT event_date, user_id, device.* FROM p.ga4.events) SELECT * FROM x"),
        ("a list with an UNNEST alias's fields, read through a star", "SELECT t.id FROM p.d.t t",
         "WITH x AS (SELECT t.id, t.name, t.payload, e.* FROM p.d.t t, UNNEST(t.items) e) SELECT * FROM x"),
        ("a list of two tables' columns of the same names",
         "SELECT a.id, a.name FROM p.d.a a JOIN p.d.b b ON a.k = b.k",
         "WITH s AS (SELECT a.id, a.name, b.id AS b_id, b.name AS b_name FROM p.d.a a JOIN p.d.b b ON a.k = b.k) "
         "SELECT * FROM s"),
        ("one CTE read by two stars whose EXCEPTs leave out different columns",
         "WITH s AS (SELECT id, name, notes FROM p.d.u) SELECT id, name FROM s",
         "WITH s AS (SELECT id, name, notes FROM p.d.u) SELECT a.* EXCEPT (notes), b.* EXCEPT (id, name) "
         "FROM s a JOIN s b ON a.id = b.id"),
    ], ids=_label)
    def test_a_star_that_reads_more_columns_than_before(self, label, before, after):
        assert self._star(before, after), label

    @pytest.mark.parametrize("label,before,after", [
        ("named columns become a staging model with the same list", "select id, name from p.d.raw",
         "with source as (select * from p.d.raw), renamed as (select id, name from source) select * from renamed"),
        ("a CTE of constants", "SELECT 1 AS a", "WITH s AS (SELECT 1 AS a, 2 AS b, 3 AS c) SELECT * FROM s"),
        ("a star whose EXCEPT leaves out what the result didn't take",
         "WITH s AS (SELECT id, name, _loaded_at FROM p.d.users) SELECT id, name FROM s",
         "WITH s AS (SELECT id, name, _loaded_at FROM p.d.users) SELECT * EXCEPT (_loaded_at) FROM s"),
        ("the same through an import CTE",
         "with source as (select * from p.d.raw), renamed as (select id, name, _synced from source) select id, name from renamed",
         "with source as (select * from p.d.raw), renamed as (select id, name, _synced from source) "
         "select * except (_synced) from renamed"),
        ("derived entries and constants read no more columns", "SELECT id, amount FROM p.d.orders",
         "WITH final AS (SELECT id, amount, amount * 100 AS amount_cents, CURRENT_TIMESTAMP() AS loaded_at "
         "FROM p.d.orders) SELECT * FROM final"),
        ("a MERGE's USING moved into a CTE",
         "MERGE p.d.x T USING (SELECT id, v FROM p.d.s) S ON T.id = S.id WHEN MATCHED THEN UPDATE SET v = S.v",
         "MERGE p.d.x T USING (WITH c AS (SELECT id, v FROM p.d.s) SELECT * FROM c) S ON T.id = S.id "
         "WHEN MATCHED THEN UPDATE SET v = S.v"),
        ("the join key the result already read",
         "WITH o AS (SELECT * FROM p.d.orders), c AS (SELECT * FROM p.d.customers) "
         "SELECT o.*, c.name FROM o JOIN c USING (customer_id)",
         "WITH o AS (SELECT * FROM p.d.orders), c AS (SELECT customer_id, name FROM p.d.customers) "
         "SELECT o.* EXCEPT (customer_id), c.* FROM o JOIN c USING (customer_id)"),
    ], ids=_label)
    def test_a_star_over_a_list_no_wider_than_the_result_is_no_star_added(self, label, before, after):
        assert not self._star(before, after), label

    @pytest.mark.parametrize("label,before,after", [
        ("a staging model's column list grows",
         "with source as (select * from p.d.raw), renamed as (select id, name from source) select * from renamed",
         "with source as (select * from p.d.raw), renamed as (select id, name, email from source) select * from renamed"),
        ("a derived table's list grows", "SELECT * FROM (SELECT id, name FROM p.d.t)",
         "SELECT * FROM (SELECT id, name, email FROM p.d.t)"),
        ("a CTE renamed", "WITH a AS (SELECT id, x, y FROM p.d.t) SELECT * FROM a",
         "WITH b AS (SELECT id, x, y FROM p.d.t) SELECT * FROM b"),
        ("a table's star narrowed to a CTE's list", "SELECT * FROM p.d.users",
         "WITH s AS (SELECT id, name, email FROM p.d.users) SELECT * FROM s"),
    ], ids=_label)
    def test_a_column_list_changed_under_a_star_the_base_had_is_no_star_added(self, label, before, after):
        """A star the base version already had reaches the same tables: a list that grows under
        it is a wider SELECT (priced by the dry run), not a new star."""
        assert not self._star(before, after), label

    @pytest.mark.parametrize("label,before,after", [
        ("under an import CTE",
         "WITH source AS (SELECT * FROM `p`.`ga4`.`events`) SELECT event_date, device.category FROM source",
         "WITH source AS (SELECT * FROM `p`.`ga4`.`events`) SELECT event_date, device.* FROM source"),
        ("from the table", "SELECT event_date, device.category FROM `p`.`ga4`.`events`",
         "SELECT event_date, device.* FROM `p`.`ga4`.`events`"),
    ], ids=_label)
    def test_a_struct_columns_fields_read_only_that_column(self, label, before, after):
        assert not self._star(before, after), label

    @pytest.mark.parametrize("label,before,after", [
        ("through its alias",
         "SELECT product, Q1 FROM p.d.sales PIVOT(SUM(amount) FOR quarter IN ('Q1', 'Q2')) AS pv",
         "SELECT pv.* FROM p.d.sales PIVOT(SUM(amount) FOR quarter IN ('Q1', 'Q2')) AS pv"),
        ("through a bare star", "SELECT product, Q1 FROM p.d.sales PIVOT(SUM(amount) FOR quarter IN ('Q1', 'Q2'))",
         "SELECT * FROM p.d.sales PIVOT(SUM(amount) FOR quarter IN ('Q1', 'Q2'))"),
    ], ids=_label)
    def test_a_pivoted_table_is_read_in_full_in_both_versions(self, label, before, after):
        """PIVOT groups by every input column it doesn't aggregate or pivot: the table is read in
        full whatever the result takes, so a star over the pivot reads nothing more."""
        assert not self._star(before, after), label

    def test_a_star_inside_exists_only_tests_rows(self):
        assert not self._star("SELECT id FROM p.d.t", "SELECT id, EXISTS(SELECT * FROM p.d.t WHERE id > 0) AS e FROM p.d.t")


class TestOutermostColumnCount:
    def test_union_counts_its_first_select(self):
        findings = analyze_structural_changes(
            "SELECT a FROM t1 UNION ALL SELECT x FROM (SELECT x FROM t2)",
            "SELECT a, b, c, d, e, f FROM t1 UNION ALL SELECT x, x, x, x, x, x FROM (SELECT x FROM t2)",
        )
        assert any(f.description.startswith("SELECT expanded by 5 columns") for f in findings)

    def test_create_table_as_counts_the_query_it_stores_not_a_subquery(self):
        findings = analyze_structural_changes(
            "CREATE TABLE x AS SELECT a FROM (SELECT a FROM t)",
            "CREATE TABLE x AS SELECT a FROM (SELECT a, b, c, d FROM t)",
        )
        assert not any(f.description.startswith("SELECT expanded") for f in findings)


class TestUnnestStarInsideAParenthesizedJoin:
    def test_star_of_an_unnest_alias_in_a_parenthesized_join_reads_no_table(self):
        findings = analyze_structural_changes(
            "SELECT t.id FROM (p.d.t t JOIN UNNEST(t.items) i ON TRUE)",
            "SELECT t.id, i.* FROM (p.d.t t JOIN UNNEST(t.items) i ON TRUE)",
        )
        assert not any(f.description == "SELECT * added" for f in findings)
