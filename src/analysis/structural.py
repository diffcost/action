"""Structural SQL analysis using sqlglot AST comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# Column names that are almost always partition / date filters.
_PARTITION_EXACT = frozenset({
    "_partitiontime", "_partitiondate", "_partition_date", "_table_suffix",
    "created_at", "updated_at", "event_time", "order_date", "ingestion_time", "load_date",
})
# Whole-word tokens (split on "_") that mark a date-like column, e.g. event_date, dt, event_ts.
# Matching whole tokens avoids false positives such as `candidate_id` or `is_validated`.
_PARTITION_TOKENS = frozenset({"date", "dt", "timestamp", "ts", "datetime", "partitiontime"})
# Weaker tokens (`response_time_ms`, `day_of_week`, `hour_bucket` are rarely partitions):
# these count only when the predicate also compares against a date/time value.
_WEAK_PARTITION_TOKENS = frozenset({"time", "day", "hour"})
_DATE_VALUE_RE = re.compile(
    r"'\d{4}-\d{2}-\d{2}|\b(current_date|current_timestamp|current_datetime|date|timestamp|datetime"
    r"|date_sub|date_add|timestamp_sub|timestamp_add|date_trunc|timestamp_trunc|parse_date|parse_timestamp)\s*\(",
    re.IGNORECASE,
)


@dataclass
class Finding:
    """A structural change finding with risk assessment."""

    description: str  # plain text, no SQL
    risk: str  # "high", "medium", "low"
    detail: str = ""
    subject: str = ""  # SQL snippet or table name the finding is about (rendered as code)
    # True when this finding says the analysis could not run (unparseable SQL, a failed
    # check): "no risk found" can't be concluded from it
    skipped: bool = False
    # Tables whose partition column the removed filter was on, from table metadata (e.g.
    # "analytics.events (4.20 TiB table, partitioned daily)"). Empty when the partition
    # filter is a guess from the column name.
    partition_tables: tuple[str, ...] = ()
    # True when the PR keeps another filter that prunes on the removed filter's column(s), in
    # the same WHERE clause and on the same table qualifier: the query reads more partitions,
    # not necessarily all of them
    partition_filter_kept: bool = False


def analyze_structural_changes(
    production_sql: str,
    pr_sql: str,
    partition_columns: dict[str, list[str]] | None = None,
    cluster_columns: dict[str, list[str]] | None = None,
) -> list[Finding]:
    """
    Parse both SQL versions into ASTs and detect cost-relevant structural changes.
    Returns list of findings ordered by risk (high first).

    From table metadata (lowercased column names; columns are matched by name, without
    resolving aliases):
    - `partition_columns`: partition column -> labels of the tables it partitions. A
      removed filter on one is a confirmed partition filter removal.
    - `cluster_columns`: clustering column -> the tables it clusters. A removed filter on
      one that doesn't look like a partition filter gets its own finding: the dry run can't
      show what cluster pruning saved.
    Any other removed filter on a date-like column counts as a removed partition filter.
    """
    try:
        prod_ast = sqlglot.parse_one(production_sql, dialect="bigquery")
    except SqlglotError:
        return [Finding("Could not parse production SQL", "low", "sqlglot parse error — skipping structural analysis", skipped=True)]

    try:
        pr_ast = sqlglot.parse_one(pr_sql, dialect="bigquery")
    except SqlglotError:
        return [Finding("Could not parse PR SQL", "low", "sqlglot parse error — skipping structural analysis", skipped=True)]

    try:
        findings = _run_checks(prod_ast, pr_ast, partition_columns, cluster_columns)
    except Exception as e:  # never let a sqlglot edge case fail the run
        return [Finding("Structural analysis failed", "low", f"{type(e).__name__} — skipping structural analysis", skipped=True)]

    return findings


def _run_checks(
    prod_ast: exp.Expression,
    pr_ast: exp.Expression,
    partition_columns: dict[str, list[str]] | None = None,
    cluster_columns: dict[str, list[str]] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []

    # Check each pattern
    findings.extend(_check_partition_filter_removal(prod_ast, pr_ast, partition_columns, cluster_columns))
    findings.extend(_check_select_star(prod_ast, pr_ast))
    findings.extend(_check_join_changes(prod_ast, pr_ast))
    findings.extend(_check_where_clause_changes(prod_ast, pr_ast))
    findings.extend(_check_table_references(prod_ast, pr_ast))
    findings.extend(_check_subquery_changes(prod_ast, pr_ast))
    findings.extend(_check_window_functions(prod_ast, pr_ast))
    findings.extend(_check_column_expansion(prod_ast, pr_ast))

    findings = _dedupe(findings)

    # Sort: high risk first
    risk_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: risk_order.get(f.risk, 3))

    return findings


def _check_partition_filter_removal(
    prod: exp.Expression,
    pr: exp.Expression,
    partition_columns: dict[str, list[str]] | None = None,
    cluster_columns: dict[str, list[str]] | None = None,
) -> list[Finding]:
    """Detect removed WHERE conditions that filter on partition columns: confirmed from table
    metadata when it has them, otherwise guessed from the column name (see
    analyze_structural_changes).

    The WHERE conditions of the whole query (CTEs and subqueries included) are compared one
    copy for one copy. A condition the PR has as often as the base version is kept, wherever
    it now sits, so a filter moved between a CTE and the SELECT that reads it is no change; one
    the PR has fewer times is removed, even if the same text stays in another subquery. A
    removed copy is a changed filter, not a removal, when the PR adds a condition on the same
    column in its place (with the same qualifier when both are qualified): on a partition
    column (from metadata, or by its name) a comparison that prunes partitions only by another
    one that does, any other condition by any condition on that column. Each added condition
    stands in for one removed copy, a BETWEEN for two (a lower and an upper bound)."""
    known = partition_columns or {}
    clustering = cluster_columns or {}
    base, head = _condition_copies(prod), _condition_copies(pr)
    _match_kept(base, head)
    removed = [c for c in base if not c.matched]
    # Only a condition the base version doesn't have can stand in for a removed one: an extra
    # copy of an existing filter in another subquery doesn't replace the one dropped here
    base_texts = {c.text for c in base}
    added = [c for c in head if not c.matched and c.text not in base_texts]
    _match_changed(removed, added, known)

    by_text: dict[str, list[_Copy]] = {}
    for copy in removed:
        if not copy.changed:
            by_text.setdefault(copy.text, []).append(copy)

    findings = []
    for pred, copies in by_text.items():
        pred_columns = {c.name.lower() for c in _outer_columns(copies[0].cond)}

        confirmed = sorted(col for col in pred_columns if col in known)
        if confirmed:
            tables = tuple(dict.fromkeys(label for col in confirmed for label in known[col]))
            kept = _filter_kept(copies, set(confirmed))
            effect = (
                f"The query still has another filter on {' and '.join(confirmed)}, so without this one it reads "
                "more partitions, not necessarily all of them."
                if kept else "Without this filter the query reads every partition."
            )
            findings.append(
                Finding(
                    "Partition filter removed",
                    "high",
                    f"{' and '.join(confirmed)} {'is' if len(confirmed) == 1 else 'are'} the partition column of "
                    f"{', '.join(tables)}. {effect}",
                    subject=_truncate(pred, 80),
                    partition_tables=tables,
                    partition_filter_kept=kept,
                )
            )
            continue

        is_likely_partition = _names_partition_column(pred, pred_columns)

        clustered = sorted(col for col in pred_columns if col in clustering)
        if clustered and not is_likely_partition:
            tables = tuple(dict.fromkeys(t for col in clustered for t in clustering[col]))
            findings.append(
                Finding(
                    "Cluster filter removed",
                    "medium",
                    f"{' and '.join(clustered)} {'clusters' if len(clustered) == 1 else 'cluster'} "
                    f"{', '.join(tables)}. BigQuery skips clustered blocks only when the query runs, so the "
                    "dry run can't show what this filter saved: the real increase can be larger than the estimate.",
                    subject=_truncate(pred, 80),
                )
            )
            continue

        if is_likely_partition:
            kept = _filter_kept(copies, pred_columns)
            findings.append(
                Finding(
                    "Partition filter removed",
                    "high",
                    (
                        "The query still has another filter on this column, so without this one it reads more of "
                        "the table, not necessarily all of it."
                        if kept else
                        "Removing a partition filter causes a full table scan. "
                        "This is the most common cause of BQ cost explosions."
                    ),
                    subject=_truncate(pred, 80),
                    partition_filter_kept=kept,
                )
            )
        else:
            findings.append(
                Finding(
                    "WHERE clause removed",
                    "medium",
                    "Broadening query scope may increase bytes scanned.",
                    subject=_truncate(pred, 80),
                )
            )

    return findings


def _names_partition_column(text: str, columns: set[str]) -> bool:
    """Whether a condition's column names look like a partition / date filter column."""
    lower = text.lower()
    has_date_value = bool(_DATE_VALUE_RE.search(text)) or "current_date" in lower
    return any(_is_partition_like(col, has_date_value) for col in columns) or any(
        kw in lower for kw in ("_partitiontime", "_partitiondate")
    )


@dataclass(eq=False)
class _Copy:
    """One condition of one WHERE clause."""

    cond: exp.Expression
    clause: list  # the copies of its WHERE clause, itself included
    text: str = ""
    matched: bool = False  # the other version has the same condition (one copy for one copy)
    changed: bool = False  # an added condition on the same column stands in for it

    def __post_init__(self):
        self.text = self.cond.sql(dialect="bigquery")


def _condition_copies(ast: exp.Expression) -> list[_Copy]:
    """Every condition of every WHERE clause in the query (CTEs, subqueries, UPDATE, DELETE)."""
    copies: list[_Copy] = []
    for where in ast.find_all(exp.Where):
        clause: list[_Copy] = []
        clause.extend(_Copy(cond, clause) for cond in _split_and_predicates(where.this))
        copies.extend(clause)
    return copies


def _match_kept(base: list[_Copy], head: list[_Copy]) -> None:
    """Pair each base condition with one PR copy of the same text: first the copies whose WHERE
    clauses are most alike (shared conditions over the larger clause), so a filter dropped
    from one clause is the one left unpaired."""
    pairs = []
    for b in base:
        b_texts = {c.text for c in b.clause}
        for h in head:
            if h.text == b.text:
                h_texts = {c.text for c in h.clause}
                pairs.append((len(b_texts & h_texts) / max(len(b_texts), len(h_texts)), b, h))
    for _score, b, h in sorted(pairs, key=lambda p: -p[0]):
        if not b.matched and not h.matched:
            b.matched = h.matched = True


def _match_changed(removed: list[_Copy], added: list[_Copy], known: dict) -> None:
    """Mark each removed copy that an added condition stands in for (one for one; a BETWEEN for
    two conditions of one WHERE clause, a lower and an upper bound): the same kind of condition
    first (`>=` for `>=`), then any."""
    capacity = [2 if isinstance(_unwrap(a.cond), exp.Between) else 1 for a in added]
    first_clause: dict[int, list] = {}  # a BETWEEN's second slot: only the same base clause
    for same_kind in (True, False):
        for copy in removed:
            if copy.changed:
                continue
            for i, other in enumerate(added):
                if not capacity[i] or (i in first_clause and first_clause[i] is not copy.clause):
                    continue
                if same_kind and type(_unwrap(other.cond)) is not type(_unwrap(copy.cond)):
                    continue
                if _stands_in_for(other, copy, known):
                    capacity[i] -= 1
                    first_clause.setdefault(i, copy.clause)
                    copy.changed = True
                    break


def _stands_in_for(new: _Copy, old: _Copy, known: dict) -> bool:
    """Whether condition `new` replaces `old` on the same column: on a partition column, a
    comparison that prunes partitions only by another one; any other condition by any
    condition on that column."""
    old_column = _pruning_column(old.cond)
    if old_column is not None and (
        old_column.name.lower() in known or _names_partition_column(old.text, {old_column.name.lower()})
    ):
        new_column = _pruning_column(new.cond)
        return new_column is not None and _same_column(old_column, new_column)
    old_columns, new_columns = _outer_columns(old.cond), _outer_columns(new.cond)
    if not old_columns and not new_columns:  # e.g. an edited EXISTS (...)
        return type(_unwrap(old.cond)) is type(_unwrap(new.cond))
    return any(_same_column(a, b) for a in old_columns for b in new_columns)


def _same_column(old: exp.Column, new: exp.Column) -> bool:
    """Same name, and when both are qualified, qualifiers that can stand for the same table: a
    different qualifier counts as another table only when the PR's SELECT also reads a source
    under the old one (`a.dt` -> `b.dt` in `FROM a JOIN b`), not when the filter moved to a
    SELECT that names its table differently (a CTE's reader `x.dt` -> the CTE's own `e.dt`)."""
    if old.name.lower() != new.name.lower():
        return False
    if not old.table or not new.table or old.table.lower() == new.table.lower():
        return True
    return old.table.lower() not in _source_names(new.find_ancestor(exp.Select))


def _source_names(select: exp.Select | None) -> set[str]:
    """The names (aliases, or table names) of the sources a SELECT reads directly."""
    if select is None:
        return set()
    names = set()
    for source in [select.args.get("from"), *(select.args.get("joins") or [])]:
        node = source.this if source is not None else None
        if isinstance(node, (exp.Table, exp.Subquery)) and node.alias_or_name:
            names.add(node.alias_or_name.lower())
    return names


def _filter_kept(copies: list[_Copy], columns: set[str]) -> bool:
    """Whether the PR still prunes on the `columns` (lowercased names) of the removed
    condition in every WHERE clause it was removed from: each of those columns, written with
    the same table qualifier, is compared with a value (bare or inside a function BigQuery
    prunes on, see _pruning_column) in another condition of that clause that the PR keeps (or
    changes into another such comparison).
    A same-named column of another table (`a.dt` next to `b.dt`), or a condition that doesn't
    prune (`dt IS NOT NULL`, an OR, a function such as `EXTRACT(MONTH FROM dt)`, a join
    condition), doesn't count."""
    for copy in copies:
        removed_refs = {_ref(c) for c in _outer_columns(copy.cond) if c.name.lower() in columns}
        kept_refs = {
            _ref(column)
            for other in copy.clause
            if other is not copy and (other.matched or other.changed)
            for column in [_pruning_column(other.cond)]
            if column is not None
        }
        if not removed_refs or not removed_refs <= kept_refs:
            return False
    return bool(copies)


_COMPARISONS = (exp.EQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
# Functions of a partition column that BigQuery still prunes partitions on, when their other
# arguments are constant. BigQuery's "Query partitioned tables" page lists DATE_ADD, DATE_SUB,
# DATE_DIFF, DATE_TRUNC, DATETIME_DIFF, TIMESTAMP_ADD, TIMESTAMP_SUB, TIMESTAMP_DIFF,
# TIMESTAMP_TRUNC, EXTRACT with the YEAR part (DATE or YEAR on a timestamp) and FORMAT_TIMESTAMP
# with %F, %Y-%m-%d or %Y%m%d. DATE(), a CAST to a date or time type and DATETIME_TRUNC count too.
# The column is the function's first argument:
_PRUNABLE_WRAPPERS = (exp.Date, exp.TimestampTrunc, exp.DateTrunc, exp.DatetimeTrunc)
_PRUNABLE_CASTS = ("date", "timestamp", "timestamptz", "datetime")
# ... the first argument, and the amount added or subtracted is constant:
_PRUNABLE_SHIFTS = (exp.DateAdd, exp.DateSub, exp.TimestampAdd, exp.TimestampSub)
# ... either argument, and the other one is constant:
_PRUNABLE_DIFFS = (exp.DateDiff, exp.TimestampDiff, exp.DatetimeDiff)
# The column's type isn't known here, so both parts count on any column
_PRUNABLE_EXTRACT_PARTS = ("year", "date")
_PRUNABLE_TIMESTAMP_FORMATS = ("%F", "%Y-%m-%d", "%Y%m%d")


def _is_constant(value: exp.Expression | None) -> bool:
    """A value with no column or subquery in it: a literal, a parameter, CURRENT_DATE()."""
    return value is not None and value.find(exp.Column, exp.Select, exp.Subquery) is None


def _pruned_argument(side: exp.Expression) -> exp.Expression | None:
    """The argument BigQuery still prunes on when `side` is one of the functions above (with
    its other arguments constant), or a parenthesized expression; None for anything else."""
    if isinstance(side, exp.Paren):
        return side.this
    if isinstance(side, exp.Cast):  # TRY_CAST / SAFE_CAST included
        return side.this if side.to.is_type(*_PRUNABLE_CASTS) else None
    if isinstance(side, _PRUNABLE_WRAPPERS):  # with a constant time zone, if any
        zone = side.args.get("zone")
        return side.this if zone is None or _is_constant(zone) else None
    if isinstance(side, _PRUNABLE_SHIFTS):
        return side.this if _is_constant(side.expression) else None
    if isinstance(side, _PRUNABLE_DIFFS):
        if _is_constant(side.expression):
            return side.this
        return side.expression if _is_constant(side.this) else None
    if isinstance(side, exp.Extract):
        return side.expression if side.this.name.lower() in _PRUNABLE_EXTRACT_PARTS else None
    # FORMAT_TIMESTAMP (FORMAT_DATE and FORMAT_DATETIME, which don't prune, parse to the same
    # node over a TsOrDsToDate or TsOrDsToDatetime; the parser drops a time zone argument)
    if isinstance(side, exp.TimeToStr) and isinstance(side.this, exp.TsOrDsToTimestamp):
        fmt = side.args.get("format")
        if isinstance(fmt, exp.Literal) and fmt.is_string and fmt.this in _PRUNABLE_TIMESTAMP_FORMATS:
            return side.this.this
    return None


def _unwrap(cond: exp.Expression) -> exp.Expression:
    while isinstance(cond, exp.Paren):
        cond = cond.this
    return cond


def _ref(column: exp.Column) -> str:
    """A column as written, with its qualifier (lowercased)."""
    return column.sql(dialect="bigquery").lower()


def _outer_columns(expr: exp.Expression) -> list[exp.Column]:
    """The columns a condition names itself (a bare `is_active` included), not those inside a
    subquery it contains."""
    found = []
    for column in expr.find_all(exp.Column):
        node = column
        while node is not None and node is not expr and not isinstance(node, (exp.Select, exp.Subquery)):
            node = node.parent
        if node is None or node is expr:
            found.append(column)
    return found


def _pruning_column(cond: exp.Expression) -> exp.Column | None:
    """The column `cond` compares with a value in a way BigQuery prunes partitions on:
    `dt >= '2024-01-01'`, `DATE(ts) = @day`, `e.dt BETWEEN a AND b`, `dt IN (...)`, or the
    column inside one of the functions above, such as `EXTRACT(YEAR FROM dt) = 2024` or
    `DATE_DIFF(CURRENT_DATE(), dt, DAY) <= 7`. None for anything else: an OR, NOT, IS NULL, a
    comparison with another column or a subquery, `dt IN UNNEST(...)`, another function such
    as `EXTRACT(MONTH FROM dt)` or `FORMAT_DATE(...)`, or a listed one with a column among
    its other arguments."""
    cond = _unwrap(cond)
    if isinstance(cond, _COMPARISONS):
        candidates = [(cond.left, [cond.right]), (cond.right, [cond.left])]
    elif isinstance(cond, exp.Between):
        candidates = [(cond.this, [cond.args.get("low"), cond.args.get("high")])]
    elif isinstance(cond, exp.In) and not cond.args.get("query") and not cond.args.get("unnest"):
        candidates = [(cond.this, list(cond.expressions))]
    else:
        return None
    for side, values in candidates:
        while side is not None and not isinstance(side, exp.Column):
            side = _pruned_argument(side)
        if isinstance(side, exp.Column) and all(_is_constant(v) for v in values):
            return side
    return None


def _is_partition_like(column: str, has_date_value: bool = True) -> bool:
    """Whether a column name looks like a partition / date filter column (whole-word match).

    `has_date_value`: the predicate compares against a date/time literal or function;
    required before weak tokens (time, day, hour) count.
    """
    col = column.lower()
    if col in _PARTITION_EXACT:
        return True
    tokens = [t for t in col.split("_") if t]
    if any(t in _PARTITION_TOKENS for t in tokens):
        return True
    return has_date_value and any(t in _WEAK_PARTITION_TOKENS for t in tokens)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop duplicate findings. A removed partition filter already explains a WHERE-count drop,
    and a new JOIN already reports the table it joins."""
    has_partition_removal = any(f.description.startswith("Partition filter removed") for f in findings)
    joined = {
        f.subject.lower().split(".")[-1]
        for f in findings
        if f.subject and (f.description.endswith("JOIN") or f.description == "CROSS JOIN added")
    }
    seen: set[tuple[str, str]] = set()
    result = []
    for f in findings:
        if has_partition_removal and (
            f.description == "All WHERE conditions removed" or f.description.endswith("WHERE condition(s) removed")
        ):
            continue
        if f.description == "New table reference" and f.subject.lower().split(".")[-1] in joined:
            continue
        key = (f.description, f.risk, f.subject)
        if key in seen:
            continue
        seen.add(key)
        result.append(f)
    return result


def _check_select_star(prod: exp.Expression, pr: exp.Expression) -> list[Finding]:
    """Detect SELECT * additions: a bare `*`, or a `t.*` for a table not starred before.

    A bare `*` already in the production version covers any qualified star the PR adds.
    """
    prod_stars = _star_sources(prod)
    added = set() if "" in prod_stars else _star_sources(pr) - prod_stars

    if added:
        return [
            Finding(
                "SELECT * added",
                "high",
                "SELECT * scans all columns. In columnar storage (BQ), this is significantly "
                "more expensive than selecting specific columns.",
            )
        ]
    return []


def _check_join_changes(prod: exp.Expression, pr: exp.Expression) -> list[Finding]:
    """Detect new JOINs, especially CROSS JOINs."""
    prod_joins = _extract_joins(prod)
    pr_joins = _extract_joins(pr)

    findings = []
    new_joins = pr_joins - prod_joins

    for join_type, table in new_joins:
        if "CROSS" in join_type.upper():
            findings.append(
                Finding(
                    "CROSS JOIN added",
                    "high",
                    "Cross joins produce a cartesian product. Cost grows multiplicatively.",
                    subject=table,
                )
            )
        else:
            findings.append(
                Finding(
                    f"New {join_type}" if "JOIN" in join_type else f"New {join_type} JOIN",
                    "medium",
                    "Additional joins increase data scanned.",
                    subject=table,
                )
            )

    return findings


def _check_where_clause_changes(prod: exp.Expression, pr: exp.Expression) -> list[Finding]:
    """Detect WHERE clause weakening (fewer predicates)."""
    prod_count = _count_where_predicates(prod)
    pr_count = _count_where_predicates(pr)

    if prod_count > 0 and pr_count == 0:
        return [
            Finding(
                "All WHERE conditions removed",
                "high",
                f"Production had {prod_count} filter condition(s). PR version has none.",
            )
        ]
    elif prod_count > pr_count > 0:
        diff = prod_count - pr_count
        return [
            Finding(
                f"{diff} WHERE condition(s) removed",
                "medium",
                f"Production had {prod_count} conditions, PR has {pr_count}. "
                "Fewer filters = more data scanned.",
            )
        ]
    return []


def _check_table_references(prod: exp.Expression, pr: exp.Expression) -> list[Finding]:
    """Detect new table references in FROM/JOIN clauses."""
    prod_tables = _extract_table_names(prod)
    pr_tables = _extract_table_names(pr)

    new_tables = pr_tables - prod_tables
    findings = []
    for table in new_tables:
        findings.append(
            Finding(
                "New table reference",
                "medium",
                "Additional table references increase the total data scanned.",
                subject=table,
            )
        )
    return findings


def _check_subquery_changes(prod: exp.Expression, pr: exp.Expression) -> list[Finding]:
    """Detect new subqueries or CTEs."""
    prod_ctes = len(list(prod.find_all(exp.CTE)))
    pr_ctes = len(list(pr.find_all(exp.CTE)))

    prod_subqueries = len(list(prod.find_all(exp.Subquery)))
    pr_subqueries = len(list(pr.find_all(exp.Subquery)))

    findings = []
    if pr_ctes > prod_ctes:
        findings.append(
            Finding(
                f"{pr_ctes - prod_ctes} new CTE(s) added",
                "low",
                "New CTEs may increase computation but don't necessarily increase bytes scanned.",
            )
        )
    if pr_subqueries > prod_subqueries:
        findings.append(
            Finding(
                f"{pr_subqueries - prod_subqueries} new subquery(ies) added",
                "low",
                "Additional subqueries may increase computation.",
            )
        )
    return findings


def _check_window_functions(prod: exp.Expression, pr: exp.Expression) -> list[Finding]:
    """Detect new window functions."""
    prod_windows = len(list(prod.find_all(exp.Window)))
    pr_windows = len(list(pr.find_all(exp.Window)))

    if pr_windows > prod_windows:
        return [
            Finding(
                f"{pr_windows - prod_windows} new window function(s) added",
                "low",
                "Window functions increase computation but may not increase bytes scanned.",
            )
        ]
    return []


def _check_column_expansion(prod: exp.Expression, pr: exp.Expression) -> list[Finding]:
    """Detect significant column list expansion in SELECT."""
    prod_cols = _count_select_columns(prod)
    pr_cols = _count_select_columns(pr)

    if pr_cols > prod_cols and (pr_cols - prod_cols) >= 3:
        return [
            Finding(
                f"SELECT expanded by {pr_cols - prod_cols} columns ({prod_cols} → {pr_cols})",
                "low",
                "More columns = more bytes read from columnar storage.",
            )
        ]
    return []


# --- Helper functions ---


def _extract_where_predicates(ast: exp.Expression) -> set[str]:
    """Extract individual WHERE predicate strings from the AST."""
    predicates = set()
    for where in ast.find_all(exp.Where):
        for predicate in _split_and_predicates(where.this):
            predicates.add(predicate.sql(dialect="bigquery"))
    return predicates


def _split_and_predicates(expr: exp.Expression) -> list[exp.Expression]:
    """Split AND-connected predicates into individual conditions."""
    if isinstance(expr, exp.And):
        return _split_and_predicates(expr.left) + _split_and_predicates(expr.right)
    return [expr]


def _count_where_predicates(ast: exp.Expression) -> int:
    """Count the number of WHERE predicates."""
    return len(_extract_where_predicates(ast))


def _star_sources(ast: exp.Expression) -> set[str]:
    """What the query's SELECTs star: "" for a bare `*`, the lowercased qualifier for a
    `t.*` (a Column whose name is a Star)."""
    sources: set[str] = set()
    for select in ast.find_all(exp.Select):
        for expr in select.expressions:
            if isinstance(expr, exp.Star):
                sources.add("")
            elif isinstance(expr, exp.Column) and expr.is_star:
                sources.add((expr.table or "").lower())
    return sources


def _extract_joins(ast: exp.Expression) -> set[tuple[str, str]]:
    """Extract (join_type, table_name) tuples."""
    joins = set()
    for join in ast.find_all(exp.Join):
        join_type = join.side or ""
        if join.kind:
            join_type = f"{join_type} {join.kind}".strip()
        join_type = join_type.upper() or "JOIN"

        table = ""
        if join.this:
            if isinstance(join.this, exp.Table):
                table = join.this.name
            else:
                table = join.this.sql(dialect="bigquery")[:50]
        joins.add((join_type, table))
    return joins


def _extract_table_names(ast: exp.Expression) -> set[str]:
    """Extract all table names from FROM and JOIN clauses, ignoring CTE aliases."""
    cte_names = {cte.alias_or_name.lower() for cte in ast.find_all(exp.CTE) if cte.alias_or_name}
    tables = set()
    for table in ast.find_all(exp.Table):
        if table.name.lower() in cte_names:
            continue
        parts = [p for p in [table.catalog, table.db, table.name] if p]
        if parts:
            tables.add(".".join(parts))
    return tables


def _count_select_columns(ast: exp.Expression) -> int:
    """Count columns in the outermost SELECT."""
    if isinstance(ast, exp.Select):
        return len(ast.expressions)
    selects = list(ast.find_all(exp.Select))
    if selects:
        return len(selects[-1].expressions)
    return 0


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
