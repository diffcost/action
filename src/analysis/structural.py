"""Structural SQL analysis using sqlglot AST comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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

    A file with several statements is compared as a whole: every statement's conditions,
    tables, joins and stars count, whichever statement they are in (see _parse_statements).
    """
    prod_ast = _parse_statements(production_sql)
    if isinstance(prod_ast, str):
        return [Finding("Could not parse production SQL", "low", prod_ast, skipped=True)]

    pr_ast = _parse_statements(pr_sql)
    if isinstance(pr_ast, str):
        return [Finding("Could not parse PR SQL", "low", pr_ast, skipped=True)]

    try:
        findings = _run_checks(prod_ast, pr_ast, partition_columns, cluster_columns)
    except Exception as e:  # never let a sqlglot edge case fail the run
        return [Finding("Structural analysis failed", "low", f"{type(e).__name__} — skipping structural analysis", skipped=True)]

    return findings


class _Statements(exp.Expression):
    """The statements of one SQL text that has several, as one tree: the checks walk a whole
    tree, so they see every statement."""

    arg_types = {"expressions": True}


_PARSE_ERROR = "sqlglot parse error — skipping structural analysis"
_NOTHING_TO_ANALYZE = (
    "nothing to analyze (only statements the SQL parser keeps as plain text, such as DECLARE or CALL) "
    "— skipping structural analysis"
)
# Script statements sqlglot keeps as text (`exp.Command`) end at the first `;`, so the one that
# opens a block also holds the block's first statement: `BEGIN SELECT ...`, `IF c THEN SELECT ...`.
# The text after these openers is parsed again. `None`: the whole text after the keyword.
_BLOCK_OPENERS = {
    "BEGIN": None,
    "ELSE": None,
    "IF": re.compile(r"\bTHEN\b", re.IGNORECASE),
    "ELSEIF": re.compile(r"\bTHEN\b", re.IGNORECASE),
    "EXCEPTION": re.compile(r"^\s*WHEN\s+ERROR\s+THEN\b", re.IGNORECASE),
}


def _parse_statements(sql: str) -> exp.Expression | str:
    """Parse every statement of `sql`: the statement itself when there is one, a `_Statements`
    node over all of them otherwise (they are never paired by position: a statement added in
    front would shift every pair). When nothing can be analyzed, the reason instead (a
    Finding detail): the text doesn't parse (a `WHILE`, `LOOP`, `REPEAT` or `CASE` statement
    block among them), or it holds only statements sqlglot keeps as text. Such a block that is
    the first statement of a block opener (see _BLOCK_OPENERS) is part of the opener's text:
    it fails to parse there and is skipped, with the first statement of its body, while the
    rest of the text is analyzed.

    Left out: empty statements (a trailing `;`, a comment after it) and the ones sqlglot only
    keeps as text (`exp.Command`: DECLARE, CALL, END), which nothing can be read from. A block
    opener's text is parsed again (see _BLOCK_OPENERS), so the statements inside a
    `BEGIN … END`, `IF … THEN` or `EXCEPTION WHEN ERROR THEN` block count, the first one
    included.
    """
    try:
        pending = list(sqlglot.parse(sql, dialect="bigquery"))
    except (SqlglotError, RecursionError):  # sqlglot's parser recurses once per nested subquery
        return _PARSE_ERROR
    statements: list[exp.Expression] = []
    script = False  # the text has script statements, which carry nothing to analyze
    while pending:
        statement = pending.pop(0)
        if statement is None or isinstance(statement, exp.Semicolon):
            continue
        if not isinstance(statement, exp.Command):
            statements.append(statement)
            continue
        script = True
        inner = _block_text(statement)
        if inner:
            try:
                pending[:0] = sqlglot.parse(inner, dialect="bigquery")
            except Exception:  # never let a sqlglot edge case fail the run
                pass  # e.g. an IF condition with its own THEN: that statement stays unread
    if not statements:
        return _NOTHING_TO_ANALYZE if script else _PARSE_ERROR
    return statements[0] if len(statements) == 1 else _Statements(expressions=statements)


def _block_text(command: exp.Command) -> str:
    """The statement a block-opening script statement holds (see _BLOCK_OPENERS), or ""."""
    keyword = str(command.this or "").upper()
    if keyword not in _BLOCK_OPENERS:
        return ""
    text = command.expression  # a Literal or the plain text, depending on the keyword
    text = text.this if isinstance(text, exp.Literal) else text if isinstance(text, str) else ""
    marker = _BLOCK_OPENERS[keyword]
    if marker is not None:
        found = marker.search(text)
        text = text[found.end():] if found else ""
    return text.strip()


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
    """Drop duplicate findings. A removed partition filter already explains a query left
    without WHERE conditions, and a new JOIN already reports the table it joins."""
    has_partition_removal = any(f.description.startswith("Partition filter removed") for f in findings)
    joined = {
        f.subject.lower().split(".")[-1]
        for f in findings
        if f.subject and (f.description.endswith("JOIN") or f.description == "CROSS JOIN added")
    }
    seen: set[tuple[str, str]] = set()
    result = []
    for f in findings:
        if has_partition_removal and f.description == "All WHERE conditions removed":
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
    """Detect SELECT * additions: a table whose columns all reach the query's result through
    `*` or `t.*` in the PR version and didn't in the production version (see _star_reach).

    So a star added inside a CTE or derived table counts when the result reads it through
    another star (`WITH b AS (SELECT * FROM t) SELECT * FROM b`), whether or not the query had
    a star before, and doesn't when the result takes named columns from it: BigQuery reads
    the columns the result uses.

    A star that ends at a CTE's or derived table's own column list passes on that list. When
    the list reads more source columns than the production version's result does
    (_list_columns, less the star's `EXCEPT`), each of the list's tables that the production
    version's stars didn't reach counts on its own: `SELECT id, name FROM s` becoming
    `SELECT * FROM s` reads every column `s` lists, while `SELECT id FROM t` becoming
    `WITH s AS (SELECT id FROM t) SELECT * FROM s` reads nothing more.

    A table the production version didn't read at all counts only when the PR also writes a
    star the production version doesn't have: under an unchanged `*`, a swapped or newly
    joined table is reported as a new table or JOIN, not as a star."""
    prod_reach = _star_reach(prod)
    pr_reach = _star_reach(pr)
    added = pr_reach.tables - prod_reach.tables
    width = _result_width(prod)
    # What the production stars reached; each table of a list counts on its own
    known = prod_reach.tables.union(*prod_reach.reached)
    for tables, columns in pr_reach.lists:
        if columns > width:
            added |= tables - known
    if added and not _star_sources(pr) - _star_sources(prod):
        added &= _extract_table_names(prod)

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
    """Detect a query that lost every WHERE condition.

    A query that keeps some conditions gets no count-based finding: each removed condition
    already has its own (see _check_partition_filter_removal), and a count of distinct
    condition texts also drops when nothing was removed (two bounds rewritten as one BETWEEN)."""
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
    """Split AND-connected predicates into individual conditions. A parenthesized group of
    ANDs, `(a AND b)`, is split like `a AND b`; any other parenthesized condition, such as
    `(a OR b)`, stays one condition as written."""
    conditions: list[exp.Expression] = []
    pending = [expr]  # an explicit stack: a long chain of ANDs is a deep tree
    while pending:
        node = pending.pop()
        inner = _unwrap(node)
        if isinstance(inner, exp.And):
            pending.extend([inner.right, inner.left])  # the left one first
        else:
            conditions.append(node)
    return conditions


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
    """Extract all table names from FROM and JOIN clauses, ignoring references to CTEs. A
    CTE is referenced by its bare name: `raw.orders` is a table even when a CTE is named
    `orders` (an import CTE is often named after the table it reads)."""
    cte_names = {cte.alias_or_name.lower() for cte in ast.find_all(exp.CTE) if cte.alias_or_name}
    tables = set()
    for table in ast.find_all(exp.Table):
        if not table.db and not table.catalog and table.name.lower() in cte_names:
            continue
        name = _table_name(table)
        if name:
            tables.add(name)
    return tables


def _table_name(table: exp.Table) -> str:
    """`project.dataset.table`, with the parts the query wrote. A table function is named by
    its function (`p.d.tvf`, `ML.PREDICT`): sqlglot gives it no table name."""
    name = table.name
    if isinstance(table.this, exp.Func):
        name = table.this.name if isinstance(table.this, exp.Anonymous) else table.this.sql_name()
    return ".".join(p for p in [table.catalog, table.db, name] if p)


def _result_selects(node: exp.Expression | None) -> list[exp.Select]:
    """The SELECTs whose lists make up what `node` returns or stores: the query itself, every
    branch of a set operation, the query of a CREATE ... AS, an INSERT or a MERGE's USING, and
    those of each statement of a multi-statement text. Empty for anything else. Walked with an
    explicit stack: sqlglot nests a long UNION ALL one level per branch."""
    selects: list[exp.Select] = []
    pending = [node]
    while pending:
        node = pending.pop()
        if isinstance(node, exp.Select):
            selects.append(node)
        elif isinstance(node, _Statements):
            pending.extend(node.expressions)
        elif isinstance(node, exp.SetOperation):
            pending.extend([node.this, node.expression])
        elif isinstance(node, exp.Subquery):
            pending.append(node.this)
        elif isinstance(node, (exp.Create, exp.Insert)):
            pending.append(node.expression)
        elif isinstance(node, exp.Merge):
            pending.append(node.args.get("using"))
    return selects


def _select_sources(select: exp.Select) -> list[exp.Expression]:
    """The tables (CTE references included) and derived tables a SELECT reads directly: its
    FROM and JOINs, and the members of a parenthesized join. UNNEST and other sources that
    aren't a table or a subquery are left out."""
    sources: list[exp.Expression] = []
    pending = [select.args.get("from"), *(select.args.get("joins") or [])]
    while pending:
        item = pending.pop(0)
        node = item.this if isinstance(item, (exp.From, exp.Join)) else item
        if isinstance(node, exp.Subquery) and isinstance(node.this, exp.Table):
            node = node.this  # (a JOIN b): the table, with the join hanging off it
        if isinstance(node, exp.Table):
            sources.append(node)
            pending.extend(node.args.get("joins") or [])
        elif isinstance(node, exp.Subquery):
            sources.append(node)
    return sources


def _referenced_cte(table: exp.Table) -> exp.CTE | None:
    """The CTE a bare table name refers to, looked up from where the name is written: the
    nearest enclosing WITH that defines it. A CTE's own name inside its definition is the
    table of that name unless the WITH is recursive."""
    if table.db or table.catalog or not table.name:
        return None
    name = table.name.lower()
    own = None  # the CTE whose definition holds the name, at the WITH being looked at
    node = table.parent
    while node is not None:
        if isinstance(node, exp.CTE):
            own = node
        with_ = node.args.get("with")
        if isinstance(with_, exp.With):
            for cte in with_.expressions:
                if (cte.alias_or_name or "").lower() == name and (cte is not own or with_.args.get("recursive")):
                    return cte
        node = node.parent
    return None


def _other_source_names(select: exp.Select) -> set[str]:
    """The names a SELECT gives its sources that aren't tables or subqueries (`UNNEST(x) AS e`,
    `t.arr AS e`): what `e.*` or a bare `e` refers to there reads no table's row."""
    names = set()
    pending = [select.args.get("from"), *(select.args.get("joins") or [])]
    while pending:  # the same walk as _select_sources, parenthesized joins included
        item = pending.pop(0)
        node = item.this if isinstance(item, (exp.From, exp.Join)) else item
        if isinstance(node, exp.Subquery) and isinstance(node.this, exp.Table):
            node = node.this
        if isinstance(node, exp.Table):
            pending.extend(node.args.get("joins") or [])
        elif isinstance(node, exp.Unnest):
            alias = node.args.get("alias")
            names.update(c.name.lower() for c in (alias.columns if alias is not None else []))
            if node.alias:
                names.add(node.alias.lower())
    return names


@dataclass
class _StarReach:
    """What reaches a query's result through stars (see _star_reach)."""

    tables: set[str] = field(default_factory=set)  # read in full
    # The tables under each SELECT a star reached, below the result's own SELECTs
    reached: list[frozenset[str]] = field(default_factory=list)
    # A star that ended at a column list: the tables under that SELECT, and its width
    lists: list[tuple[frozenset[str], int]] = field(default_factory=list)


def _source_selects(source: exp.Expression) -> list[exp.Select] | None:
    """The SELECTs a source's rows come from: a derived table's, or the referenced CTE's.
    None for a table."""
    if isinstance(source, exp.Subquery):
        return _result_selects(source)
    cte = _referenced_cte(source)
    return _result_selects(cte.this) if cte is not None else None


def _tables_under(select: exp.Select) -> frozenset[str]:
    """The tables a SELECT reads from, through its CTEs and derived tables (FROM and JOINs)."""
    tables: set[str] = set()
    seen: set[int] = set()
    pending = [select]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for source in _select_sources(node):
            inner = _source_selects(source)
            if inner is not None:
                pending.extend(inner)
            elif _table_name(source):
                tables.add(_table_name(source))
    return frozenset(tables)


def _projection_queries(expr: exp.Expression) -> list[exp.Select]:
    """The SELECTs a SELECT-list entry holds (`ARRAY(SELECT AS STRUCT * ...)`, a scalar
    subquery): their rows are part of the result. Not those under EXISTS or IN, which only test."""
    found: list[exp.Select] = []
    pending = [expr]
    while pending:
        node = pending.pop()
        if node is not expr and isinstance(node, (exp.Select, exp.Subquery, exp.SetOperation)):
            found.extend(_result_selects(node))
            continue
        if isinstance(node, (exp.Exists, exp.In)):
            continue
        pending.extend(node.iter_expressions())
    return found


def _sources_by_name(select: exp.Select) -> dict[str, list[exp.Expression]]:
    """A SELECT's sources (_select_sources) by the lowercased name the query uses for them: the
    alias or table name, and the alias of a PIVOT / UNPIVOT applied to them."""
    by_name: dict[str, list[exp.Expression]] = {}
    for source in _select_sources(select):
        by_name.setdefault((source.alias_or_name or "").lower(), []).append(source)
        for pivot in source.args.get("pivots") or []:
            if pivot.alias:
                by_name.setdefault(pivot.alias.lower(), []).append(source)
    return by_name


def _star_except(star: exp.Expression) -> frozenset[str]:
    """The columns a `* EXCEPT (…)` or `t.* EXCEPT (…)` leaves out (lowercased)."""
    node = star.this if isinstance(star, exp.Column) else star
    return frozenset(c.name.lower() for c in (node.args.get("except") or []) if isinstance(c, exp.Column))


def _column_key(column: exp.Column) -> tuple[str, str]:
    """A column as (qualifier, name), lowercased: `a.id` and `b.id` are two columns."""
    return (column.table or "").lower(), column.name.lower()


def _join_columns(select: exp.Select) -> set[tuple[str, str]]:
    """The columns a SELECT's join conditions read (`ON`, and `USING` columns unqualified),
    parenthesized joins included."""
    columns: set[tuple[str, str]] = set()
    joins = list(select.args.get("joins") or [])
    for source in _select_sources(select):
        joins.extend(source.args.get("joins") or [])
    for join in joins:
        on = join.args.get("on")
        if on is not None:
            columns.update(_column_key(c) for c in _outer_columns(on) if not c.is_star)
        for column in join.args.get("using") or []:
            columns.add(("", column.name.lower()))
    return columns


def _list_columns(select: exp.Select, left_out: frozenset[str] = frozenset()) -> set[tuple[str, str]]:
    """How wide a SELECT is, as the source columns (qualifier, name) its entries and its join
    conditions read: a constant or `CURRENT_TIMESTAMP()` reads none, `amount * 100 AS cents`
    reads `amount`, `a.id` and `b.id` are two. A struct column's `s.*` counts as the column
    `s`, an UNNEST alias's `e.*` as nothing, a star over a source as nothing (what it reaches
    is followed instead). Entries whose output name is in `left_out` (the `EXCEPT` of the
    stars that read this list) don't count."""
    sources = _sources_by_name(select)
    unread = _other_source_names(select)
    columns: set[tuple[str, str]] = set()
    for expr in select.expressions:
        if isinstance(expr, exp.Star):
            continue
        if isinstance(expr, exp.Column) and expr.is_star:
            qualifier = (expr.table or "").lower()
            if qualifier and qualifier not in sources and qualifier not in unread:
                columns.add((qualifier, ""))
            continue
        if (expr.alias_or_name or "").lower() in left_out:
            continue
        columns.update(_column_key(c) for c in _outer_columns(expr) if not c.is_star)
    return columns | _join_columns(select)


def _result_width(ast: exp.Expression) -> int:
    """How wide the query's own result is (_list_columns of its outermost SELECT: for a set
    operation its first SELECT, for a CREATE ... AS or an INSERT the query it stores, for a
    MERGE its USING), summed over the statements of a multi-statement text."""
    if isinstance(ast, _Statements):
        return sum(_result_width(statement) for statement in ast.expressions)
    node: exp.Expression | None = ast
    while node is not None and not isinstance(node, exp.Select):
        if isinstance(node, (exp.SetOperation, exp.Subquery)):
            node = node.this
        elif isinstance(node, (exp.Create, exp.Insert)):
            node = node.expression
        elif isinstance(node, exp.Merge):
            node = node.args.get("using")
        else:
            return 0
    return len(_list_columns(node)) if node is not None else 0


def _struct_rows(qualifier: str, sources: list[exp.Expression]) -> list[exp.Expression]:
    """The sources whose whole row a struct column `qualifier` holds, for `qualifier.*` where
    `qualifier` names no source: a derived table's or CTE's SELECT-list entry of that name,
    built from a source's row (`ARRAY_AGG(original ...)[OFFSET(0)] AS unique`). A table's own
    struct column, or one passed on by a `*`, reads only that column."""
    rows: list[exp.Expression] = []
    for source in sources:
        for select in _source_selects(source) or []:
            by_name = _sources_by_name(select)
            for expr in select.expressions:
                if (expr.alias_or_name or "").lower() != qualifier:
                    continue
                for column in _outer_columns(expr):
                    name = column.name.lower()
                    if not column.table and not column.is_star and name in by_name:
                        rows.extend(by_name[name])
    return rows


def _star_reach(ast: exp.Expression) -> _StarReach:
    """The tables whose every column reaches the query's result: walked from the result's
    SELECT lists (_result_selects, and the SELECTs inside a SELECT-list entry) to the sources
    each one reads in full, and on through CTEs and derived tables, to the tables. A SELECT
    reads a source in full through a bare `*` (every source), a `t.*` (the source named `t`),
    or the source's name used as a value: a bare `t` or an argument such as
    `TO_JSON_STRING(t)` or `ARRAY_AGG(t ...)` is the whole row, as in BigQuery, where a
    source's name wins over a column of the same name. A `s.*` whose `s` names no source is a
    struct column's fields: only a struct built from a source's row reads that row
    (_struct_rows). A SELECT list of named columns ends the walk (recorded in `lists`, as wide
    as the source columns it reads, less the `EXCEPT` of the star that reached it): BigQuery
    reads only the columns the result uses. A `*` anywhere else (an EXISTS or IN subquery)
    isn't followed, nor one in a SELECT-list subquery of a SELECT the walk doesn't reach. A
    table function counts as a table, named by its function. A PIVOT's alias names the
    source it pivots, and a table a PIVOT or UNPIVOT reads directly is read in full wherever
    it is (BigQuery groups by every input column it doesn't aggregate or pivot)."""
    reach = _StarReach()
    for table in ast.find_all(exp.Table):
        if table.args.get("pivots") and _referenced_cte(table) is None and _table_name(table):
            reach.tables.add(_table_name(table))
    # SELECTs reached through a star: the columns every star that reached it leaves out
    below: dict[int, frozenset[str]] = {}
    lists: list[tuple[exp.Select, frozenset[str]]] = []  # measured once every star is known
    seen: set[int] = set()  # SELECTs already walked (a recursive CTE reads itself)
    pending = _result_selects(ast)
    while pending:
        select = pending.pop()
        if id(select) in seen:
            continue
        seen.add(id(select))
        sources = _select_sources(select)
        by_name = _sources_by_name(select)
        unread = _other_source_names(select)
        starred: list[tuple[exp.Expression, frozenset[str]]] = []  # each with its star's EXCEPT
        for expr in select.expressions:
            if isinstance(expr, exp.Star):
                starred.extend((source, _star_except(expr)) for source in sources)
                continue
            if isinstance(expr, exp.Column) and expr.is_star:
                qualifier = (expr.table or "").lower()
                if qualifier in by_name:
                    starred.extend((source, _star_except(expr)) for source in by_name[qualifier])
                elif qualifier not in unread:
                    starred.extend((source, frozenset()) for source in _struct_rows(qualifier, sources))
                continue
            for column in _outer_columns(expr):
                name = column.name.lower()
                if not column.table and not column.is_star and name in by_name:
                    starred.extend((source, frozenset()) for source in by_name[name])
            pending.extend(_projection_queries(expr))
        if id(select) in below:
            tables = _tables_under(select)
            if tables:
                reach.reached.append(tables)
                if not starred:
                    lists.append((select, tables))
        for source, left_out in starred:
            inner = _source_selects(source)
            if inner is not None:
                for s in inner:
                    below[id(s)] = below[id(s)] & left_out if id(s) in below else left_out
                pending.extend(inner)
            elif _table_name(source):
                reach.tables.add(_table_name(source))
    reach.lists = [(tables, len(_list_columns(select, below[id(select)]))) for select, tables in lists]
    return reach


def _count_select_columns(ast: exp.Expression) -> int:
    """Count the columns of the query's own SELECT list: for a set operation its first SELECT
    (whose columns name the result), for a CREATE ... AS or an INSERT the query it stores, and
    never a subquery's. Summed over the statements of a multi-statement text."""
    if isinstance(ast, _Statements):
        return sum(_count_select_columns(statement) for statement in ast.expressions)
    node: exp.Expression | None = ast
    while node is not None and not isinstance(node, exp.Select):
        if isinstance(node, (exp.SetOperation, exp.Subquery)):
            node = node.this
        elif isinstance(node, (exp.Create, exp.Insert)):
            node = node.expression
        else:
            return 0
    return len(node.expressions) if node is not None else 0


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
