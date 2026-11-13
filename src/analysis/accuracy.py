"""How far a dry run's byte estimate can be trusted.

BigQuery says it itself (`totalBytesProcessedAccuracy`): UPPER_BOUND when a query reads
clustered tables (blocks are pruned only when the query runs), LOWER_BOUND for external
tables (often 0 bytes) and for multi-statement scripts it stops estimating early (at DDL,
CALL, EXECUTE IMMEDIATE, IF, WHILE or FOR). A lower bound can hide any cost, so it makes the
cost unknown (fail closed); an upper bound is shown with a "≤" marker. A missing accuracy
field is trusted, except for a version that reads an external table: its estimate is
treated as a lower bound. A table with row-level security gets no byte estimate at all.
"""

from __future__ import annotations

PRECISE = "PRECISE"
LOWER_BOUND = "LOWER_BOUND"
UPPER_BOUND = "UPPER_BOUND"
_SIDES = (("base", "before"), ("PR", "after"))


def versions_label(sides: list[str]) -> str:
    """"the base version", "the PR version" or "both versions" for sides ("base", "PR")."""
    if len(set(sides)) > 1:
        return "both versions"
    return f"the {sides[0]} version" if sides else "the query"


def assess(dr: dict, meta: dict) -> None:
    """Annotate a dry_run_pair result in place, with `meta` (table id -> TableMeta) for the
    tables it references:

    - `unknown_reason`, the one place it is set for an estimate BigQuery can't stand behind:
      either version reads a table with row-level security (no byte estimate), or its
      estimate is only a lower bound, or of unknown accuracy (BigQuery's UNKNOWN, or a value
      this code doesn't know);
    - that version's bytes and cost are dropped (None, like a version with no estimate), so
      a figure that could hide any cost is never shown, summed or sent, also when the item
      stays costed because the PR deletes the query;
    - `upper_bound_before` / `upper_bound_after`: that version's figure is an upper bound.
    """
    rls = [side for side, key in _SIDES if dr.get(f"rls_{key}")]
    lower = [side for side, key in _SIDES if _lower_bound(dr, meta, key)]
    other = [
        side for side, key in _SIDES
        if dr.get(f"accuracy_{key}") not in (None, PRECISE, UPPER_BOUND, LOWER_BOUND)
    ]
    if rls:
        who = versions_label(rls)
        dr["unknown_reason"] = (
            f"{who} {_reads(who)} a table with row-level security, "
            "and BigQuery dry runs give no byte estimate for those tables"
        )
    elif lower:
        dr["unknown_reason"] = _lower_bound_reason(dr, meta, lower)
    elif other:
        dr["unknown_reason"] = f"BigQuery doesn't know how accurate the dry run of {versions_label(other)} is"
    untrusted = [key for side, key in _SIDES if side in lower or side in other]
    for key in untrusted:
        dr[f"bytes_{key}"] = dr[f"cost_{key}"] = None
    for _side, key in _SIDES:
        dr[f"upper_bound_{key}"] = dr.get(f"accuracy_{key}") == UPPER_BOUND


def _lower_bound(dr: dict, meta: dict, key: str) -> bool:
    accuracy = dr.get(f"accuracy_{key}")
    if accuracy == LOWER_BOUND:
        return True
    # No accuracy field: trusted, unless the version reads an external table, whose data
    # BigQuery measures only when the query runs
    return accuracy is None and dr.get(f"bytes_{key}") is not None and bool(_external(dr, meta, [key]))


def _external(dr: dict, meta: dict, keys: list[str]) -> list[str]:
    """The external tables (dataset.table) the versions `keys` read, from their metadata."""
    return sorted({
        meta[t].short_name for k in keys for t in dr.get(f"tables_{k}") or [] if t in meta and meta[t].external
    })


def _reads(who: str) -> str:
    return "read" if who == "both versions" else "reads"


def _lower_bound_reason(dr: dict, meta: dict, sides: list[str]) -> str:
    who = versions_label(sides)
    keys = [key for side, key in _SIDES if side in sides]
    external = _external(dr, meta, keys)
    if external:
        names = ", ".join(external[:3]) + (f" and {len(external) - 3} more" if len(external) > 3 else "")
        noun = "external tables" if len(external) > 1 else "external table"
        return (
            f"{who} {_reads(who)} the {noun} {names}, whose data BigQuery measures only when the query runs, "
            "so the dry run gives only a lower bound"
        )
    if any(dr.get(f"statement_type_{k}") == "SCRIPT" for k in keys):
        return (
            "BigQuery stops estimating a multi-statement script at a DDL statement, CALL, EXECUTE IMMEDIATE, "
            f"IF, WHILE or FOR, so the dry run of {who} gives only a lower bound"
        )
    return f"the dry run of {who} gives only a lower bound"
