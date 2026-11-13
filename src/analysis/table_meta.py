"""Metadata of the tables a query reads (BigQuery `tables.get`): partitioning, clustering,
table type and size.

A metadata read is an API call, not a job: it is free, reads no rows, and needs only
`bigquery.tables.get`, which `roles/bigquery.dataViewer` (already needed for the dry runs)
includes. Table and column names stay on the runner: they appear in the PR comment and the
job summary, never in the payload sent to the DiffCost service.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

# Per-call timeout, and the total time one run spends reading metadata (seconds)
META_CALL_TIMEOUT_SECONDS = 5
META_RUN_BUDGET_SECONDS = 20
_WORKERS = 8

# Table types whose partitioning and clustering describe the data a query scans (a view has
# none of its own, and an external or snapshot table's isn't what a dry run prunes).
STORED_TYPES = frozenset({"TABLE", "MATERIALIZED_VIEW"})

_KIND_LABELS = {"HOUR": "hourly", "DAY": "daily", "MONTH": "monthly", "YEAR": "yearly"}


def format_bytes(num_bytes: int | float | None) -> str:
    """Byte counts in binary units (BigQuery bills per TiB): "4.20 TiB", "—" for None."""
    if num_bytes is None:
        return "—"
    b = float(num_bytes)
    sign = "-" if b < 0 else ""
    b = abs(b)
    if b < 1024:
        return f"{sign}{b:.0f} B"
    elif b < 1024 ** 2:
        return f"{sign}{b / 1024:.1f} KiB"
    elif b < 1024 ** 3:
        return f"{sign}{b / (1024 ** 2):.1f} MiB"
    elif b < 1024 ** 4:
        return f"{sign}{b / (1024 ** 3):.2f} GiB"
    else:
        return f"{sign}{b / (1024 ** 4):.2f} TiB"


@dataclass(frozen=True)
class TableMeta:
    table_id: str  # project.dataset.table
    table_type: str | None = None  # TABLE, VIEW, EXTERNAL, MATERIALIZED_VIEW, SNAPSHOT, ...
    partition_column: str | None = None  # the column; "_PARTITIONTIME" for ingestion-time partitioning
    partition_kind: str | None = None  # HOUR, DAY, MONTH, YEAR or RANGE
    clustering_fields: tuple[str, ...] = ()
    num_bytes: int | None = None

    @property
    def short_name(self) -> str:
        """dataset.table, as the comment names it."""
        return ".".join(self.table_id.split(".")[-2:])

    @property
    def ingestion_time(self) -> bool:
        return self.partition_column == "_PARTITIONTIME"

    @property
    def partition_columns(self) -> frozenset[str]:
        """Lowercased column names a filter prunes partitions on."""
        if not self.partition_column:
            return frozenset()
        if self.ingestion_time:
            return frozenset({"_partitiontime", "_partitiondate"})
        return frozenset({self.partition_column.lower()})

    @property
    def label(self) -> str:
        """The table as findings name it: "analytics.events (4.20 TiB table, partitioned
        daily)", "(… table, partitioned by ingestion time, daily)", "(… table, partitioned by
        integer range)"; without a size "(partitioned daily)"."""
        parts = [f"{format_bytes(self.num_bytes)} table"] if self.num_bytes is not None else []
        if self.partition_kind == "RANGE":
            parts.append("partitioned by integer range")
        elif self.partition_kind:
            kind = _KIND_LABELS.get(self.partition_kind, self.partition_kind.lower())
            parts.append(f"partitioned by ingestion time, {kind}" if self.ingestion_time else f"partitioned {kind}")
        return f"{self.short_name} ({', '.join(parts)})" if parts else self.short_name

    @property
    def stored(self) -> bool:
        return self.table_type in STORED_TYPES

    @property
    def external(self) -> bool:
        return self.table_type == "EXTERNAL"


def parse_table(table_id: str, table) -> TableMeta:
    """A TableMeta from a google-cloud-bigquery Table, keeping only well-typed values."""

    def text(value) -> str | None:
        return value if isinstance(value, str) and value else None

    column = kind = None
    time_partitioning = getattr(table, "time_partitioning", None)
    range_partitioning = getattr(table, "range_partitioning", None)
    if time_partitioning is not None and text(getattr(time_partitioning, "type_", None)):
        kind = time_partitioning.type_
        # No field: ingestion-time partitioning (_PARTITIONTIME / _PARTITIONDATE)
        column = text(getattr(time_partitioning, "field", None)) or "_PARTITIONTIME"
    elif range_partitioning is not None and text(getattr(range_partitioning, "field", None)):
        kind, column = "RANGE", range_partitioning.field

    clustering = getattr(table, "clustering_fields", None)
    num_bytes = getattr(table, "num_bytes", None)
    return TableMeta(
        table_id=table_id,
        table_type=text(getattr(table, "table_type", None)),
        partition_column=column,
        partition_kind=kind,
        clustering_fields=tuple(c for c in clustering if isinstance(c, str)) if isinstance(clustering, list) else (),
        num_bytes=num_bytes if isinstance(num_bytes, int) and not isinstance(num_bytes, bool) else None,
    )


def _get_one(client, table_id: str) -> TableMeta | None:
    try:
        # No retry: one call stops at its timeout, so a cut-off read can't keep the
        # process alive long after the run (Python joins pool threads at exit)
        table = client.get_table(table_id, timeout=META_CALL_TIMEOUT_SECONDS, retry=None)
        return parse_table(table_id, table)
    except Exception:  # no metadata for this table; never a failed run
        return None


def fetch_table_meta(client, table_ids: list[str], timeout: float = META_RUN_BUDGET_SECONDS) -> dict[str, TableMeta]:
    """Metadata of each table, read in parallel. A table whose read fails or doesn't finish
    within `timeout` seconds is left out."""
    ids = list(dict.fromkeys(table_ids))
    if not ids or timeout <= 0:
        return {}
    pool = ThreadPoolExecutor(max_workers=min(_WORKERS, len(ids)))
    futures = {pool.submit(_get_one, client, t): t for t in ids}
    done, _ = wait(futures, timeout=timeout)
    pool.shutdown(wait=False, cancel_futures=True)
    found = {}
    for future in done:
        meta = future.result()
        if meta is not None:
            found[futures[future]] = meta
    return found


class TableMetaCache:
    """One metadata read per distinct table per run, within META_RUN_BUDGET_SECONDS in total."""

    def __init__(self, client, budget_seconds: float = META_RUN_BUDGET_SECONDS):
        self._client = client
        self._remaining = budget_seconds
        self._meta: dict[str, TableMeta | None] = {}  # None: the read failed or timed out

    def get(self, table_ids: list[str]) -> dict[str, TableMeta]:
        """Metadata for the tables that have it (reads the ones not tried yet)."""
        ids = list(dict.fromkeys(table_ids))
        todo = [t for t in ids if t not in self._meta]
        if todo and self._remaining > 0:
            started = time.monotonic()
            found = fetch_table_meta(self._client, todo, timeout=self._remaining)
            self._remaining -= time.monotonic() - started
            for t in todo:
                self._meta[t] = found.get(t)
        return {t: self._meta[t] for t in ids if self._meta.get(t) is not None}

    @property
    def counts(self) -> tuple[int, int]:
        """(tables with metadata, tables tried)"""
        return sum(1 for m in self._meta.values() if m is not None), len(self._meta)
