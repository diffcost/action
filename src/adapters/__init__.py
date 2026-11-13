"""Input adapters for different SQL tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChangedSQL:
    """Represents a SQL file or dbt model that changed between base and PR branches."""

    name: str  # Display name (model name for dbt, filename for SQL files)
    path: str  # File path relative to repo root
    production_sql: str | None  # SQL from base branch (None if new file)
    pr_sql: str | None  # SQL from PR branch (None if deleted file)
    adapter: str = "sql_files"  # Which adapter produced this
    metadata: dict | None = None  # Adapter-specific metadata (e.g., dbt unique_id)
