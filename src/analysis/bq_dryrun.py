"""BigQuery dry-run wrapper for cost estimation."""

from __future__ import annotations

import json
import os
import re

from google.cloud import bigquery

# BigQuery bills on-demand queries per TiB (2**40 bytes): $6.25/TiB.
BYTES_PER_TIB = 2**40
# Per-request timeout and total retry deadline for one dry-run job (seconds).
DRY_RUN_TIMEOUT_SECONDS = 30
DRY_RUN_RETRY_DEADLINE_SECONDS = 60

# "Not found: Table proj:ds.tbl ..." / "Not found: Dataset proj:ds" — a relation that
# does not exist yet (e.g. a dbt model added in the same PR).
_UNMATERIALIZED_RE = re.compile(r"not found:\s*(table|dataset|view)\b", re.IGNORECASE)
# The missing object's name in that error ("Not found: Table my-project:ds.tbl was not found ...")
_MISSING_OBJECT_RE = re.compile(r"not found:\s*(table|dataset|view)\s+([^\s,;]+)", re.IGNORECASE)
# "Unrecognized name: discount_pct" — on the PR side this usually means the query reads a
# column that an upstream model in the same PR adds (prod does not have it yet).
_UNRECOGNIZED_NAME_RE = re.compile(r"unrecognized name:", re.IGNORECASE)
# "Query parameter 'run_date' not found at [1:40]": the query uses a named parameter that the
# job doesn't set, while it sets others (with none set, BigQuery binds every one as NULL)
_MISSING_PARAMETER_RE = re.compile(r"query parameter '([A-Za-z_][A-Za-z0-9_]*)' not found", re.IGNORECASE)


def _container_credentials_path(path: str) -> str:
    """Map a runner-host credentials path to the Docker action's workspace mount.

    google-github-actions/auth writes its credentials file into the workspace and
    exports the runner-host path (e.g. /home/runner/work/repo/repo/gha-creds-x.json).
    Inside a Docker action the workspace is mounted at $GITHUB_WORKSPACE
    (/github/workspace), so the host path does not exist there.
    """
    if not path or os.path.exists(path):
        return path
    workspace = os.environ.get("GITHUB_WORKSPACE", "")
    if workspace:
        candidate = os.path.join(workspace, os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
    return path


def create_bq_client(config) -> bigquery.Client | None:
    """Create a BigQuery client from GCP credentials or Application Default Credentials (ADC)."""
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not config.gcp_credentials and not adc_path:
        return None
    if adc_path and not config.gcp_credentials:
        mapped = _container_credentials_path(adc_path)
        if mapped != adc_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = mapped
        elif not os.path.exists(adc_path):
            print(
                "::warning::GOOGLE_APPLICATION_CREDENTIALS points to a file that does not exist in the "
                "DiffCost container. Run google-github-actions/auth with its default `create_credentials_file: true`."
            )

    if config.gcp_credentials:
        creds_data = config.gcp_credentials.strip()
        if not creds_data.startswith("{"):
            path = _container_credentials_path(creds_data)
            if not os.path.isfile(path):
                # Never echo the value: it is usually a secret stored in the wrong format
                # (e.g. base64), and PR comments are not masked like logs are.
                print(f"::warning::{_CREDENTIALS_FORMAT_ERROR}")
                config.bq_client_error = _CREDENTIALS_FORMAT_ERROR
                return None
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

    try:
        if config.gcp_credentials and config.gcp_credentials.strip().startswith("{"):
            # JSON string — use service_account.Credentials directly (no temp file)
            from google.oauth2 import service_account
            try:
                parsed = json.loads(config.gcp_credentials)
            except ValueError:
                print(f"::warning::{_CREDENTIALS_FORMAT_ERROR}")
                config.bq_client_error = _CREDENTIALS_FORMAT_ERROR
                return None
            project = config.gcp_project or parsed.get("project_id")
            credentials = service_account.Credentials.from_service_account_info(parsed)
            return bigquery.Client(project=project, credentials=credentials)

        # Fallback to standard Application Default Credentials (ADC/WIF)
        project = config.gcp_project or None
        return bigquery.Client(project=project)
    except Exception as e:
        detail = f"{type(e).__name__}: {_sanitize_error(str(e))[:120]}"
        print(f"::warning::Failed to create BigQuery client: {detail}")
        # Shown in the PR comment instead of "no credentials configured"
        config.bq_client_error = detail
        return None


_CREDENTIALS_FORMAT_ERROR = (
    "the gcp_credentials input must be the raw JSON key of a service account (starting with a { character). "
    "If you stored it base64-encoded, store the decoded JSON in the secret instead"
)


def _job_statistics(job) -> tuple[dict, dict]:
    """The dry-run job's raw `statistics` and `statistics.query` (the Python client has no
    properties for the accuracy and row-level security fields). Empty dicts when absent."""
    props = getattr(job, "_properties", None)
    stats = props.get("statistics") if isinstance(props, dict) else None
    stats = stats if isinstance(stats, dict) else {}
    query = stats.get("query")
    return stats, query if isinstance(query, dict) else {}


def _undeclared_parameters(query_stats: dict) -> list[tuple[str, str]]:
    """(name, type) of the parameters a dry run found no value for
    (`undeclaredQueryParameters`, with the type BigQuery inferred), sorted by name. An entry
    without a usable name (a positional `?` parameter, or one this code can't read) is "?",
    and an unreadable type is "": either way the version is never priced."""
    found = query_stats.get("undeclaredQueryParameters")
    if not found:
        return []
    params: dict[str, str] = {}
    for entry in found if isinstance(found, list) else [found]:
        entry = entry if isinstance(entry, dict) else {}
        name = entry.get("name")
        ptype = entry.get("parameterType")
        ptype = ptype.get("type") if isinstance(ptype, dict) else None
        params.setdefault(name if isinstance(name, str) and name else "?", ptype if isinstance(ptype, str) else "")
    return sorted(params.items(), key=lambda p: p[0].lower())


def dry_run(client: bigquery.Client, sql: str, default_dataset: str = "", query_parameters=None) -> dict:
    """
    Submit a dry-run job to BigQuery.
    Returns dict with bytes_processed, cost estimate, or error.
    Dry-runs are free — they return a byte estimate and no rows.

    `accuracy` is BigQuery's own word on the estimate (`totalBytesProcessedAccuracy`:
    PRECISE, UPPER_BOUND for clustered tables, LOWER_BOUND for external tables and for
    scripts it stops estimating early). `row_level_security`: the query reads a table with
    row access policies, for which BigQuery returns no byte estimate (bytes_processed None).

    `default_dataset` ("project.dataset") and `query_parameters` (config.QueryParameter) are
    set on the job when given. `unbound_parameters` lists the (name, type) of named
    parameters the query uses without a value: with no parameters set, BigQuery binds them
    as NULL and still returns an estimate (often 0 bytes, because `= NULL` matches nothing),
    so that estimate can't be trusted; with some set, a missing one fails the dry run, and
    is reported here instead of as an error (type "", which BigQuery doesn't name then).
    """
    if not sql or not sql.strip():
        return {"error": "Empty SQL"}

    try:
        # No query cache: a cached result would make the dry run report 0 bytes billed
        job_config = bigquery.QueryJobConfig(dry_run=True, use_legacy_sql=False, use_query_cache=False)
        if default_dataset:
            job_config.default_dataset = default_dataset
        if query_parameters:
            job_config.query_parameters = [
                bigquery.ScalarQueryParameter(p.name, p.type, p.value) for p in query_parameters
            ]
        job = client.query(
            sql,
            job_config=job_config,
            timeout=DRY_RUN_TIMEOUT_SECONDS,
            retry=bigquery.DEFAULT_RETRY.with_deadline(DRY_RUN_RETRY_DEADLINE_SECONDS),
        )
        stats, query_stats = _job_statistics(job)
        rls = stats.get("rowLevelSecurityStatistics")
        row_level_security = isinstance(rls, dict) and rls.get("rowLevelSecurityApplied") is True
        accuracy = query_stats.get("totalBytesProcessedAccuracy")
        statement_type = getattr(job, "statement_type", None)

        referenced_tables = []
        if hasattr(job, "referenced_tables") and job.referenced_tables:
            referenced_tables = [
                f"{t.project}.{t.dataset_id}.{t.table_id}" for t in job.referenced_tables
            ]

        # No estimate stays None (unknown cost), never $0. BigQuery returns one for every
        # statement type, DDL included (0 bytes), except for tables with row-level security.
        total = job.total_bytes_processed
        if row_level_security or total is None:
            bytes_processed = None
        else:
            bytes_processed = total

        return {
            "bytes_processed": bytes_processed,
            "referenced_tables": referenced_tables,
            "unmaterialized_upstream": False,
            "unrecognized_name": False,
            "error": None,
            "accuracy": accuracy if isinstance(accuracy, str) else None,
            "statement_type": statement_type if isinstance(statement_type, str) else None,
            "row_level_security": row_level_security,
            "unbound_parameters": _undeclared_parameters(query_stats),
        }
    except Exception as e:
        missing_parameter = _MISSING_PARAMETER_RE.search(str(e))
        if missing_parameter:
            return {
                "bytes_processed": None,
                "referenced_tables": [],
                "unmaterialized_upstream": False,
                "unrecognized_name": False,
                "error": None,
                "unbound_parameters": [(missing_parameter.group(1), "")],
            }
        if _is_unmaterialized(e):
            return {
                "bytes_processed": None,
                "referenced_tables": [],
                "unmaterialized_upstream": True,
                "unrecognized_name": False,
                "error": None,
                # From the raw error, for matching against relations this PR creates. In
                # memory only: never in the payload, the comment or the logs.
                "missing_object": _missing_object(str(e)),
            }

        # Sanitize error: strip table names, project IDs, and other schema details
        # that could leak sensitive info into PR comments.
        sanitized = _sanitize_error(str(e))
        return {
            "bytes_processed": 0,
            "referenced_tables": [],
            "unmaterialized_upstream": False,
            "unrecognized_name": bool(_UNRECOGNIZED_NAME_RE.search(str(e))),
            "error": sanitized[:200],
        }


def dry_run_pair(
    client: bigquery.Client,
    production_sql: str | None,
    pr_sql: str | None,
    cost_per_tb: float = 6.25,
    default_dataset: str = "",
    query_parameters=None,
) -> dict:
    """
    Dry-run both production and PR versions of a query.
    Returns a comparison dict with cost_before and cost_after (comment.classify_cost
    derives the change from them).

    `pr_depends_on_new_objects` is set when the production side ran (or there is no
    production SQL) but the PR side failed because it reads a relation or column that
    does not exist in production yet — typically created by another model in this PR.
    The caller decides whether something upstream in this PR explains it; `pr_error`
    keeps the sanitized PR-side error ("Unrecognized name: …"; None for "Not found"), and
    `pr_missing_object` the object a "Not found" names (in memory only, never shown).

    Per side (`_before` / `_after`): `accuracy` and `statement_type` from BigQuery, the
    referenced tables (`tables_before` / `tables_after`; in memory only, never sent), and
    `rls_before` / `rls_after` when that side reads a table with row-level security (no byte
    estimate). `unbound_parameters_before` / `unbound_parameters_after`: the named parameters
    that version uses without a value. accuracy.assess turns those into `unknown_reason`.

    `default_dataset` and `query_parameters` apply to both versions (see `dry_run`).
    """
    result = {
        "cost_before": None,
        "cost_after": None,
        "bytes_before": None,
        "bytes_after": None,
        "unmaterialized_upstream": False,
        "pr_depends_on_new_objects": False,
        "error": None,
        "tables_before": [],
        "tables_after": [],
        "rls_before": False,
        "rls_after": False,
        "accuracy_before": None,
        "accuracy_after": None,
        "statement_type_before": None,
        "statement_type_after": None,
        "unbound_parameters_before": [],
        "unbound_parameters_after": [],
        "unknown_reason": "",
    }
    options = {"default_dataset": default_dataset, "query_parameters": query_parameters}
    # Dry-run production version
    if production_sql:
        prod_result = dry_run(client, production_sql, **options)
        result["unbound_parameters_before"] = prod_result.get("unbound_parameters") or []
        if prod_result.get("unmaterialized_upstream"):
            result["unmaterialized_upstream"] = True
        elif prod_result.get("error"):
            result["error"] = f"Production dry-run failed: {prod_result['error']}"
        elif prod_result.get("row_level_security"):
            result["rls_before"] = True
            result["tables_before"] = prod_result["referenced_tables"]
        elif prod_result.get("bytes_processed") is not None:
            result["bytes_before"] = prod_result["bytes_processed"]
            result["cost_before"] = _bytes_to_cost(prod_result["bytes_processed"], cost_per_tb)
            result["tables_before"] = prod_result["referenced_tables"]
            result["accuracy_before"] = prod_result.get("accuracy")
            result["statement_type_before"] = prod_result.get("statement_type")

    # Dry-run PR version
    if pr_sql:
        pr_result = dry_run(client, pr_sql, **options)
        result["unbound_parameters_after"] = pr_result.get("unbound_parameters") or []
        # A base that hit row-level security did run: its reason is the one to show
        base_ok = not production_sql or result["cost_before"] is not None or result["rls_before"]
        missing = pr_result.get("unmaterialized_upstream") or pr_result.get("unrecognized_name")
        if missing and base_ok:
            result["pr_depends_on_new_objects"] = True
            # Shown if the caller finds nothing upstream that this PR creates or changes
            result["pr_error"] = pr_result.get("error")
            result["pr_missing_object"] = pr_result.get("missing_object")
        elif pr_result.get("unmaterialized_upstream"):
            result["unmaterialized_upstream"] = True
        elif pr_result.get("error"):
            if result["error"]:
                result["error"] += f"; PR dry-run failed: {pr_result['error']}"
            else:
                result["error"] = f"PR dry-run failed: {pr_result['error']}"
        elif pr_result.get("row_level_security"):
            result["rls_after"] = True
            result["tables_after"] = pr_result["referenced_tables"]
        elif pr_result.get("bytes_processed") is not None:
            result["bytes_after"] = pr_result["bytes_processed"]
            result["cost_after"] = _bytes_to_cost(pr_result["bytes_processed"], cost_per_tb)
            result["tables_after"] = pr_result["referenced_tables"]
            result["accuracy_after"] = pr_result.get("accuracy")
            result["statement_type_after"] = pr_result.get("statement_type")

    return result


def _sanitize_error(raw: str) -> str:
    """Keep the BigQuery error reason, drop the request URL and fully qualified names."""
    msg = raw.strip().splitlines()[0] if raw.strip() else "Unknown error"
    # google-api-core prefixes "400 POST https://bigquery.googleapis.com/...: <reason>"
    msg = re.sub(r"^\d{3}\s+[A-Z]+\s+https?://\S+?:\s*", "", msg)
    msg = re.sub(r"https?://\S+", "<url>", msg)
    # Remove anything that looks like project.dataset.table or project:dataset.table references
    msg = re.sub(r"`[^`]+`", "`<redacted>`", msg)
    msg = re.sub(r"[a-zA-Z0-9_-]+[.:][a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", "<redacted>", msg)
    return msg


def _missing_object(raw: str) -> dict | None:
    """`{"kind": "table" | "view" | "dataset", "name": "project:dataset.table"}` from a raw
    "Not found" error, or None when the name can't be read."""
    match = _MISSING_OBJECT_RE.search(raw)
    if not match:
        return None
    name = match.group(2).strip("`'\".")
    return {"kind": match.group(1).lower(), "name": name} if name else None


def _is_unmaterialized(e: Exception) -> bool:
    """Whether a dry-run error means a referenced table/dataset does not exist yet."""
    return bool(_UNMATERIALIZED_RE.search(str(e)))


def _bytes_to_cost(bytes_processed: int, cost_per_tb: float) -> float:
    """Convert bytes processed to USD. `cost_per_tb` is USD per TiB (2**40 bytes)."""
    return (bytes_processed / BYTES_PER_TIB) * cost_per_tb
