"""DiffCost GitHub Action entrypoint.

Flow: load config → (with api_key) fetch dashboard config → find changed SQL →
structural analysis + BigQuery dry runs → per-run totals and worst single query →
local threshold check → send numbers to the backend (which posts the
`diffcost/cost-check` commit status) → Slack for the .diffcost.yml webhook → PR
comment, posted once with the backend's verdict (and saying so when no status was
posted, or when a Slack alert failed) → job summary → outputs.

Pull request runs (`pull_request` / `pull_request_target`) and merge queue runs
(`merge_group`) are analyzed; other events are skipped. A merge queue run analyzes the
queue commit against its parent, the backend posts the status on the queue commit, and
the report goes to the job summary only (a merge queue has no PR comment to post).

All costs are USD per run of each query. The Action exits 0 unless `fail_on_error`
is true; problems are reported in the log and the PR comment instead.
"""

from __future__ import annotations

import os
import re
import sys
import time
import warnings
from urllib.parse import urlparse

from comment import (
    COMMENT_MARKER,
    EARLIER_STATUS_NOTE,
    KNOWN_STATUSES,
    NOT_COSTED,
    NOT_ENFORCED_REASONS,
    SLACK_DOCS_URL,
    SLACK_SOURCE_DASHBOARD,
    SLACK_SOURCE_YML,
    STRUCTURAL_NOT_RUN,
    UNCONFIRMED_REASONS,
    no_api_key_cause,
    status_note,
    truncate_summary,
    unestimated_removal,
)
from config import DEFAULT_RUNS_PER_MONTH, ActionConfig, ThresholdConfig, load_config, settable_parameter

# google-api-core warns at import that Google will stop supporting the image's Python
# version in its future releases. Customers can't act on it, and the lock pins the version.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\.api_core\._python_version_support")

_DEFAULT_API_URL = "https://diffcost.dev"
_ALLOWED_HOSTS = frozenset({"diffcost.dev", "api.diffcost.dev"})

# Stop starting new dry runs after this many seconds (BigQuery incidents can stall each call).
DRY_RUN_BUDGET_SECONDS = 600
MAX_QUERIES_SENT = 500
# Backend `enforcement` values for which the `diffcost/cost-check` status was posted
_STATUS_POSTED = ("active", "plan")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# The limits the backend reports having applied
_LIMIT_KEYS = ("threshold_absolute_usd", "threshold_relative_pct")
# Structural findings at these risk levels make a "not costed" query count as unknown
_STRUCTURAL_RISKS = ("high", "medium")


def _get_api_url() -> str:
    """Get and validate the DiffCost API URL. Enforces HTTPS and strict domain allowlist."""
    url = os.environ.get("DIFFCOST_API_URL", _DEFAULT_API_URL)
    parsed = urlparse(url)
    # Allow plain HTTP for local development only
    if parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1"):
        return url
    # Production: require HTTPS and exact hostname match (prevents domain suffix attacks
    # like "api.diffcost.dev.attacker.com")
    if parsed.scheme == "https" and parsed.hostname in _ALLOWED_HOSTS:
        return url
    log_warning(f"DIFFCOST_API_URL must be an official DiffCost domain. Falling back to {_DEFAULT_API_URL}")
    return _DEFAULT_API_URL


def set_output(name: str, value: str) -> None:
    """Set a GitHub Actions output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if not output_file:
        return
    try:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")
    except OSError as e:
        log_warning(f"Could not write output `{name}`: {e}")


def _one_line(text) -> str:
    """Text for a log line: workflow commands end at a newline."""
    return " ".join(str(text).split())


# Log lines carry values from the pull request (file and model names, error text), and a
# file name may contain a newline. A workflow command ends at a newline, and the runner
# would read the rest as a new line, which could itself be a command (`::error::…`,
# `::stop-commands::…`). Every log line is therefore written as one line.


def log_info(msg: str) -> None:
    print(_one_line(msg))


def log_notice(msg: str) -> None:
    print(f"::notice::{_one_line(msg)}")


def log_warning(msg: str) -> None:
    print(f"::warning::{_one_line(msg)}")


def log_error(msg: str) -> None:
    print(f"::error::{_one_line(msg)}")


def detect_adapter(config: ActionConfig) -> str:
    """Pick the adapter from inputs and the base branch. Raises ManifestError (reported as
    "could not run") rather than falling back to scanning uncompiled dbt models as SQL."""
    from pathlib import Path

    from adapters.dbt import ManifestError

    workspace = Path(config.workspace) if config.workspace else Path(".")

    if config.manifest_path:
        if (workspace / config.manifest_path).exists():
            return "dbt"
        projects = config.base_dbt_project_dirs() or []
        hint = ""
        if [d for d in projects if d]:
            hint = " For a project in a subdirectory it is usually " + ", ".join(
                f"`{d}/target/manifest.json`" for d in projects if d
            ) + "."
        raise ManifestError(
            f"Manifest not found at `{config.manifest_path}` (the `manifest_path` input). "
            f"Run `dbt compile --target prod` before DiffCost.{hint}"
        )

    if config.sql_paths:
        return "sql_files"

    # Auto-detect a dbt project, at the repo root or in a subdirectory. A PR must not be
    # able to switch a plain-SQL repo to the dbt adapter by committing a manifest, so the
    # project's dbt_project.yml must already exist on the base commit. None: the base
    # can't be read; the adapters then report the git error, so it isn't masked here.
    projects = config.base_dbt_project_dirs()
    found = []
    for project in [""] if projects is None else projects:
        rel = f"{project}/{_AUTO_MANIFEST}" if project else _AUTO_MANIFEST
        if (workspace / rel).exists():
            found.append(rel)
    if projects is not None and "" not in projects and (workspace / _AUTO_MANIFEST).exists():
        log_warning(f"Ignoring `{_AUTO_MANIFEST}`: the base branch has no dbt_project.yml.")
    if len(found) > 1:
        raise ManifestError(
            f"Found manifests for several dbt projects ({', '.join(f'`{m}`' for m in found)}). "
            "DiffCost analyzes one dbt project per pull request: set `manifest_path` to the one to analyze. "
            "Running DiffCost more than once on the same pull request is not supported (the runs would share "
            "one commit status and one PR comment)."
        )
    if found:
        config.manifest_path = found[0]
        return "dbt"
    if projects:
        where = ", ".join(f"`{d or '.'}`" for d in projects)
        raise ManifestError(
            f"The base branch has a dbt project ({where}) but no compiled manifest was found. "
            "Run `dbt compile --target prod` before DiffCost (and set `manifest_path` for a project in a "
            "subdirectory), or set `sql_paths` to analyze plain .sql files instead."
        )

    # Default: scan all .sql files
    config.sql_paths = "**/*.sql"
    return "sql_files"


_AUTO_MANIFEST = "target/manifest.json"


# --- Summaries --------------------------------------------------------------------


def _empty_summary() -> dict:
    return {
        "total_before": None,
        "total_after": None,
        "total_delta": 0.0,
        "total_bytes_before": None,
        "total_bytes_after": None,
        "files_analyzed": 0,
        "items_unknown": 0,
        "items_not_costed": 0,
        "known": 0,
        "worst": None,
        "worst_pct": None,
        "max_regression_pct": 0.0,
    }


def summarize(results: list[dict]) -> dict:
    """Per-run totals over items with a known cost, plus the worst single query."""
    summary = _empty_summary()
    summary["files_analyzed"] = len(results)
    total_before = total_after = 0.0
    bytes_before = bytes_after = 0
    has_bytes = False

    for r in results:
        rec = r["cost"]
        if rec["status"] == NOT_COSTED and not rec.get("blocking"):
            summary["items_not_costed"] += 1
            continue
        if rec["status"] not in KNOWN_STATUSES:
            summary["items_unknown"] += 1
            continue
        if unestimated_removal(rec):
            continue  # a deleted query BigQuery couldn't estimate: no figure to sum, never blocks
        summary["known"] += 1
        total_before += rec["cost_before"] or 0.0
        total_after += rec["cost_after"] or 0.0
        dr = r.get("dry_run") or {}
        if dr.get("bytes_before") is not None or dr.get("bytes_after") is not None:
            has_bytes = True
            bytes_before += dr.get("bytes_before") or 0
            bytes_after += dr.get("bytes_after") or 0

        delta = rec["delta"] or 0.0
        if summary["worst"] is None or delta > summary["worst"]["delta"]:
            summary["worst"] = {"name": r["name"], "delta": delta, "pct": rec["pct"]}
        if rec["pct"] is not None and (summary["worst_pct"] is None or rec["pct"] > summary["worst_pct"]["pct"]):
            summary["worst_pct"] = {"name": r["name"], "pct": rec["pct"]}

    if summary["known"]:
        summary["total_before"] = total_before
        summary["total_after"] = total_after
        summary["total_delta"] = total_after - total_before
    if has_bytes:
        summary["total_bytes_before"] = bytes_before
        summary["total_bytes_after"] = bytes_after
    if summary["worst_pct"]:
        summary["max_regression_pct"] = max(0.0, summary["worst_pct"]["pct"])
    return summary


def evaluate_thresholds(
    thresholds: ThresholdConfig, summary: dict, editions: bool = False
) -> tuple[bool, list[str]]:
    """Per-run thresholds on the worst single query. Unknown costs block when a threshold is set.

    With Editions (slot) pricing the report shows scan volume only, so the dollar figure in
    a reason is labelled as the on-demand equivalent the threshold is compared against.
    """
    from comment import format_pct_change, format_usd_change, format_usd_limit

    if not thresholds.is_set():
        return False, []

    reasons: list[str] = []
    worst = summary["worst"]
    if thresholds.absolute_usd is not None and worst and worst["delta"] > thresholds.absolute_usd:
        label = " (on-demand equivalent)" if editions else ""
        reasons.append(
            f"{worst['name']}: +{format_usd_change(worst['delta'], thresholds.absolute_usd)}/run{label} exceeds the "
            f"{format_usd_limit(thresholds.absolute_usd)}/run per-query threshold"
        )
    worst_pct = summary["worst_pct"]
    if thresholds.relative_pct is not None and worst_pct and worst_pct["pct"] > thresholds.relative_pct:
        reasons.append(
            f"{worst_pct['name']}: +{format_pct_change(worst_pct['pct'], thresholds.relative_pct)}% exceeds the "
            f"{thresholds.relative_pct:g}% per-query threshold"
        )
    unknown = summary["items_unknown"]
    if unknown:
        noun = "query" if unknown == 1 else "queries"
        reasons.append(f"{unknown} {noun} could not be costed, so the cost policy cannot be confirmed")
    return bool(reasons), reasons


def _set_outputs(
    summary: dict,
    blocked: bool,
    runs_per_month: int,
    status: str = "ok",
    policy_exceeded: bool = False,
) -> None:
    """`blocked`: the failing commit status was posted. `policy_exceeded`: the cost policy
    was exceeded, enforced or not. `status`: ok | no_changes | error."""
    before_b = summary["total_bytes_before"] or 0
    after_b = summary["total_bytes_after"] or 0
    worst = summary["worst"]
    set_output("analysis_status", status)
    set_output("total_cost_delta", f"{summary['total_delta']:.4f}")
    set_output("monthly_cost_delta", f"{summary['total_delta'] * runs_per_month:.4f}")
    set_output("max_query_cost_delta", f"{max(worst['delta'], 0.0) if worst else 0.0:.4f}")
    set_output("max_regression_pct", f"{summary['max_regression_pct']:.1f}")
    set_output("files_analyzed", str(summary["files_analyzed"]))
    set_output("items_unknown", str(summary["items_unknown"]))
    set_output("total_bytes_before", str(before_b))
    set_output("total_bytes_after", str(after_b))
    set_output("scan_multiplier", f"{(after_b / before_b) if before_b > 0 else 1.0:.2f}")
    set_output("blocked", "true" if blocked else "false")
    set_output("policy_exceeded", "true" if policy_exceeded else "false")


# --- Main flow ------------------------------------------------------------------------


def run(config: ActionConfig) -> int:
    """Main execution flow. Returns the process exit code (non-zero only with fail_on_error)."""
    from adapters.dbt import ManifestError, find_changed_dbt_models
    from adapters.sql_files import find_changed_sql
    from comment import format_comment, format_failure_comment, post_or_update_comment
    from gitutil import GitError

    failure_code = 1 if config.fail_on_error else 0

    if not config.pr_number and not config.merge_group:
        log_warning("Not running in a pull request context. Skipping.")
        _set_outputs(_empty_summary(), False, config.default_runs_per_month, status="no_changes")
        return 0

    if config.merge_group:
        which = f"#{config.pr_number}" if config.pr_number else "an unknown pull request"
        log_info(f"Merge queue run for {which} (base: {config.resolved_base_ref[:12] or 'unresolved'})")
    # Why the `api_key` input may be empty ("" when a key is set): the run posts no status.
    no_key_cause = "" if config.api_key else no_api_key_cause(config.fork_pr, config.dependabot_pr)
    if no_key_cause:
        log_notice(
            "No DiffCost API key (the api_key input is empty), so no diffcost/cost-check status is posted. "
            f"{no_key_cause[0].upper()}{no_key_cause[1:]}."
        )

    def fail(message: str) -> int:
        log_error(f"DiffCost could not run: {message}")
        if config.api_key:
            log_warning(
                "No diffcost/cost-check commit status was posted. If that check is required, the pull request "
                f"can't merge until DiffCost runs successfully. {EARLIER_STATUS_NOTE}"
            )
        body = format_failure_comment(message, no_status=bool(config.api_key))
        if config.github_token and not config.merge_group:
            post_or_update_comment(config.github_repository, config.pr_number, config.github_token, body)
        _write_step_summary(body, _merge_queue_header(config))
        _set_outputs(_empty_summary(), False, config.default_runs_per_month, status="error")
        return failure_code

    if config.config_error:
        return fail(config.config_error)

    # 1. Dashboard config first: pricing and thresholds apply to the analysis
    plan = _get_plan(config)

    # 2. Detect adapter and find changed SQL
    try:
        adapter = detect_adapter(config)
        log_info(f"Using adapter: {adapter} (base: {config.base_label})")
        changed = find_changed_dbt_models(config) if adapter == "dbt" else find_changed_sql(config)
    except (GitError, ManifestError) as e:
        return fail(str(e))

    if not changed:
        return _handle_no_changes(config, no_key_cause)

    log_info(f"Found {len(changed)} changed SQL file(s)/model(s)")

    # 3. Analyze each changed item (one failing item never stops the run)
    from analysis.bq_dryrun import create_bq_client
    from analysis.table_meta import TableMetaCache

    bq_client = create_bq_client(config)
    started_at = time.monotonic()
    meta_cache = TableMetaCache(bq_client) if bq_client is not None else None
    results = [_analyze_item(item, bq_client, config, started_at, meta_cache) for item in changed]
    if meta_cache is not None and meta_cache.counts[1]:
        found, tried = meta_cache.counts
        log_info(f"Read the metadata of {found} of {tried} referenced table(s)")
    has_bq = bq_client is not None
    summary = summarize(results)

    # 4. Local threshold check: paid plans, and runs whose dashboard config could not be
    # fetched with an API key (the comment then says the .diffcost.yml policy was exceeded
    # but not enforced, instead of showing a Free-plan upsell). The backend decides.
    config_unavailable = bool(config.api_key and config.backend_error)
    is_blocked, block_reasons = False, []
    if "merge_blocking" in config.features or config_unavailable:
        is_blocked, block_reasons = evaluate_thresholds(
            config.thresholds, summary, editions=config.bq_pricing_model == "editions"
        )

    # 5. Send the numbers to the backend first: it evaluates the cost policy and posts the
    # commit status, so the comment below is posted once, with the verdict GitHub shows.
    exit_code = 0
    not_enforced = ""
    note = ""
    unconfirmed = False
    merge_on = "merge_blocking" in config.features and not config_unavailable
    # Where the limits came from: the backend's answer, or `.diffcost.yml` for a local check
    # of the file ("" when the local check used dashboard limits of an unknown scope)
    threshold_source = "" if config.dashboard_thresholds else "yml"
    backend_res = None
    if config.api_key:
        backend_res = _send_to_backend(config, _build_payload(config, summary, results))
        if backend_res is None:
            reason_key = config.backend_error or "unreachable"
            if config_unavailable and threshold_source == "yml":
                threshold_source = "yml_local"
            if config.fail_on_error:
                log_error("DiffCost backend did not accept the results (fail_on_error is true).")
                exit_code = 1
            not_enforced = NOT_ENFORCED_REASONS[reason_key] if is_blocked else ""
            note, merge_on, unconfirmed = status_note(reason_key), False, reason_key in UNCONFIRMED_REASONS
            _warn_no_status(reason_key)
        else:
            backend_blocked = bool(backend_res.get("blocked"))
            enforcement = str(backend_res.get("enforcement") or "")
            if enforcement not in _STATUS_POSTED:
                reason_key = enforcement if enforcement in NOT_ENFORCED_REASONS else "unconfirmed"
                note, unconfirmed = status_note(reason_key), reason_key in UNCONFIRMED_REASONS
                _warn_no_status(reason_key)
            if backend_blocked and enforcement != "active":
                not_enforced = NOT_ENFORCED_REASONS.get(enforcement, NOT_ENFORCED_REASONS["unconfirmed"])
            merge_on = enforcement == "active"
            threshold_source = str(backend_res.get("threshold_source") or "")
            if backend_blocked != is_blocked or (
                backend_blocked and _backend_used_other_limits(config, backend_res, config_unavailable)
            ):
                # The backend's verdict is what the commit status says; the comment and Slack match it.
                is_blocked = backend_blocked
                block_reasons = _backend_reasons(config, summary, backend_res) if backend_blocked else []

    # 6. Slack: the .diffcost.yml webhook, only when the .diffcost.yml policy blocked this
    # PR (a dashboard threshold's own webhook is sent by the backend, so never both). Sent
    # before the comment, so the comment can say when an alert failed.
    slack_failures: list[tuple[str, str]] = []
    if config.slack_webhook and "slack" in config.features:
        # By now `is_blocked` is the backend's verdict when it answered, else the local one
        if is_blocked and threshold_source in ("yml", "yml_local"):
            slack_error = _send_slack_notification(config, summary, block_reasons)
            if slack_error:
                slack_failures.append((SLACK_SOURCE_YML, slack_error))
    backend_slack_error = _one_line((backend_res or {}).get("slack_error") or "")[:200]
    if backend_slack_error:
        slack_failures.append((SLACK_SOURCE_DASHBOARD, backend_slack_error))
    for source, reason in slack_failures:
        log_warning(f"The Slack alert for the {source} failed: {reason}. What to check: {SLACK_DOCS_URL}")

    # 7. PR comment, once, with the final verdict
    comment_body = format_comment(
        results=results,
        total_before=summary["total_before"] or 0.0,
        total_after=summary["total_after"] or 0.0,
        total_delta=summary["total_delta"],
        has_bq_creds=has_bq,
        blocked=is_blocked,
        block_reasons=block_reasons,
        plan=plan,
        trial_days_remaining=config.trial_days_remaining,
        bq_pricing_model=config.bq_pricing_model,
        runs_per_month=config.default_runs_per_month,
        items_unknown=summary["items_unknown"],
        worst_query=summary["worst"],
        not_enforced_reason=not_enforced,
        enforcement_unconfirmed=unconfirmed,
        items_not_costed=summary["items_not_costed"],
        policy_warnings=config.policy_warnings,
        bq_error=config.bq_client_error,
        trial_ended=config.trial_ended,
        status_note_line=note,
        merge_blocking_on=merge_on,
        service_problem=config_unavailable,
        no_key_cause=no_key_cause,
        slack_failures=slack_failures,
        subscription_inactive=config.subscription_inactive,
        threshold_source=threshold_source,
        query_parameters=(
            [(p.name, p.display) for p in config.query_parameters]
            if any(r.get("priced_with_parameters") for r in results) else None
        ),
    )
    if config.merge_group:
        pass  # a merge queue has no PR comment: the job summary is the report
    elif config.github_token:
        post_or_update_comment(config.github_repository, config.pr_number, config.github_token, comment_body)
    else:
        log_warning("No GitHub token available (set the `github_token` input). Printing the comment instead:")
        print(comment_body)

    _write_step_summary(comment_body, _merge_queue_header(config))

    # 8. Outputs
    enforced = is_blocked and not not_enforced
    _set_outputs(summary, enforced, config.default_runs_per_month, status="ok", policy_exceeded=is_blocked)

    if enforced:
        # Merge blocking is enforced by the `diffcost/cost-check` commit status; the step itself passes.
        log_error(f"Merge blocked by DiffCost cost policy: {'; '.join(block_reasons)}")
    elif is_blocked and unconfirmed:
        log_warning(
            f"DiffCost cost policy exceeded; enforcement could not be confirmed ({not_enforced}): "
            f"{'; '.join(block_reasons)}"
        )
    elif is_blocked:
        log_warning(f"DiffCost cost policy exceeded but not enforced ({not_enforced}): {'; '.join(block_reasons)}")

    return 1 if _policy_file_failed(config) else exit_code


def _backend_reasons(config: ActionConfig, summary: dict, backend_res: dict) -> list[str]:
    """Reasons for the backend's block, worded as the local check words them (naming the
    query) when that check, rerun with the limits the backend reported applying, gives the
    same reasons; otherwise the backend's own reasons (it never sees query names)."""
    editions = config.bq_pricing_model == "editions"
    theirs = [str(r) for r in (backend_res.get("block_reasons") or [])]
    if theirs and any(k in backend_res for k in _LIMIT_KEYS):
        try:
            absolute, relative = (None if backend_res.get(k) is None else float(backend_res[k]) for k in _LIMIT_KEYS)
        except (TypeError, ValueError):
            absolute = relative = None  # unreadable limits: keep the backend's reasons
        blocked, mine = evaluate_thresholds(ThresholdConfig(absolute, relative), summary, editions=editions)
        if blocked and len(mine) == len(theirs):
            return mine
    if editions:
        theirs = [f"{r} (on-demand equivalent)" if "$" in r else r for r in theirs]
    return theirs


def _backend_used_other_limits(config: ActionConfig, backend_res: dict, config_unavailable: bool) -> bool:
    """Whether the backend blocked on other limits than the local check used, so the local
    reasons are replaced by ones for the limits the commit status used (`_backend_reasons`).

    That happens when the dashboard config could not be fetched (the local check used
    .diffcost.yml) but the backend applied a dashboard threshold, or when a dashboard
    threshold changed between the config fetch and the results.
    """
    if config_unavailable and backend_res.get("threshold_source") in ("org", "repo"):
        return True
    if not any(k in backend_res for k in _LIMIT_KEYS):
        return False
    local = (config.thresholds.absolute_usd, config.thresholds.relative_pct)
    for mine, key in zip(local, _LIMIT_KEYS):
        theirs = backend_res.get(key)
        if (mine is None) != (theirs is None):
            return True
        try:
            if mine is not None and abs(float(mine) - float(theirs)) > 1e-6:
                return True
        except (TypeError, ValueError):
            return True
    return False


def _warn_no_status(reason_key: str) -> None:
    reason = NOT_ENFORCED_REASONS.get(reason_key, NOT_ENFORCED_REASONS["unconfirmed"])
    if reason_key in UNCONFIRMED_REASONS:
        log_warning(
            f"The diffcost/cost-check commit status could not be confirmed ({reason}). "
            "If that check is required and the commit has no status yet, the pull request can't merge until one "
            "is posted. A status shown from an earlier run on this commit may be out of date."
        )
        return
    log_warning(
        f"No diffcost/cost-check commit status was posted ({reason}). "
        f"If that check is required, the pull request can't merge until the commit has one. {EARLIER_STATUS_NOTE}"
    )


def _policy_file_failed(config: ActionConfig) -> bool:
    """`fail_on_error` also covers a .diffcost.yml that could not be read or parsed (its
    thresholds were dropped, so the cost policy is weaker than configured)."""
    if config.fail_on_error and config.policy_warnings:
        log_error(
            "The cost policy file could not be applied (fail_on_error is true): " + "; ".join(config.policy_warnings)
        )
        return True
    return False


def _merge_queue_header(config: ActionConfig) -> str:
    """The job summary's first line on a merge queue run ("" otherwise)."""
    if not config.merge_group:
        return ""
    if config.pr_number:
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        pr = f"[#{config.pr_number}]({server}/{config.github_repository}/pull/{config.pr_number})"
    else:
        pr = "a pull request"
    return f"🔀 **Merge queue run** for {pr}. Merge queue runs post no PR comment: this job summary is the report."


def _write_step_summary(body: str, header: str = "") -> None:
    """Also write the report to the job summary: it is the only place the report shows
    on pull requests from forks, whose read-only token can't comment, and on merge queue
    runs (`header` says which PR the queue run is for)."""
    path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not path or not body:
        return
    body = body.replace(COMMENT_MARKER, "").strip()
    body = truncate_summary(f"{header}\n\n{body}" if header else body)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(body + "\n\n")
    except OSError as e:
        log_warning(f"Could not write the job summary: {type(e).__name__}")


def _handle_no_changes(config: ActionConfig, no_key_cause: str) -> int:
    from comment import format_clean_comment, format_no_changes_comment, post_or_update_comment

    log_info("No SQL changes detected.")
    summary = _empty_summary()

    exit_code = 0
    reason_key = ""
    if config.api_key:
        # files_analyzed=0: the backend posts a passing status so a required check never hangs.
        res = _send_to_backend(config, _build_payload(config, summary, []))
        if res is None:
            reason_key = config.backend_error or "unreachable"
            if config.fail_on_error:
                log_error("DiffCost backend did not accept the results (fail_on_error is true).")
                exit_code = 1
        else:
            enforcement = str(res.get("enforcement") or "")
            if enforcement not in _STATUS_POSTED:
                reason_key = enforcement if enforcement in NOT_ENFORCED_REASONS else "unconfirmed"
    note = ""
    if reason_key:
        _warn_no_status(reason_key)
        note = status_note(reason_key)

    # Post the clean comment with comment_on_clean; otherwise only replace an earlier
    # DiffCost comment (e.g. a stale "Merge blocked" result), unless a required check may
    # now wait for a status that wasn't posted: then the PR must say why. A new comment
    # never says "earlier results no longer apply" (there were none). A merge queue run
    # has no PR comment.
    clean = format_clean_comment(note, no_key_cause)
    body = clean if config.comment_on_clean else format_no_changes_comment(note, no_key_cause)
    if config.github_token and not config.merge_group:
        post_or_update_comment(
            config.github_repository, config.pr_number, config.github_token, body,
            create=config.comment_on_clean or bool(note), body_if_new=clean,
        )

    _write_step_summary(clean, _merge_queue_header(config))
    _set_outputs(summary, False, config.default_runs_per_month, status="no_changes")
    return 1 if _policy_file_failed(config) else exit_code


def _table_info(dr: dict | None, table_meta: dict) -> dict:
    """Keyword arguments for analyze_structural_changes, from the metadata of the tables the
    base version's dry run read (a removed filter comes from the base version, so a table only
    the PR reads can't explain it):

    - `partition_columns`: partition column -> labels of the tables it partitions
      ("analytics.events (4.20 TiB table, partitioned daily)");
    - `cluster_columns`: clustering column -> the tables it clusters ("analytics.events").

    Empty without metadata: structural analysis then guesses from column names.
    """
    if not dr or not table_meta:
        return {}
    partition: dict[str, list[str]] = {}
    cluster: dict[str, list[str]] = {}
    for t in dict.fromkeys(dr.get("tables_before") or []):
        m = table_meta.get(t)
        if m is None or not m.stored:
            continue
        if m.partition_kind:
            for col in m.partition_columns:
                partition.setdefault(col, []).append(m.label)
        for col in m.clustering_fields:
            cluster.setdefault(col.lower(), []).append(m.short_name)
    return {"partition_columns": partition, "cluster_columns": cluster}


def _analyze_item(item, bq_client, config: ActionConfig, started_at: float, meta_cache=None) -> dict:
    """`meta_cache` (TableMetaCache): reads the metadata of the tables the dry runs reference."""
    from analysis.accuracy import assess, unbound_parameters
    from analysis.bq_dryrun import dry_run_pair
    from adapters.sql_files import exempt_entry
    from analysis.structural import analyze_structural_changes
    from comment import classify_cost

    meta = item.metadata or {}
    result = {
        "name": item.name,
        "structural_findings": [],
        "dry_run": None,
        # `ignored_paths` entries the comment suggests for SQL that wasn't analyzed
        "ignore_globs": list(meta.get("ignore_globs") or []),
    }
    baseline_unknown = bool(meta.get("baseline_unknown"))
    unknown_reason = str(meta.get("unknown_reason") or "")
    # .diffcost.yml's default dataset and query parameter values are for plain SQL files:
    # dbt's compiled SQL names its relations in full and has no parameters
    plain_sql = item.adapter == "sql_files"
    options = (
        {"default_dataset": config.default_dataset, "query_parameters": config.query_parameters} if plain_sql else {}
    )

    # A modified dbt model without a baseline is unknown cost whatever the dry run says,
    # so it doesn't spend the shared dry-run time budget
    table_meta: dict = {}
    if bq_client is not None and not unknown_reason and not baseline_unknown:
        if time.monotonic() - started_at > DRY_RUN_BUDGET_SECONDS:
            dr = {"error": "skipped: dry-run time budget exceeded", "cost_before": None, "cost_after": None}
        else:
            # The comment says which values were used (format_comment's `query_parameters`)
            result["priced_with_parameters"] = bool(plain_sql and config.query_parameters)
            try:
                dr = dry_run_pair(bq_client, item.production_sql, item.pr_sql, config.cost_per_tb, **options)
            except Exception as e:
                dr = {"error": f"dry run failed ({type(e).__name__})", "cost_before": None, "cost_after": None}
        if not dr.get("error"):
            tables = (dr.get("tables_before") or []) + (dr.get("tables_after") or [])
            try:
                table_meta = meta_cache.get(tables) if meta_cache is not None and tables else {}
            except Exception as e:  # defensive: metadata only sharpens the analysis
                log_warning(f"Reading table metadata failed for {item.name}: {type(e).__name__}")
            assess(dr, table_meta)
        result["dry_run"] = dr

    structural_ran = False  # both versions were parsed and every check ran
    try:
        if item.production_sql and item.pr_sql:
            findings = analyze_structural_changes(
                item.production_sql, item.pr_sql, **_table_info(result["dry_run"], table_meta)
            )
            result["structural_findings"] = findings
            structural_ran = not any(f.skipped for f in findings)
    except Exception as e:  # defensive: structural analysis never fails the run
        log_warning(f"Structural analysis failed for {item.name}: {type(e).__name__}")

    rec = classify_cost(
        result["dry_run"],
        has_production=item.production_sql is not None,
        has_pr=item.pr_sql is not None,
        baseline_unknown=baseline_unknown,
        unknown_reason=unknown_reason,
    )
    if rec["status"] == NOT_COSTED:
        if item.adapter != "dbt":
            # The exemption is for dbt models that read objects created by other models in
            # this PR. A plain SQL file that reads a missing table or column is unknown cost.
            rec.update(status="unknown", reason="the PR version reads a table or column that does not exist yet")
        elif not _explained_by_this_pr(meta, result["dry_run"] or {}):
            # The exemption needs something upstream of this model (transitively) that the
            # PR adds or changes. Otherwise the error is not explained by this PR (a typo, a
            # manifest compiled against another target).
            dr = result["dry_run"] or {}
            if dr.get("pr_error"):
                reason = f"PR dry run failed, and nothing upstream changes in this PR: {dr['pr_error']}"
            elif dr.get("pr_missing_object"):
                reason = "the PR version reads an object that does not exist, and nothing upstream creates it in this PR"
            else:
                reason = "the PR version reads an object that does not exist, and nothing upstream changes in this PR"
            rec.update(status="unknown", reason=reason)
        elif any(f.risk in _STRUCTURAL_RISKS for f in result["structural_findings"]):
            # A dbt model that reads objects this PR creates blocks (as unknown cost) only
            # when structural analysis also flags a cost risk on it...
            rec["blocking"] = True
        elif item.production_sql is not None and not structural_ran:
            # ...or, for a model that already exists, when structural analysis could not run
            # on it (unparseable SQL, a failed check): "no risk found" is not known (fail closed).
            # A new model has no earlier version to compare, so it stays exempt.
            rec.update(blocking=True, blocking_note=STRUCTURAL_NOT_RUN)
    unbound = unbound_parameters(result["dry_run"] or {})
    if rec["status"] == "unknown" and unbound:
        # Fixed by giving the parameters values, not by exempting the file, except for those
        # `query_parameters` can't set (arrays, structs, positional `?`). The comment says how
        # for plain SQL files (`query_parameters` doesn't apply to dbt models).
        if plain_sql:
            result["unbound_parameters"] = unbound
            if item.path and not all(settable_parameter(name, ptype) for name, ptype in unbound):
                result["parameter_exempt_entry"] = exempt_entry(item.path)
    elif rec["status"] == "unknown" and (result["dry_run"] or {}).get("unknown_reason") and item.path:
        # BigQuery can't estimate this query before it runs: the comment says how to exempt the file
        result["exempt_entry"] = exempt_entry(item.path)
    result["cost"] = rec
    return result


def _explained_by_this_pr(meta: dict, dry_run: dict) -> bool:
    """Whether a dbt PR-side "Unrecognized name" / "Not found" can mean "reads something this
    PR creates": a "Not found" whose object name could be read must be a relation an upstream
    node creates in this PR; otherwise something upstream must be new or changed."""
    from adapters.dbt import relation_created

    missing = dry_run.get("pr_missing_object")
    if missing:
        return relation_created(missing, list(meta.get("new_upstream_relations") or []))
    return bool(meta.get("changed_upstream"))


# --- Backend --------------------------------------------------------------------------


_CONFIG_NOT_APPLIED = "Dashboard settings were not applied; .diffcost.yml thresholds are checked locally."


def _get_plan(config: ActionConfig) -> str:
    """Fetch the org's dashboard config. Returns the effective plan ('free' without an api_key).

    Side effects on `config`: features, trial days, BigQuery pricing (server wins),
    thresholds (dashboard thresholds replace .diffcost.yml ones when set). On failure,
    `config.backend_error` says why ("rejected", "rate_limited" or "unreachable").
    """
    if not config.api_key:
        return "free"

    import requests

    for attempt in range(2):
        try:
            params = {"repo": config.github_repository} if config.github_repository else {}
            resp = requests.get(
                f"{_get_api_url()}/v1/config",
                headers={"Authorization": f"Bearer {config.api_key}"},
                params=params,
                timeout=12,
            )
        except requests.RequestException as e:
            if attempt == 0:
                continue
            log_warning(f"Could not reach DiffCost backend ({type(e).__name__}). {_CONFIG_NOT_APPLIED}")
            config.backend_error = "unreachable"
            return "free"

        if resp.status_code != 200:
            if resp.status_code >= 500 and attempt == 0:
                continue
            if resp.status_code in (401, 403):
                log_warning(
                    f"DiffCost rejected the API key for {config.github_repository} (HTTP {resp.status_code}). "
                    "Check the DIFFCOST_API_KEY secret and the key's repository scope in DiffCost Settings. "
                    + _CONFIG_NOT_APPLIED
                )
                config.backend_error = "rejected"
            elif resp.status_code == 429:
                log_warning(f"DiffCost rate-limited this runner (HTTP 429). {_CONFIG_NOT_APPLIED}")
                config.backend_error = "rate_limited"
            else:
                log_warning(f"DiffCost backend returned HTTP {resp.status_code}. {_CONFIG_NOT_APPLIED}")
                config.backend_error = "unreachable"
            return "free"

        try:
            data = resp.json()
        except ValueError:
            log_warning(f"DiffCost backend returned invalid JSON. {_CONFIG_NOT_APPLIED}")
            config.backend_error = "unreachable"
            return "free"
        config.backend_error = ""
        return _apply_server_config(config, data if isinstance(data, dict) else {})
    return "free"


def _apply_server_config(config: ActionConfig, data: dict) -> str:
    plan = str(data.get("plan") or "free")
    config.trial_days_remaining = data.get("trial_days_remaining")
    config.trial_ended = bool(data.get("trial_ended"))
    config.subscription_inactive = data.get("subscription_inactive") is True

    features = data.get("features")
    config.features = [str(f) for f in features] if isinstance(features, list) else []

    bq_pricing = data.get("bq_pricing") or {}
    model = str(bq_pricing.get("model") or "").lower()
    if model in ("on_demand", "editions"):
        config.bq_pricing_model = model
    try:
        server_rate = float(bq_pricing.get("cost_per_tb"))
        if server_rate > 0:
            config.cost_per_tb = server_rate
    except (TypeError, ValueError):
        pass

    server_thresholds = data.get("thresholds") or {}
    absolute = server_thresholds.get("absolute_usd")
    relative = server_thresholds.get("relative_pct")
    if absolute is not None or relative is not None:
        # A dashboard threshold replaces the .diffcost.yml thresholds as a whole
        config.dashboard_thresholds = True
        config.thresholds = ThresholdConfig(
            absolute_usd=float(absolute) if absolute is not None else None,
            relative_pct=float(relative) if relative is not None else None,
        )

    return plan


def _build_payload(config: ActionConfig, summary: dict, results: list[dict]) -> dict:
    """The numbers sent to the backend. No SQL, file paths, model or table names, and no
    per-query identifier: each query is only its bytes and per-run costs."""
    sha = (config.commit_sha or "").lower()
    # A deleted query BigQuery couldn't estimate has no figure to send
    costed = [r for r in results if r["cost"]["status"] in KNOWN_STATUSES and not unestimated_removal(r["cost"])]
    # The backend sees at most MAX_QUERIES_SENT queries and takes the worst one from them:
    # send the largest increases first, and always include the largest % increase.
    costed.sort(key=lambda r: r["cost"]["delta"] or 0.0, reverse=True)
    selected = costed[:MAX_QUERIES_SENT]
    with_pct = [r for r in costed if r["cost"]["pct"] is not None]
    if with_pct and len(costed) > MAX_QUERIES_SENT:
        top_pct = max(with_pct, key=lambda r: r["cost"]["pct"])
        if top_pct not in selected:
            selected[-1] = top_pct
    queries = []
    for r in selected:
        rec = r["cost"]
        dr = r.get("dry_run") or {}
        queries.append({
            "bytes_before": dr.get("bytes_before") if rec["status"] != "new" else None,
            "bytes_after": dr.get("bytes_after") if rec["status"] != "removed" else None,
            "cost_before": rec["cost_before"],
            "cost_after": rec["cost_after"],
        })

    thresholds = None
    if config.yml_thresholds.is_set():
        thresholds = {
            "absolute_usd": config.yml_thresholds.absolute_usd,
            "relative_pct": config.yml_thresholds.relative_pct,
        }

    return {
        "repo": config.github_repository,
        "pr_number": config.pr_number,
        "pr_author": config.pr_author if config.send_author else "",
        "commit_sha": sha if _SHA_RE.match(sha) else "",
        "total_cost_before": summary["total_before"],
        "total_cost_after": summary["total_after"],
        "total_cost_delta": summary["total_delta"] if summary["known"] else None,
        "total_bytes_before": summary["total_bytes_before"],
        "total_bytes_after": summary["total_bytes_after"],
        "files_analyzed": summary["files_analyzed"],
        "items_unknown": summary["items_unknown"],
        "runs_per_month": config.default_runs_per_month,
        "queries": queries,
        "thresholds": thresholds,
        # A merge queue run: the status goes on the queue commit, and the dashboard keeps
        # the run out of the PR's history and the totals
        "merge_group": config.merge_group,
    }


def _send_to_backend(config: ActionConfig, payload: dict) -> dict | None:
    """POST /v1/comparisons. Returns the response dict, or None if the backend did not accept
    it or its reply carries no verdict.

    Never raises: a rejected or failed submission is a warning (the caller applies
    fail_on_error), and `config.backend_error` says why. The POST is not idempotent (it
    stores a comparison and posts the commit status), so it is retried only when the
    request cannot have been processed: a connection failure, or a 502/503 from the proxy.
    """
    import requests

    config.backend_error = ""
    for attempt in range(2):
        try:
            resp = requests.post(
                f"{_get_api_url()}/v1/comparisons",
                headers={"Authorization": f"Bearer {config.api_key}"},
                json=payload,
                timeout=12,
            )
        except requests.ConnectionError as e:  # includes ConnectTimeout
            if attempt == 0:
                continue
            log_warning(f"Failed to send results to the DiffCost backend: {type(e).__name__}")
            config.backend_error = "unreachable"
            return None
        except requests.Timeout:
            log_warning(
                "The DiffCost backend did not answer within 12 s. Not retried: it may already have "
                "recorded the results and posted the commit status."
            )
            config.backend_error = "timeout"
            return None
        except requests.RequestException as e:
            log_warning(f"Failed to send results to the DiffCost backend: {type(e).__name__}")
            config.backend_error = "unreachable"
            return None

        if resp.status_code in (200, 201):
            try:
                data = resp.json()
            except ValueError:
                data = None
            if isinstance(data, dict) and isinstance(data.get("blocked"), bool):
                return data
            # E.g. a proxy or captive portal's page. Not a verdict: the local verdict stands,
            # and the status is unconfirmed (the backend may have posted it).
            log_warning(
                f"The DiffCost backend's reply (HTTP {resp.status_code}) is not a DiffCost result, "
                "so the commit status could not be confirmed."
            )
            config.backend_error = "unconfirmed"
            return None
        if resp.status_code in (502, 503) and attempt == 0:
            continue

        detail = ""
        try:
            body = resp.json()
            detail = str(body.get("detail", ""))[:200] if isinstance(body, dict) else ""
        except (ValueError, AttributeError):
            pass
        if resp.status_code in (401, 403):
            log_warning(
                f"DiffCost rejected results for {config.github_repository} (HTTP {resp.status_code}"
                f"{': ' + detail if detail else ''}). The repository must belong to your DiffCost organization "
                "and be in the API key's scope."
            )
            config.backend_error = "rejected"
        elif resp.status_code == 429:
            log_warning("DiffCost rate-limited this runner (HTTP 429); the results were not accepted.")
            config.backend_error = "rate_limited"
        else:
            log_warning(f"DiffCost backend returned HTTP {resp.status_code}{': ' + detail if detail else ''}.")
            config.backend_error = "error" if resp.status_code >= 500 else "invalid"
        return None
    return None


def _slack_escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_URL_RE = re.compile(r"https?://\S+")


def _slack_detail(body: str, webhook: str) -> str:
    """Slack's error text (e.g. `no_service`) for the log and the comment, cut to 60
    characters. The webhook URL is a secret and a proxy may echo it, so the configured
    webhook and any URL are removed first."""
    if webhook:
        body = body.replace(webhook, "<url>")
    return _one_line(_URL_RE.sub("<url>", body))[:60]


def _send_slack_notification(config: ActionConfig, summary: dict, reasons: list[str]) -> str:
    """Tell the .diffcost.yml Slack webhook that this PR exceeds the cost policy (per-run figures).

    Returns "" when Slack accepted the message (HTTP 200 `ok`), else why it failed. The
    reason never contains the webhook URL (a secret): Slack answers a revoked webhook with e.g.
    HTTP 404 `no_service`, an archived channel with HTTP 410 `channel_is_archived`.
    """
    from comment import _format_cost

    try:
        import requests

        repo = config.github_repository
        pr = config.pr_number
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        pr_url = f"{server}/{repo}/pull/{pr}"
        delta = summary["total_delta"]
        runs = config.default_runs_per_month
        worst = summary["worst"]

        queue = " (merge queue)" if config.merge_group else ""
        text = f"🚫 DiffCost: <{pr_url}|{_slack_escape(repo)}#{pr}>{queue} exceeds the cost policy in .diffcost.yml"
        for reason in reasons[:5]:
            text += f"\n• {_slack_escape(reason)}"
        if worst and worst["delta"] > 0:
            text += (
                f"\nLargest increase: `{_slack_escape(worst['name'])}` +{_format_cost(worst['delta'])}/run "
                f"(≈{_format_cost(worst['delta'] * runs)}/mo at {runs} runs/mo)"
            )
        if summary["known"]:
            text += f"\nNet change across the PR: {'+' if delta >= 0 else ''}{_format_cost(delta)}/run"

        try:
            resp = requests.post(config.slack_webhook, json={"text": text}, timeout=5)
        except requests.Timeout:
            return "timed out"
        except requests.RequestException as e:
            return f"request failed ({type(e).__name__})"
        body = str(resp.text or "")
        if resp.status_code == 200 and body.strip() == "ok":
            return ""
        detail = _slack_detail(body, config.slack_webhook)
        return f"Slack returned HTTP {resp.status_code}" + (f": {detail}" if detail else "")
    except Exception as e:  # defensive: a Slack alert never fails the run
        return f"request failed ({type(e).__name__})"


def main() -> int:
    fail_on_error = os.environ.get("INPUT_FAIL_ON_ERROR", "false").strip().lower() == "true"
    try:
        cfg = load_config()
        fail_on_error = cfg.fail_on_error
        return run(cfg)
    except Exception as e:
        # Log only the exception type and message — never the full traceback,
        # which may contain secrets from local variables (api_key, gcp_credentials, etc.)
        log_error(f"DiffCost failed: {type(e).__name__}: {str(e)[:200]}")
        _set_outputs(_empty_summary(), False, DEFAULT_RUNS_PER_MONTH, status="error")
        return 1 if fail_on_error else 0


if __name__ == "__main__":
    sys.exit(main())
