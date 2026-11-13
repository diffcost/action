"""PR comment generation and posting via GitHub API.

All dollar figures are per run of each query. A monthly projection
(per-run x runs_per_month) is shown as secondary text only.
Every value that comes from the repository (file paths, dbt model names,
SQL snippets, BigQuery error text) is escaped before it is placed in markdown.
"""

from __future__ import annotations

import json
import os
import re

import requests

from analysis.table_meta import format_bytes as _format_bytes
from config import settable_parameter

COMMENT_MARKER = "<!-- diffcost-report -->"
# GitHub rejects comment bodies over 65,536 characters.
MAX_COMMENT_CHARS = 65_000
# GitHub rejects a step's job summary over 1 MiB (1,048,576 bytes of UTF-8).
MAX_SUMMARY_BYTES = 1_000_000
COMMENT_TRUNCATED_NOTE = (
    "*Report truncated to fit GitHub's comment size limit. "
    "A longer version is in the job summary of this workflow run.*"
)
SUMMARY_TRUNCATED_NOTE = "*Report truncated to fit GitHub's job summary size limit.*"
_MAX_COMMENT_PAGES = 50
_DEFAULT_BOT_LOGIN = "github-actions[bot]"

SETUP_DOCS_URL = "https://diffcost.dev/docs/bigquery-access"
TROUBLESHOOTING_URL = "https://diffcost.dev/docs/troubleshooting#diffcost-could-not-analyze-this-pr"
PRICING_URL = "https://diffcost.dev/pricing"
BILLING_URL = "https://diffcost.dev/dashboard/billing"
NO_API_KEY_DOCS_URL = "https://diffcost.dev/docs/troubleshooting#no-diffcost-api-key"
SLACK_DOCS_URL = "https://diffcost.dev/docs/troubleshooting#slack-alerts-dont-arrive"
THRESHOLDS_DOCS_URL = "https://diffcost.dev/docs/thresholds#where-to-set-thresholds"

# Where the limits of a blocked run came from, by the backend's `threshold_source`
THRESHOLD_SOURCES = {
    "repo": "this repository's dashboard threshold",
    "org": "the org-wide dashboard threshold (`*`)",
    "yml": "`.diffcost.yml` on the base branch",
    # The dashboard settings couldn't be fetched, so the file was checked on the runner; a
    # dashboard threshold, if any, would replace it once the DiffCost service is reachable
    "yml_local": "`.diffcost.yml` on the base branch, checked on the runner (the dashboard settings weren't fetched)",
}

# Statuses of a costed item (see classify_cost)
KNOWN_STATUSES = ("changed", "new", "removed")
UNKNOWN_STATUSES = ("unknown", "error")
# The PR side reads objects this PR creates; blocks only with a structural risk, or when
# structural analysis couldn't run on a model that already exists (see main)
NOT_COSTED = "not_costed"
NOT_COSTED_REASON = "depends on objects this PR creates"
# Why a Not costed row blocks: a structural cost risk, or (`blocking_note`) this
STRUCTURAL_NOT_RUN = "structural analysis could not run on it"

# Why a cost-policy breach was not enforced, by the backend's `enforcement` value
NOT_ENFORCED_REASONS = {
    "not_configured": "merge blocking is not set up on the DiffCost server",
    "no_installation": "the DiffCost GitHub App is not installed for this repository",
    "failed": "DiffCost could not post the commit status",
    "skipped": "no commit SHA was available for the commit status",
    "plan": "merge blocking is off on the Free plan",
    "suspended": "the DiffCost GitHub App is suspended in this organization",
    "unreachable": "the DiffCost service could not be reached",
    "rejected": "the DiffCost API key was rejected",
    "rate_limited": "the DiffCost service rate-limited the request",
    "invalid": "the DiffCost service did not accept the results",
    "error": "the DiffCost service returned an error",
    "timeout": "the DiffCost service did not answer in time",
    "unconfirmed": "DiffCost could not tell whether GitHub received the status",
}
# Reasons for which the status may or may not have been posted (the backend may have
# finished after the Action stopped waiting): never claim "no status was posted".
UNCONFIRMED_REASONS = ("timeout", "unconfirmed")

SUPPORT_EMAIL = "support@diffcost.dev"

# What to do when no `diffcost/cost-check` status was posted, by reason
_NO_STATUS_FIX = {
    "no_installation": "Install the DiffCost GitHub App on this repository, then re-run this job.",
    "suspended": (
        "A GitHub organization owner can unsuspend it in the organization's **Settings → GitHub Apps → DiffCost**, "
        "then re-run this job."
    ),
    "rejected": (
        "Check the DIFFCOST_API_KEY secret and the key's repository scope in DiffCost Settings, "
        "then re-run this job."
    ),
    "invalid": (
        f"Update to the latest `diffcost/action@v1` and re-run this job. If it persists, email {SUPPORT_EMAIL} "
        "with the warning from the job log."
    ),
    "not_configured": f"Merge blocking is not available on this DiffCost server. Contact {SUPPORT_EMAIL}.",
    "rate_limited": (
        "Too many requests for this API key, or too many failed API-key checks from this runner's IP address. "
        "Wait a minute, then re-run this job."
    ),
}


# A run that posts no status leaves the one an earlier run on the same commit posted (a
# re-run, a reopened PR) in place, so never say the PR "can't merge" without this.
EARLIER_STATUS_NOTE = "A status an earlier run posted on this commit still applies, passing or failing."


def status_note(reason_key: str) -> str:
    """One comment line: no commit status was posted (or it can't be confirmed), why, and what to do."""
    reason = md_text(NOT_ENFORCED_REASONS.get(reason_key, NOT_ENFORCED_REASONS["unconfirmed"]), in_table=False)
    if reason_key in UNCONFIRMED_REASONS:
        return (
            f"> ⚠️ **The `diffcost/cost-check` status could not be confirmed** ({reason}). "
            "If that check is required and this commit has no status yet, this PR can't merge until one is "
            "posted. A status shown from an earlier run on this commit may be out of date. Re-run this job."
        )
    fix = _NO_STATUS_FIX.get(reason_key, "Re-run this job.")
    return (
        f"> ⚠️ **No `diffcost/cost-check` status was posted** ({reason}). "
        f"If that check is required, this PR can't merge until this commit has one. {EARLIER_STATUS_NOTE} {fix}"
    )


def no_api_key_cause(fork_pr: bool, dependabot_pr: bool) -> str:
    """Why the `api_key` input may be empty. GitHub passes "" for a missing or misnamed
    secret, on fork PRs and on Dependabot PRs alike, so this can't tell those apart from
    keyless Free use: it names the likely cause."""
    if fork_pr:
        return "pull requests from forks don't get repository secrets"
    if dependabot_pr:
        return "Dependabot pull requests only get Dependabot secrets: add `DIFFCOST_API_KEY` there too"
    return "check that the `DIFFCOST_API_KEY` secret exists and the workflow passes it as `api_key`"


def no_api_key_line(cause: str) -> str:
    """The footer line of a run without an API key (replaces the Free-plan upsell)."""
    return (
        f"🔑 **No DiffCost API key**, so this run can't block merges. On a paid plan? The `api_key` input is "
        f"empty: {cause}. [Fix it]({NO_API_KEY_DOCS_URL}) · [Plans]({PRICING_URL})"
    )


# Where a failed Slack alert's webhook is configured
SLACK_SOURCE_YML = ".diffcost.yml webhook"
SLACK_SOURCE_DASHBOARD = "dashboard threshold webhook"


def slack_failure_line(source: str, reason: str) -> str:
    """One comment line: a Slack alert for this PR failed (`reason` never holds the webhook URL)."""
    return (
        f"⚠️ **Slack alert failed** ({source}): {md_text(shorten(reason, 200), in_table=False)}. "
        f"[What to check]({SLACK_DOCS_URL})"
    )


# --- Escaping -----------------------------------------------------------------


def _one_line(text: str) -> str:
    return re.sub(r"[\r\n\t]+", " ", str(text)).strip()


def md_code(text: str, in_table: bool = True) -> str:
    """Render untrusted text as an inline code span that cannot break out.

    The fence is one backtick longer than the longest backtick run in the value.
    """
    value = _one_line(text)
    if in_table:
        value = value.replace("|", "\\|")
    if not value:
        return "` `"
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    if longest == 0:
        return f"`{value}`"
    fence = "`" * (longest + 1)
    return f"{fence} {value} {fence}"


# Zero-width word joiner: stops GitHub turning @name into a mention and URLs into links
_WJ = "\u2060"


def _escape_plain(value: str, in_table: bool) -> str:
    value = re.sub(r"([\\`*_\[\]<>#!~])", r"\\\1", value)
    value = value.replace("@", f"@{_WJ}").replace("://", f":{_WJ}//").replace("www.", f"www{_WJ}.")
    if in_table:
        value = value.replace("|", "\\|")
    return value


def md_text(text: str, in_table: bool = True) -> str:
    """Escape untrusted text for plain markdown (no formatting, links, mentions or HTML)."""
    return _escape_plain(_one_line(text), in_table)


_CODE_SPAN = re.compile(r"`([^`]+)`")


def md_rich(text: str, in_table: bool = True) -> str:
    """Escape a DiffCost message that marks code with backticks, keeping those code spans.

    The messages interpolate untrusted values (refs, file names, error text), so each
    span goes through `md_code` and the text around it through the `md_text` escaping. A
    value containing backticks can only move where a code span starts or ends; it can't
    add formatting, links, mentions or HTML. Two spans that touch would merge their fences
    into one backtick run (one span left unclosed, its text rendered as markdown), so such
    a message, which DiffCost's own wording never produces, is escaped whole.
    """
    value = _one_line(text)
    parts: list[str] = []
    pos = 0
    for m in _CODE_SPAN.finditer(value):
        if parts and m.start() == pos:
            return md_text(value, in_table)
        parts.append(_escape_plain(value[pos:m.start()], in_table))
        parts.append(md_code(m.group(1), in_table))
        pos = m.end()
    parts.append(_escape_plain(value[pos:], in_table))
    return "".join(parts)


# Characters of a cost-unavailable reason shown in its table row: enough for a whole
# BigQuery error ("…, or perhaps it does not exist.") or a short list of files
REASON_LIMIT = 240


def shorten(text: str, limit: int) -> str:
    """Cut at a word boundary and add an ellipsis (never mid-word)."""
    text = _one_line(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut.rstrip(" ,.;:") + "…"


# --- Cost classification --------------------------------------------------------


def classify_cost(
    dry_run: dict | None,
    has_production: bool | None = None,
    has_pr: bool | None = None,
    baseline_unknown: bool = False,
    unknown_reason: str = "",
) -> dict:
    """Classify one item's dry-run result into a per-run cost record.

    status: changed | new | removed | unknown | error | not_costed | structural
    Unknown/error/not_costed items have no usable cost and are excluded from totals.
    """
    record = {
        "status": "structural",
        "cost_before": None,
        "cost_after": None,
        "delta": None,
        "pct": None,
        "reason": "",
        "blocking": False,
    }
    if unknown_reason:
        record.update(status="unknown", reason=unknown_reason)
        return record
    if baseline_unknown:
        # Before the no-dry-run return: without credentials it must still read as unknown,
        # not as "No structural changes".
        record.update(status="unknown", reason="no production baseline (no baseline manifest in this run)")
        return record
    if dry_run is None:
        return record

    before = dry_run.get("cost_before")
    after = dry_run.get("cost_after")
    if has_production is None:
        has_production = before is not None or dry_run.get("bytes_before") is not None
    if has_pr is None:
        has_pr = after is not None or dry_run.get("bytes_after") is not None

    deleted = bool(has_production and not has_pr)

    def no_estimate(status: str, reason: str) -> dict:
        # A query the PR deletes can only lower the cost: whatever kept BigQuery from
        # estimating its base version, it stays "removed", with the reason in place of a figure.
        if deleted:
            record.update(status="removed", delta=0.0, reason=reason)
        else:
            record.update(status=status, reason=reason)
        return record

    if dry_run.get("error"):
        return no_estimate("error", str(dry_run["error"]))
    if dry_run.get("unknown_reason"):
        # BigQuery's estimate can't be trusted (row-level security, or only a lower bound)
        return no_estimate("unknown", str(dry_run["unknown_reason"]))
    if dry_run.get("pr_depends_on_new_objects"):
        record.update(status=NOT_COSTED, cost_before=before, reason=NOT_COSTED_REASON)
        return record
    if dry_run.get("unmaterialized_upstream"):
        return no_estimate(
            "unknown",
            "the base version reads a table that does not exist" if deleted
            else "a referenced table does not exist yet (unmaterialized upstream)",
        )

    if has_production and has_pr and before is not None and after is not None:
        delta = after - before
        pct = (delta / before) * 100 if before > 0 else None
        record.update(status="changed", cost_before=before, cost_after=after, delta=delta, pct=pct)
    elif not has_production and has_pr and after is not None:
        record.update(status="new", cost_after=after, delta=after)
    elif deleted and before is not None:
        record.update(status="removed", cost_before=before, delta=-before)
    else:
        return no_estimate("unknown", "dry run returned no byte estimate")
    return record


def _cost_record(r: dict) -> dict:
    return r.get("cost") or classify_cost(r.get("dry_run"))


def unestimated_removal(rec: dict) -> bool:
    """A query the PR deletes whose base version BigQuery couldn't estimate: shown with the
    reason, left out of the totals and the worst query, never blocking."""
    return rec.get("status") == "removed" and rec.get("cost_before") is None


# --- Comment bodies ---------------------------------------------------------------


def format_comment(
    results: list[dict],
    total_before: float,
    total_after: float,
    total_delta: float,
    has_bq_creds: bool,
    blocked: bool,
    block_reasons: list[str],
    plan: str,
    trial_days_remaining: int | None = None,
    bq_pricing_model: str = "on_demand",
    runs_per_month: int = 30,
    items_unknown: int = 0,
    worst_query: dict | None = None,
    not_enforced_reason: str = "",
    enforcement_unconfirmed: bool = False,
    items_not_costed: int = 0,
    policy_warnings: list[str] | None = None,
    bq_error: str = "",
    trial_ended: bool = False,
    status_note_line: str = "",
    merge_blocking_on: bool = True,
    service_problem: bool = False,
    no_key_cause: str = "",
    slack_failures: list[tuple[str, str]] | None = None,
    subscription_inactive: bool = False,
    threshold_source: str = "",
    query_parameters: list[tuple[str, str]] | None = None,
) -> str:
    """Generate the markdown PR comment body.

    `blocked` means the cost policy is exceeded. Unless `not_enforced_reason` is empty
    (the backend posted the failing commit status), the comment never says
    "Merge blocked"; with `enforcement_unconfirmed` it says the enforcement could not be
    confirmed. `status_note_line` (see `status_note`) says no commit status was posted;
    `merge_blocking_on` is whether the paid footer may say merge blocking is enabled;
    `service_problem` (the dashboard config could not be fetched) hides the Free-plan
    upsell, which would be misleading for a paying org. `no_key_cause` (set when the
    `api_key` input is empty) replaces that upsell with the "No DiffCost API key" line.
    `slack_failures` are (webhook source, reason) pairs for Slack alerts that failed.
    `subscription_inactive` (from /v1/config): a Stripe subscription that no longer pays,
    so the Free plan line says to update the payment method. `threshold_source` (the
    backend's, or "yml" for a local check of `.diffcost.yml`): a blocked verdict names it,
    and a paid footer says when no threshold applies ("none"). `query_parameters`: the
    (name, value) pairs from `.diffcost.yml` that plain SQL files were dry-run with, for a
    note saying so (None when no plain SQL file was).

    The verdict, the status note, cost-policy warnings and the hints on unknown costs come
    right after the title, before the tables, so truncating a long report never drops them.
    """
    lines = [COMMENT_MARKER, ""]

    if has_bq_creds:
        report = _format_cost_report(
            results, total_before, total_after, total_delta, bq_pricing_model, runs_per_month, worst_query
        )
    else:
        report = _format_structural_report(results, bq_error)
    # report[0] is the title line, report[1] a blank line
    lines.extend(report[:2])

    blocks: list[list[str]] = []
    if blocked:
        # The status note right below gives the same reason, so the heading doesn't repeat it
        reason = "" if status_note_line else f" ({md_text(not_enforced_reason, in_table=False)})"
        if not_enforced_reason and enforcement_unconfirmed:
            heading = f"⚠️ **Cost policy exceeded — enforcement unconfirmed**{reason}:"
        elif not_enforced_reason:
            heading = f"⚠️ **Cost policy exceeded — not enforced**{reason}:"
        else:
            heading = "🚫 **Merge blocked** — this PR exceeds a DiffCost cost policy:"
        verdict = [heading] + [f"- {md_text(r, in_table=False)}" for r in block_reasons]
        source = THRESHOLD_SOURCES.get(threshold_source)
        if source:
            verdict += ["", f"The policy that applied: {source}."]
        blocks.append(verdict)
    if status_note_line:
        blocks.append([status_note_line])
    for source, reason in slack_failures or []:
        blocks.append([slack_failure_line(source, reason)])
    for warning in policy_warnings or []:
        blocks.append([f"> ⚠️ **Cost policy:** {md_rich(warning, in_table=False)}"])
    for hint in (_ignore_hint(results), _exempt_hint(results), _parameters_hint(results)):
        if hint:
            blocks.append([hint])
    for block in blocks:
        lines.extend(block)
        lines.append("")

    lines.extend(report[2:])

    if has_bq_creds and items_unknown:
        noun = "query" if items_unknown == 1 else "queries"
        lines.append("")
        lines.append(
            f"ℹ️ {items_unknown} {noun} could not be costed and {'is' if items_unknown == 1 else 'are'} "
            "excluded from the totals above."
        )
    if has_bq_creds and query_parameters:
        lines.append("")
        lines.append(query_parameters_line(query_parameters))
    if has_bq_creds and items_not_costed:
        noun = "query reads" if items_not_costed == 1 else "queries read"
        lines.append("")
        lines.append(
            f"ℹ️ {items_not_costed} {noun} tables or columns this PR creates, so {'it' if items_not_costed == 1 else 'they'} "
            "can be costed only after merge. No structural cost risk was found, so the cost policy does not block on "
            f"{'it' if items_not_costed == 1 else 'them'}."
        )

    # High Scan Warning when a partition filter was removed (named when table metadata confirms it)
    scan_warning = _scan_warning(results)
    if scan_warning:
        lines.append("")
        lines.append(scan_warning)

    # Structural findings next to the cost report (the structural-only report already lists them)
    structural_lines = _format_structural_findings(results) if has_bq_creds else []
    if structural_lines:
        lines.append("")
        lines.extend(structural_lines)

    # CTAs
    cta_lines = _format_ctas(
        has_bq_creds, plan, trial_days_remaining, trial_ended, merge_blocking_on, service_problem, no_key_cause,
        subscription_inactive=subscription_inactive, threshold_source=threshold_source,
    )
    if cta_lines:
        lines.append("")
        lines.extend(cta_lines)

    # Footer
    lines.append("")
    lines.append("---")
    lines.append("*Powered by [DiffCost](https://diffcost.dev) · Costs are per query run, from BigQuery dry runs*")

    return "\n".join(lines)  # cut to GitHub's comment limit when posted (the job summary keeps more)


def format_no_changes_comment(note: str = "", no_key_cause: str = "") -> str:
    body = (
        f"{COMMENT_MARKER}\n\n💰 **DiffCost** — No SQL changes in the latest push. "
        "Earlier cost results on this PR no longer apply."
    )
    return body + (f"\n\n{note}" if note else "") + _no_key_suffix(no_key_cause)


def format_clean_comment(note: str = "", no_key_cause: str = "") -> str:
    body = f"{COMMENT_MARKER}\n\n💰 **DiffCost** — No SQL cost changes detected in this PR."
    return body + (f"\n\n{note}" if note else "") + _no_key_suffix(no_key_cause)


def _no_key_suffix(no_key_cause: str) -> str:
    return f"\n\n{no_api_key_line(no_key_cause)}" if no_key_cause else ""


def format_failure_comment(message: str, no_status: bool = False) -> str:
    """`no_status`: an API key is set, so a required `diffcost/cost-check` may now wait."""
    body = (
        f"{COMMENT_MARKER}\n\n⚠️ **DiffCost could not analyze this PR.**\n\n"
        f"{md_rich(message, in_table=False)}\n\n"
        f"No cost result was produced, so no cost policy was evaluated. [Troubleshooting]({TROUBLESHOOTING_URL})"
    )
    if no_status:
        body += (
            "\n\n> ⚠️ **No `diffcost/cost-check` status was posted.** If that check is required, this PR can't "
            f"merge until DiffCost runs successfully. {EARLIER_STATUS_NOTE}"
        )
    return body


def _ignored_paths_code(entries: list[str]) -> str:
    return md_code("ignored_paths: [" + ", ".join(json.dumps(e) for e in entries) + "]", in_table=False)


def _collect(results: list[dict], key: str) -> list[str]:
    """The distinct values of `key` (a string or a list of strings) across results, in order."""
    found: list[str] = []
    for r in results:
        value = r.get(key)
        for v in ([value] if isinstance(value, str) else value or []):
            if v and str(v) not in found:
                found.append(str(v))
    return found


def _ignore_hint(results: list[dict]) -> str:
    """How to opt SQL that DiffCost did not analyze out of the cost policy."""
    globs = _collect(results, "ignore_globs")
    if not globs:
        return ""
    return (
        "> ℹ️ **SQL outside the analyzed dbt project** counts as unknown cost. If it doesn't run on "
        "BigQuery, add it to `ignored_paths` in `.diffcost.yml` on the base branch: " + _ignored_paths_code(globs)
    )


def _exempt_hint(results: list[dict]) -> str:
    """How to stop costing a file whose queries BigQuery can't estimate before they run
    (row-level security, a lower bound, an estimate of unknown accuracy)."""
    entries = _collect(results, "exempt_entry")
    if not entries:
        return ""
    return (
        "> ℹ️ **BigQuery can't estimate some queries before they run** (row-level security, external tables, "
        "scripts it stops estimating, or an estimate of unknown accuracy), so they count as unknown cost. To stop "
        "costing a file (DiffCost won't price or flag it again), add it to `ignored_paths` in `.diffcost.yml` on "
        "the base branch, in a separate PR: " + _ignored_paths_code(entries)
    )


QUERY_PARAMETERS_URL = "https://diffcost.dev/docs/configuration#query-parameters"
_DATE_PARAMETER_TYPES = ("DATE", "DATETIME", "TIMESTAMP")


def _parameters_hint(results: list[dict]) -> str:
    """How to give values to the query parameters of plain SQL files that have none (the
    files count as unknown cost until then), and, for parameters `query_parameters` can't set
    (other types such as arrays and structs, positional `?`), how to stop costing the file."""
    params: dict[str, tuple[str, str]] = {}
    for r in results:
        for name, ptype in r.get("unbound_parameters") or []:
            kept = params.get(name.lower())
            if kept is None or (ptype and not kept[1]):
                params[name.lower()] = (name, ptype)
    exempt = _collect(results, "parameter_exempt_entry")
    if not params and not exempt:
        return ""
    found = list(params.values())
    named = [(n, t) for n, t in found if n != "?"]
    settable = [(n, t) for n, t in found if settable_parameter(n, t)]
    names = ", ".join(md_code(f"@{n}" + (f" {t}" if t else ""), in_table=False) for n, t in named[:5])
    if len(named) > 5:
        names += f" and {len(named) - 5} more"
    parts = [
        "> ℹ️ **Some queries use query parameters that have no value**"
        + (f" ({names})" if names else "") + ", so they count as unknown cost."
    ]
    if settable:
        entries = [
            f"{n}: {{type: {t}, " + ("days_ago: 1}" if t in _DATE_PARAMETER_TYPES else "value: …}")
            for n, t in settable[:5] if t
        ]
        example = (
            f", for example: {md_code('query_parameters: {' + ', '.join(entries) + '}', in_table=False)}"
            if entries else ""
        )
        which = "each scalar parameter" if exempt else "each one"
        parts.append(
            f"Give {which} a value under `query_parameters` in `.diffcost.yml` on the base branch, in a separate "
            f"PR{example}."
        )
    if exempt:
        parts.append(
            "Parameters of other types, such as arrays and structs, and positional parameters (`?`) can't be given "
            "a value: to stop costing a file that uses them (DiffCost won't price or flag it again), add it to "
            "`ignored_paths` in `.diffcost.yml` on the base branch, in a separate PR: "
            + _ignored_paths_code(exempt) + "."
        )
    parts.append(f"See [query parameters]({QUERY_PARAMETERS_URL}).")
    return " ".join(parts)


def query_parameters_line(params: list[tuple[str, str]], limit: int = 10) -> str:
    """The note naming the `.diffcost.yml` parameter values plain SQL files were priced with."""
    shown = [md_code(f"@{name} = {value}", in_table=False) for name, value in params[:limit]]
    more = len(params) - limit
    listed = ", ".join(shown) + (f" and {more} more" if more > 0 else "")
    return f"ℹ️ Plain SQL files were priced with {listed} (`query_parameters` in `.diffcost.yml`)."


_DETAILS_CLOSE = "\n\n</details>"


def _finish_cut(cut: str, note: str) -> str:
    """End a truncated body at a line boundary with `note`, outside any <details> the cut left
    open (inside a collapsed section, readers wouldn't see the note)."""
    if "\n" in cut:
        cut = cut[: cut.rfind("\n")]
    if cut.count("<details>") > cut.count("</details>"):
        cut += _DETAILS_CLOSE
    return cut + "\n\n…\n\n" + note


def truncate_summary(body: str, limit_bytes: int = MAX_SUMMARY_BYTES) -> str:
    """Keep a job summary under GitHub's limit, which counts UTF-8 bytes."""
    data = body.encode("utf-8")
    if len(data) <= limit_bytes:
        return body
    reserve = len(("\n\n…\n\n" + SUMMARY_TRUNCATED_NOTE + _DETAILS_CLOSE).encode("utf-8"))
    # "ignore" drops a multi-byte character the byte cut splits.
    return _finish_cut(data[: limit_bytes - reserve].decode("utf-8", "ignore"), SUMMARY_TRUNCATED_NOTE)


def truncate_body(body: str) -> str:
    """Keep a comment body under GitHub's size limit, cutting at a line boundary."""
    if len(body) <= MAX_COMMENT_CHARS:
        return body
    reserve = len("\n\n…\n\n" + COMMENT_TRUNCATED_NOTE + _DETAILS_CLOSE)
    return _finish_cut(body[: MAX_COMMENT_CHARS - reserve], COMMENT_TRUNCATED_NOTE)


HIGH_SCAN_WARNING = "> ⚠️ **High Scan Warning:** Query scans full table without partition filter."
UPPER_BOUND_NOTE = (
    "ℹ️ **≤** marks an upper bound: BigQuery prunes clustered tables only when the query runs, so the run can "
    "scan less than the dry run estimates. A Cost Change isn't a bound: when the Current figure is marked, the "
    "real increase can be larger."
)


def _scan_warning(results: list[dict]) -> str:
    """The High Scan Warning: names the queries and tables when table metadata confirms that
    the removed filter was on the tables' partition column; the generic line when only the
    column name suggests it; "" when no partition filter was removed, the dry run shows no
    extra scan, or the PR keeps another filter on that column (more partitions, not a full
    scan)."""
    confirmed: list[str] = []
    queries: list[str] = []
    guessed = False
    for r in results:
        dr = r.get("dry_run") or {}
        bb, ba = dr.get("bytes_before"), dr.get("bytes_after")
        if bb is not None and ba is not None and ba <= bb:
            continue  # the dry run shows no extra scan: don't warn about a full table scan
        for f in r.get("structural_findings", []):
            if f.partition_filter_kept:
                continue
            if f.partition_tables:
                confirmed.extend(t for t in f.partition_tables if t not in confirmed)
                if r["name"] not in queries:
                    queries.append(r["name"])
                continue
            desc = f.description.lower()
            detail = f.detail.lower()
            if f.risk == "high" and ("partition" in desc or "partition" in detail or "full table scan" in detail):
                guessed = True
    if confirmed:
        one = len(queries) == 1
        return (
            f"> ⚠️ **High Scan Warning:** {_first_three([md_code(q, in_table=False) for q in queries])} no longer "
            f"{'filters' if one else 'filter'} on the partition column of "
            f"{_first_three([md_text(t, in_table=False) for t in confirmed])}, so "
            f"{'it reads' if one else 'they read'} every partition."
        )
    return HIGH_SCAN_WARNING if guessed else ""


def _first_three(names: list[str]) -> str:
    return ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def _scan_delta(bb, ba, up_before: bool = False, up_after: bool = False) -> str:
    """The Scan Delta cell; the figure of a new or removed query keeps its "≤" mark."""
    if bb is not None and ba is not None:
        b_delta = ba - bb
        return ("+" if b_delta > 0 else "") + _format_bytes(b_delta)
    if ba is not None:
        return f"+{_upper(_format_bytes(ba), up_after)}"
    if bb is not None:
        return f"-{_upper(_format_bytes(bb), up_before)}"
    return "—"


def _monthly(amount: float, runs_per_month: int) -> str:
    monthly = _format_cost(abs(amount) * runs_per_month)
    approx = "" if monthly.startswith("<") else "≈"
    return f"{approx}{monthly}/mo at {runs_per_month} runs/mo"


def _format_cost_report(
    results: list[dict],
    total_before: float,
    total_after: float,
    total_delta: float,
    bq_pricing_model: str,
    runs_per_month: int,
    worst_query: dict | None,
) -> list[str]:
    """Format the cost comparison table (per-run costs, scan delta)."""
    is_editions = bq_pricing_model == "editions"
    lines = ["💰 **DiffCost** — BigQuery Cost Report" + ("" if is_editions else " (cost per run)"), ""]

    if is_editions:
        lines.append("| Query / Model | Current scan | This PR scan | Scan Delta | Change |")
    else:
        lines.append("| Query / Model | Current | This PR | Scan Delta | Cost Change |")
    lines.append("|:---|---:|---:|---:|:---|")

    known_rows = 0
    total_bytes_before = 0
    total_bytes_after = 0
    has_bytes = False
    any_upper_before = any_upper_after = False
    upper_noted = False  # a row outside the totals shows an upper-bound figure
    unestimated_removals = 0  # deleted queries BigQuery couldn't estimate (not in the totals)
    uncosted_rows = 0  # unknown and Not costed rows

    for r in results:
        rec = _cost_record(r)
        dr = r.get("dry_run")
        if not dr and rec["status"] not in UNKNOWN_STATUSES:
            continue
        dr = dr or {}  # unknown without a dry run, e.g. "the PR manifest was not recompiled"
        name = md_code(r["name"])

        if rec["status"] in UNKNOWN_STATUSES:
            reason = md_text(shorten(rec["reason"], REASON_LIMIT))
            label = ("⚠️ " if rec["status"] == "error" else "⚠️ Cost unavailable: ") + reason
            lines.append(f"| {name} | — | — | — | {label} |")
            uncosted_rows += 1
            continue
        if rec["status"] == NOT_COSTED:
            uncosted_rows += 1
            label = "⏳ Not costed: " + md_text(rec["reason"])
            if rec.get("blocking"):
                label += f" ({rec.get('blocking_note') or 'structural cost risk'}, so it counts as unknown)"
            before_str = "—"
            if rec["cost_before"] is not None and not is_editions:
                # The base figure is shown, though not summed: an upper bound keeps its mark and note
                up = bool(dr.get("upper_bound_before"))
                before_str = _upper(_format_cost(rec["cost_before"]), up)
                upper_noted |= up
            lines.append(f"| {name} | {before_str} | — | — | {label} |")
            continue

        if unestimated_removal(rec):
            unestimated_removals += 1
            suffix = " [Editions - Slot Based]" if is_editions else ""
            lines.append(f"| {name} | — | — | — | **Removed**{_no_figure(rec)}{suffix} |")
            continue

        bb = dr.get("bytes_before")
        ba = dr.get("bytes_after")
        if bb is not None or ba is not None:
            has_bytes = True
            total_bytes_before += bb or 0
            total_bytes_after += ba or 0
        known_rows += 1
        # BigQuery's estimate is an upper bound for that version (clustered tables)
        up_before = bool(dr.get("upper_bound_before")) and bb is not None
        up_after = bool(dr.get("upper_bound_after")) and ba is not None
        any_upper_before |= up_before
        any_upper_after |= up_after
        scan_delta_str = _scan_delta(bb, ba, up_before, up_after)

        if is_editions:
            before_str = _upper(_format_bytes(bb), up_before)
            after_str = _upper(_format_bytes(ba), up_after)
            if rec["status"] == "new":
                change_str = "**New** [Editions - Slot Based]"
            elif rec["status"] == "removed":
                change_str = "**Removed** [Editions - Slot Based]"
            else:
                mult = (ba / bb) if bb else 1.0
                change_str = (
                    "[Editions - Slot Based]" if abs(mult - 1.0) < 0.05 else f"{mult:.1f}x scan [Editions - Slot Based]"
                )
        else:
            before = rec["cost_before"]
            after = rec["cost_after"]
            before_str = _upper(_format_cost(before), up_before) if before is not None else "—"
            after_str = _upper(_format_cost(after), up_after) if after is not None else "—"
            if rec["status"] == "new":
                change_str = f"**New** (+{_upper(_format_cost(after), up_after)})"
            elif rec["status"] == "removed":
                change_str = f"**Removed** (was {_upper(_format_cost(before), up_before)})"
            else:
                change_str = _format_change(rec["delta"], before)

        lines.append(f"| {name} | {before_str} | {after_str} | {scan_delta_str} | {change_str} |")

    # Total row (only if multiple costed results)
    if known_rows > 1:
        total_scan_delta_str = _scan_delta(total_bytes_before, total_bytes_after) if has_bytes else "—"
        if is_editions:
            lines.append(
                f"| **TOTAL** | **{_upper(_format_bytes(total_bytes_before), any_upper_before)}** | "
                f"**{_upper(_format_bytes(total_bytes_after), any_upper_after)}** | **{total_scan_delta_str}** | "
                "**[Editions - Slot Based]** |"
            )
        else:
            lines.append(
                f"| **TOTAL** | **{_upper(_format_cost(total_before), any_upper_before)}** | "
                f"**{_upper(_format_cost(total_after), any_upper_after)}** | "
                f"**{total_scan_delta_str}** | **{_format_change(total_delta, total_before)}** |"
            )

    # Summary line
    lines.append("")
    worst_delta = (worst_query or {}).get("delta") or 0
    if known_rows == 0:
        if uncosted_rows or not unestimated_removals:
            lines.append("⚠️ No query in this PR could be costed. See the rows above.")
    elif is_editions:
        if total_bytes_after > total_bytes_before:
            lines.append(
                f"⚠️ This PR increases query scan volume by **{_format_bytes(total_bytes_after - total_bytes_before)}** "
                "per run (Editions slot-based pricing: exact cost depends on slot reservation)."
            )
        elif total_bytes_after < total_bytes_before:
            lines.append(
                f"✅ This PR **reduces** query scan volume by **{_format_bytes(total_bytes_before - total_bytes_after)}** "
                "per run (Editions slot-based pricing)."
            )
        else:
            lines.append("✅ No scan volume change detected (Editions slot-based pricing).")
    else:
        if total_delta > 0.005:
            lines.append(
                f"⚠️ This PR increases query costs by **{_format_cost(total_delta)}/run** "
                f"({_format_pct(total_delta, total_before)}, {_monthly(total_delta, runs_per_month)})."
            )
        elif worst_delta > 0.005:
            # Savings elsewhere offset it, but thresholds apply to the worst single query
            net = f"{'+' if total_delta >= 0 else '-'}{_format_cost(abs(total_delta))}/run"
            lines.append(
                f"⚠️ {md_code(worst_query['name'], in_table=False)} costs **+{_format_cost(worst_delta)}/run** more; "
                f"savings in other queries offset it (net {net})."
            )
        elif total_delta < -0.005:
            lines.append(
                f"✅ This PR **reduces** query costs by **{_format_cost(abs(total_delta))}/run** "
                f"({_format_pct(total_delta, total_before)}, {_monthly(total_delta, runs_per_month)})."
            )
        else:
            lines.append("✅ No significant cost change detected.")

        if worst_delta > 0.005 and known_rows > 1 and total_delta > 0.005:
            pct = worst_query.get("pct")
            pct_str = f" (↑{pct:.0f}%)" if pct is not None else ""
            lines.append(
                f"Largest single-query increase: {md_code(worst_query['name'], in_table=False)} "
                f"**+{_format_cost(worst_query['delta'])}/run**{pct_str}."
            )

    if unestimated_removals:
        one = unestimated_removals == 1
        lines.append(
            f"ℹ️ {unestimated_removals} removed {'query' if one else 'queries'} BigQuery couldn't estimate "
            f"{'is' if one else 'are'} left out of the totals; removing a query never blocks."
        )

    if has_bytes and (total_bytes_after != total_bytes_before):
        lines.append(
            f"📊 **Data Scanned per run:** {_upper(_format_bytes(total_bytes_before), any_upper_before)} → "
            f"{_upper(_format_bytes(total_bytes_after), any_upper_after)} "
            f"({_scan_delta(total_bytes_before, total_bytes_after)})"
        )

    if any_upper_before or any_upper_after or upper_noted:
        lines.append(UPPER_BOUND_NOTE)

    return lines


def _no_figure(rec: dict) -> str:
    """" (not estimated: {reason}; doesn't block)" for a removed query BigQuery couldn't
    estimate (classify_cost always gives it a reason)."""
    return f" (not estimated: {md_text(shorten(rec['reason'], REASON_LIMIT))}; doesn't block)"


def _upper(text: str, is_upper_bound: bool) -> str:
    """"≤$9.66" for an upper-bound figure ("<$0.01" and "—" stay as they are)."""
    if not is_upper_bound or text == "—" or text.startswith("<"):
        return text
    return f"≤{text}"


_RISK_ICONS = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def _format_finding(f, in_table: bool = True) -> str:
    text = md_text(f.description, in_table=in_table)
    if f.subject:
        text += f": {md_code(f.subject, in_table=in_table)}"
    return text


def _format_structural_report(results: list[dict], bq_error: str) -> list[str]:
    """Format the structural-only report (no usable BigQuery client)."""
    lines = ["💰 **DiffCost** — SQL Change Analysis", ""]
    if bq_error:
        lines.append(
            f"> ⚠️ BigQuery credentials were provided but could not be used ({md_text(bq_error, in_table=False)}). "
            f"Showing structural analysis only. See the [setup guide]({SETUP_DOCS_URL})."
        )
    else:
        lines.append(
            "> ℹ️ No BigQuery credentials configured. Showing structural analysis only. "
            f"[Add BigQuery access]({SETUP_DOCS_URL}) for per-run dollar estimates."
        )
    lines.append("")

    has_findings = any(r.get("structural_findings") for r in results)
    flagged = [r for r in results if _explicit_unknown(r)]
    if not has_findings and not flagged:
        lines.append("✅ No structural cost risks detected in changed SQL files.")
        return lines

    lines.append("| File | Changes Detected | Risk |")
    lines.append("|:---|:---|:---:|")

    for r in results:
        findings = r.get("structural_findings", [])
        if _explicit_unknown(r):
            reason = md_text(shorten(r["cost"]["reason"], 160))
            lines.append(f"| {md_code(r['name'])} | ⚠️ Cost unavailable: {reason} | ⚪ |")
            if not findings:
                continue
        elif not findings:
            lines.append(f"| {md_code(r['name'])} | No structural changes | 🟢 |")
            continue
        for i, f in enumerate(findings):
            name_col = md_code(r["name"]) if i == 0 else ""
            risk_icon = _RISK_ICONS.get(f.risk, "⚪")
            lines.append(f"| {name_col} | {_format_finding(f)} | {risk_icon} |")
            if f.detail:
                lines.append(f"| | → {md_text(shorten(f.detail, 160))} | |")

    return lines


def _explicit_unknown(r: dict) -> bool:
    """An item whose cost is unknown for a stated reason (shown even without a dry run)."""
    rec = r.get("cost") or {}
    return rec.get("status") == "unknown" and bool(rec.get("reason"))


def _format_structural_findings(results: list[dict]) -> list[str]:
    """Format structural findings as a details section (shown alongside the cost report)."""
    significant = []
    for r in results:
        for f in r.get("structural_findings", []):
            if f.risk in ("high", "medium"):
                significant.append((r["name"], f))

    if not significant:
        return []

    lines = ["<details>", "<summary>🔍 Root cause analysis</summary>", ""]
    for name, f in significant:
        risk_icon = _RISK_ICONS.get(f.risk, "⚪")
        lines.append(f"- {risk_icon} **{md_code(name, in_table=False)}**: {_format_finding(f, in_table=False)}")
        if f.detail:
            lines.append(f"  - {md_text(f.detail, in_table=False)}")
    lines.append("")
    lines.append("</details>")

    return lines


def _format_ctas(
    has_bq_creds: bool,
    plan: str,
    trial_days_remaining: int | None,
    trial_ended: bool,
    merge_blocking_on: bool,
    service_problem: bool,
    no_key_cause: str,
    subscription_inactive: bool,
    threshold_source: str,
) -> list[str]:
    """Format call-to-action links based on plan and credential status."""
    lines = []

    if not has_bq_creds:
        lines.append(
            f"💡 **Get per-run dollar estimates:** [Add BigQuery dry-run access]({SETUP_DOCS_URL}) (no rows are read)"
        )

    if service_problem:
        pass  # the plan is unknown (dashboard config unavailable): no upsell
    elif no_key_cause:
        # No API key: a paying org's missing secret looks exactly like keyless Free use
        lines.append(no_api_key_line(no_key_cause))
    elif plan == "free" and trial_ended:
        # The line shown once the Team trial has ended
        lines.append(
            f"⌛ **Your Team trial has ended.** Merge blocking, Slack alerts and cost trends are off; "
            f"[upgrade]({BILLING_URL}) to turn them back on."
        )
    elif plan == "free" and subscription_inactive:
        # A Stripe subscription that no longer pays (unpaid or paused): the fix is the
        # payment method, never a second subscription (checkout is refused meanwhile)
        lines.append(
            "⚠️ **Your Stripe subscription isn't active.** Merge blocking, Slack alerts and cost trends are off; "
            f"[update the payment method]({BILLING_URL}) to turn them back on."
        )
    elif plan == "free":
        # With an API key, a Free run is a former payer (an admin's sign-in starts the
        # trial, and only admins create keys): the trial can't restart, so link Billing
        lines.append(
            f"📊 **DiffCost Free:** merge blocking is off. [Upgrade]({BILLING_URL}) to Starter or Team "
            "to block expensive PRs."
        )
    elif trial_days_remaining is not None and trial_days_remaining > 0:
        day_str = f"{trial_days_remaining} day{'s' if trial_days_remaining != 1 else ''} remaining"
        no_threshold = (
            f" Merge blocking is on, but no threshold applies to this repository ([set one]({THRESHOLDS_DOCS_URL}))."
            if merge_blocking_on and threshold_source == "none" else ""
        )
        lines.append(
            f"⏳ **DiffCost Team trial:** {day_str}.{no_threshold} [Subscribe]({BILLING_URL}) to keep merge blocking "
            "after the trial (Starter $49/mo, Team $249/mo)."
        )
    elif plan in ("starter", "team", "enterprise"):
        if merge_blocking_on and threshold_source == "none":
            enabled = (
                " — Merge blocking enabled, but no threshold applies to this repository "
                f"([set one]({THRESHOLDS_DOCS_URL}))."
            )
        elif merge_blocking_on:
            enabled = " — Merge blocking enabled."
        else:
            enabled = ""
        lines.append(f"🛡️ **DiffCost {plan.capitalize()}**{enabled} [Manage subscription]({BILLING_URL})")

    return lines


def _format_cost(amount: float | None) -> str:
    """Format a dollar amount."""
    if amount is None:
        return "—"
    if amount == 0:
        return "$0.00"
    if abs(amount) < 0.001:
        return f"{'-' if amount < 0 else ''}<$0.001"
    if abs(amount) < 1.0:
        return f"{'-' if amount < 0 else ''}${abs(amount):.3f}"
    return f"{'-' if amount < 0 else ''}${abs(amount):,.2f}"


def format_usd_limit(amount: float) -> str:
    """A per-run USD figure in a cost-policy reason: at least 2 and up to 4 decimals (the
    backend's `fmt_usd` rule), so a $0.0005 threshold never prints as $0.00."""
    value = float(amount)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if 0 < value < 0.0001:
        return f"{sign}<$0.0001"
    text = f"{value:.4f}".rstrip("0")
    if len(text.split(".")[1]) < 2:
        text = f"{value:.2f}"
    return f"{sign}${text}"


def format_usd_change(amount: float, limit: float | None = None) -> str:
    """A per-run cost change in a cost-policy reason, as the report table shows it
    (`_format_cost`: "$14.82", "$0.010", "<$0.001"), unless that rounding would make it
    read as not above `limit` ($1.004 over a $1.00 limit, $0.0052 over $0.005); then
    `format_usd_limit`'s 2-4 decimals, and if even that reads as the limit ($5.00004 over
    $5.00, $0.00053 over $0.0005), the fewest of 5 to 8 decimals that read above it.
    Same rule as the backend's `usd_change`."""
    value = float(amount)
    if limit is None:
        return _format_cost(value)
    size = abs(value)
    if size >= 0.001:
        reads_above = round(size, 2 if size >= 1 else 3) > round(float(limit), 4)
    else:  # "<$0.001" is only clearly above a $0 limit
        reads_above = size > 0 and float(limit) <= 0
    if reads_above:
        return _format_cost(value)
    text = format_usd_limit(value)
    if size >= 0.0001 and float(text.lstrip("-$")) > float(limit):
        return text
    sign = "-" if value < 0 else ""
    for places in (5, 6, 7, 8):
        digits = f"{size:.{places}f}"
        if float(digits) > float(limit):
            return f"{sign}${digits}"
    return f"{sign}${size:.8f}"


def format_pct_change(pct: float, limit: float | None = None) -> str:
    """A percentage increase in a cost-policy reason, without the sign and the `%`: whole
    percent ("200"), unless that rounding would make it read as not above `limit` (200.4%
    over a 200% limit would print "+200% exceeds 200%"); then the fewest decimals, up to
    8, that read above it ("200.4"). Same rule as the backend's `pct_change`, so the
    commit status, Slack and the PR comment show the same figure."""
    value = float(pct)
    for places in range(9):
        text = f"{value:.{places}f}"
        if limit is None or float(text) > float(limit):
            return text
    return f"{value:.8f}"


def _format_pct(delta: float, base: float) -> str:
    """Format a percentage change."""
    if base <= 0:
        return "new"
    pct = (delta / base) * 100
    if pct > 10000:
        return f">{pct / 100:.0f}x"
    return f"{pct:+.0f}%"


def _format_change(delta: float | None, base: float | None) -> str:
    """Format a cost change with percentage."""
    if delta is None or abs(delta) < 0.0005:
        return "—"
    sign = "+" if delta > 0 else ""
    if base and base > 0:
        arrow = "↑" if delta > 0 else "↓"
        return f"{sign}{_format_cost(delta)} ({arrow}{abs(delta / base) * 100:.0f}%)"
    return f"{sign}{_format_cost(delta)}"


# --- Posting -------------------------------------------------------------------------

_PERMISSION_HINT = (
    "Ensure your workflow has `permissions: pull-requests: write`. Pull requests from forks get a "
    "read-only token, so DiffCost can't comment on them; the report is in the job summary instead."
)


def _api_base() -> str:
    return os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")


def _find_existing_comments(url: str, headers: dict) -> tuple[list[dict], list[dict], int]:
    """DiffCost comments on the PR (by the hidden marker), following pagination.

    Returns (bot comments, user comments, 200 or the status of the list call that failed). Bot comments
    come first by `github-actions[bot]` (the default token), then other bots (a GitHub App
    token). User comments are candidates only when the token is that user's personal
    access token (see `_token_login`), so a human who quotes the marker is never edited.
    """
    next_url: str | None = url
    params: dict | None = {"per_page": 100}
    default_bot: list[dict] = []
    other_bots: list[dict] = []
    users: list[dict] = []
    for _ in range(_MAX_COMMENT_PAGES):
        resp = requests.get(next_url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return [], [], resp.status_code
        comments = resp.json() or []
        for c in comments:
            if COMMENT_MARKER not in (c.get("body") or ""):
                continue
            user = c.get("user") or {}
            if user.get("type") == "Bot":
                (default_bot if user.get("login") == _DEFAULT_BOT_LOGIN else other_bots).append(c)
            elif user.get("login"):
                users.append(c)
        links = resp.links if isinstance(getattr(resp, "links", None), dict) else {}
        next_url = (links.get("next") or {}).get("url")
        params = None  # the "next" link already carries the query string
        if not next_url or not comments:
            break
    return default_bot + other_bots, users, 200


def _token_login(headers: dict) -> str:
    """The login of a personal access token's user, or "" for an installation token (the
    default workflow token or a GitHub App's, for which GitHub answers 403) or on any error."""
    try:
        resp = requests.get(f"{_api_base()}/user", headers=headers, timeout=10)
        data = resp.json() if resp.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return ""
    login = data.get("login") if isinstance(data, dict) else None
    return login if isinstance(login, str) else ""


def post_or_update_comment(
    repo: str,
    pr_number: int,
    token: str,
    body: str,
    create: bool = True,
    body_if_new: str | None = None,
) -> str | None:
    """Post or update the DiffCost PR comment. Returns the comment HTML URL on success, or None.

    With create=False, only an existing DiffCost comment is updated. `body_if_new` replaces
    `body` when no DiffCost comment exists yet (wording that assumes an earlier comment).
    """
    if not token or not repo or not pr_number:
        return None

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    base_url = f"{_api_base()}/repos/{repo}/issues/{pr_number}/comments"
    body = truncate_body(body)

    try:
        bots, users, status = _find_existing_comments(base_url, headers)
        if status != 200:
            print(f"::warning::Failed to list PR comments (HTTP {status}). Cannot post comment.")
            if status in (403, 404):
                print(f"::warning::{_PERMISSION_HINT}")
            return None

        def candidates():
            yield from bots
            login = _token_login(headers) if users else ""
            if login:
                yield from (c for c in users if (c["user"].get("login") or "").lower() == login.lower())

        existing = None
        resp = None
        found = False
        for c in candidates():
            found = True
            resp = requests.patch(c["url"], headers=headers, json={"body": body}, timeout=10)
            if resp.status_code not in (403, 404):
                existing = c
                break
            # Another identity's comment (e.g. github-actions[bot]'s, edited with a GitHub
            # App token): try the next one, then post our own
        if existing is None and create:
            new_body = body if found else (truncate_body(body_if_new) if body_if_new else body)
            resp = requests.post(base_url, headers=headers, json={"body": new_body}, timeout=10)
        elif existing is None and not found:
            return None  # create=False, and there is no DiffCost comment to update

        if resp.status_code in (200, 201):
            data = resp.json() if resp.content else {}
            comment_url = data.get("html_url")
            print(f"PR comment {'updated' if existing else 'posted'} successfully.")
            return comment_url
        print(f"::warning::Failed to post PR comment (HTTP {resp.status_code}): {_one_line(resp.text)[:200]}")
        if resp.status_code in (403, 404):
            print(f"::warning::{_PERMISSION_HINT}")
        return None

    except requests.RequestException as e:
        print(f"::warning::Failed to post PR comment: {_one_line(e)}")
        return None
