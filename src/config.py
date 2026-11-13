"""Configuration loading from action inputs and .diffcost.yml.

Precedence (highest first):
  1. With an `api_key`: dashboard settings served by GET /v1/config
     (thresholds, BigQuery pricing model and $/TiB).
  2. `.diffcost.yml` read from the BASE branch (never the PR branch).
  3. Action inputs.
Invalid values never crash the run: they are reported as warnings and ignored.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from gitutil import (
    GitError,
    base_dbt_project_dirs,
    check_checkout_is_merge_group,
    check_checkout_matches_pr,
    explain_git_failure,
    resolve_base_ref,
    run_git,
)

# Git refs are passed as argv (never through a shell). Reject only what could be read
# as an option or a revision expression: a leading "-", whitespace/control characters,
# "..", "@{", and the characters git forbids in ref names (~ ^ : ? * [ \\).
_VALID_GIT_REF_RE = re.compile(r"^(?!-)(?!.*\.\.)(?!.*@\{)[^\s~^:?*\[\\\x00-\x1f\x7f]+$")
# A merge queue branch: refs/heads/gh-readonly-queue/<base branch>/pr-<number>-<base sha>
_MERGE_GROUP_PR_RE = re.compile(r"/pr-(\d+)-[0-9a-f]{40}$")
_DEPENDABOT_LOGIN = "dependabot[bot]"

DEFAULT_COST_PER_TIB = 6.25
DEFAULT_RUNS_PER_MONTH = 30


@dataclass
class ThresholdConfig:
    """Per-run USD / percent limits applied to the worst single query."""

    absolute_usd: float | None = None
    relative_pct: float | None = None

    def is_set(self) -> bool:
        return self.absolute_usd is not None or self.relative_pct is not None


@dataclass
class ActionConfig:
    sql_paths: str
    manifest_path: str
    gcp_credentials: str
    gcp_project: str
    api_key: str
    config_path: str
    base_ref: str
    cost_per_tb: float  # USD per TiB (BigQuery bills on-demand queries per TiB)
    comment_on_clean: bool
    bq_pricing_model: str = "on_demand"
    fail_on_error: bool = False
    send_author: bool = False

    # Effective thresholds (dashboard thresholds replace .diffcost.yml ones when set)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    # Thresholds as read from the base-branch .diffcost.yml (sent to the backend)
    yml_thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    ignored_paths: list[str] = field(default_factory=list)
    slack_webhook: str = ""
    default_runs_per_month: int = DEFAULT_RUNS_PER_MONTH
    trial_days_remaining: int | None = None
    trial_ended: bool = False
    # A Stripe subscription exists but no longer pays (unpaid or paused): the effective
    # plan is Free, and the plan line says to update the payment method
    subscription_inactive: bool = False

    # From GET /v1/config (empty without an api_key or when the backend is unreachable)
    features: list[str] = field(default_factory=list)
    # A dashboard threshold replaced the .diffcost.yml thresholds (its Slack webhook, if
    # any, is the backend's to send; the .diffcost.yml webhook is not used then)
    dashboard_thresholds: bool = False
    # Why the last DiffCost backend call failed (a NOT_ENFORCED_REASONS key: rejected,
    # unreachable, invalid, error, timeout); "" when it succeeded or was not made
    backend_error: str = ""

    # Runtime context (populated from GitHub env vars)
    github_token: str = ""
    github_repository: str = ""
    pr_number: int = 0
    commit_sha: str = ""
    github_event_name: str = ""
    workspace: str = ""
    # PR author login from the event payload (team attribution, opt-in via send_author)
    pr_author: str = ""
    # A merge queue run (`merge_group` event): the queue commit is analyzed against its
    # parent, the status goes on the queue commit, and no PR comment is posted
    merge_group: bool = False
    # Why `api_key` may be empty: secrets aren't passed to fork PRs, and Dependabot PRs
    # only get Dependabot secrets
    fork_pr: bool = False
    dependabot_pr: bool = False

    # Problems reading the cost policy (.diffcost.yml); shown in the PR comment
    policy_warnings: list[str] = field(default_factory=list)
    # Invalid action inputs; reported as "could not analyze" instead of crashing
    config_error: str = ""
    # BigQuery credentials were provided but a client could not be created
    bq_client_error: str = ""

    # Commit SHA actually used for diffs and for reading .diffcost.yml; empty when unresolved
    resolved_base_ref: str = ""
    # Why the base ref could not be resolved (surfaced instead of "no changes")
    git_error: str = ""
    # What `base_dbt_project_dirs` last listed, with the workspace and commit it is for
    _base_projects: tuple[str, str, list[str]] | None = field(default=None, init=False, repr=False, compare=False)

    def base_dbt_project_dirs(self) -> list[str] | None:
        """Directories ("" = repo root) holding a dbt_project.yml on the base commit, or
        None when the base is unresolved or can't be listed. Adapter detection and the dbt
        adapter both need it, so a commit's tree is listed once per run; a failed listing
        isn't kept."""
        if not self.resolved_base_ref:
            return None
        workspace = self.workspace or "."
        kept = self._base_projects
        if kept is None or kept[:2] != (workspace, self.resolved_base_ref):
            dirs = base_dbt_project_dirs(workspace, self.resolved_base_ref)
            if dirs is None:
                return None
            kept = self._base_projects = (workspace, self.resolved_base_ref, dirs)
        return list(kept[2])

    @property
    def base_rev(self) -> str:
        """The ref to diff against: the resolved ref when known, else the raw input."""
        return self.resolved_base_ref or self.base_ref

    @property
    def base_label(self) -> str:
        """Human-readable base for logs and warnings, e.g. "main (1a2b3c4d)"."""
        if self.resolved_base_ref:
            return f"{self.base_ref} ({self.resolved_base_ref[:8]})"
        return self.base_ref


def load_config() -> ActionConfig:
    """Load configuration from GitHub Actions inputs and environment.

    Invalid inputs never raise: they are recorded in `config_error` (or `git_error`)
    and reported as "DiffCost could not analyze this PR".
    """
    explicit_base_ref = _get_input("base_ref", "")
    config = ActionConfig(
        sql_paths=_get_input("sql_paths", ""),
        manifest_path=_get_input("manifest_path", ""),
        gcp_credentials=_get_input("gcp_credentials", ""),
        gcp_project=_get_input("gcp_project", ""),
        api_key=_get_input("api_key", ""),
        config_path=_get_input("config_path", ".diffcost.yml") or ".diffcost.yml",
        base_ref=explicit_base_ref or os.environ.get("GITHUB_BASE_REF", "") or "main",
        cost_per_tb=_parse_float_input("cost_per_tb", DEFAULT_COST_PER_TIB, minimum=0.0),
        comment_on_clean=_get_input("comment_on_clean", "false").lower() == "true",
        bq_pricing_model=_parse_pricing_model(_get_input("bq_pricing_model", "on_demand")),
        fail_on_error=_get_input("fail_on_error", "false").lower() == "true",
        send_author=_get_input("send_author", "false").lower() == "true",
        github_token=_get_input("github_token", "") or os.environ.get("GITHUB_TOKEN", ""),
        github_repository=os.environ.get("GITHUB_REPOSITORY", ""),
        github_event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
        workspace=os.environ.get("GITHUB_WORKSPACE", ""),
    )

    event = _read_event()
    pr = event.get("pull_request") or {}
    merge_group: dict = {}
    if config.github_event_name == "merge_group":
        raw_group = event.get("merge_group")
        merge_group = raw_group if isinstance(raw_group, dict) else {}
        config.merge_group = True
        config.pr_number = merge_group_pr_number(str(merge_group.get("head_ref") or ""))
        config.commit_sha = str(merge_group.get("head_sha") or "")
        queue_base = str(merge_group.get("base_ref") or "")
        if not explicit_base_ref and queue_base.startswith("refs/heads/"):
            config.base_ref = queue_base[len("refs/heads/"):]  # the label; the diff base is base_sha
        if not config.pr_number:
            config.config_error = (
                "Could not find the pull request number in the merge queue branch "
                f"`{str(merge_group.get('head_ref') or '')[:200]}` (expected "
                "`gh-readonly-queue/<branch>/pr-<number>-<sha>`)."
            )
    else:
        config.pr_number = _get_pr_number()
        head_sha = str((pr.get("head") or {}).get("sha") or "")
        # Without a head commit in the event: GITHUB_SHA (the merge commit on PRs, but better than nothing)
        config.commit_sha = head_sha or os.environ.get("GITHUB_SHA", "")
        config.pr_author = str((pr.get("user") or {}).get("login") or os.environ.get("GITHUB_ACTOR", ""))[:255]
        head_repo = ((pr.get("head") or {}).get("repo") or {}).get("full_name")
        base_repo = ((pr.get("base") or {}).get("repo") or {}).get("full_name")
        # Only `pull_request` withholds secrets from fork PRs (`pull_request_target` runs
        # with the base repository's secrets, so an empty key there is a missing secret)
        config.fork_pr = config.github_event_name == "pull_request" and bool(
            head_repo and base_repo and head_repo != base_repo
        )
    config.dependabot_pr = _DEPENDABOT_LOGIN in (
        str((pr.get("user") or {}).get("login") or ""), os.environ.get("GITHUB_ACTOR", ""),
    )

    # Validate user-supplied paths to prevent path traversal
    for path_name, path_val in [("manifest_path", config.manifest_path), ("config_path", config.config_path)]:
        if path_val and (".." in path_val or path_val.startswith("/")):
            config.config_error = f"`{path_name}` must be a relative path without `..` components (got {path_val[:80]!r})."
            config.manifest_path = "" if path_name == "manifest_path" else config.manifest_path
            config.config_path = ".diffcost.yml" if path_name == "config_path" else config.config_path

    # Validate base_ref: refs are passed as argv, but must not be read as options or revision expressions
    if not _VALID_GIT_REF_RE.match(config.base_ref):
        config.git_error = f"`base_ref` is not a valid branch name: {config.base_ref[:80]!r}."
        return config

    workspace = config.workspace or "."
    if config.merge_group:
        if not explicit_base_ref and not merge_group.get("base_sha"):
            config.git_error = "The merge queue event has no base commit (`merge_group.base_sha`)."
            return config
        if not config.commit_sha:
            config.git_error = "The merge queue event has no head commit (`merge_group.head_sha`)."
            return config
        try:
            base_sha = "" if explicit_base_ref else str(merge_group["base_sha"])
            config.resolved_base_ref = resolve_base_ref(workspace, config.base_ref, base_sha=base_sha, merge_group=True)
            check_checkout_is_merge_group(workspace, config.commit_sha)
        except GitError as e:
            config.git_error = str(e)
            config.resolved_base_ref = ""
        _load_diffcost_yml(config)
        return config

    base_sha = "" if explicit_base_ref else str((pr.get("base") or {}).get("sha") or "")
    try:
        config.resolved_base_ref = resolve_base_ref(workspace, config.base_ref, base_sha=base_sha)
        if head_sha:
            merge_base_parent = check_checkout_matches_pr(workspace, head_sha)
            # On GitHub's merge commit, diff against exactly the base it merged into
            if merge_base_parent and not explicit_base_ref:
                resolved = resolve_base_ref(workspace, "", base_sha=merge_base_parent)
                config.resolved_base_ref = resolved
    except GitError as e:
        config.git_error = str(e)
        config.resolved_base_ref = ""

    _load_diffcost_yml(config)
    return config


def _get_input(name: str, default: str) -> str:
    """Read a GitHub Actions input. Inputs are passed as INPUT_<NAME> env vars."""
    env_name = f"INPUT_{name.upper()}"
    return os.environ.get(env_name, default).strip()


def _warn(msg: str) -> None:
    # One line: a workflow command ends at a newline (a YAML error spans several)
    print(f"::warning::{' '.join(str(msg).split())}")


def _policy_warn(config: ActionConfig, msg: str) -> None:
    """A problem with the cost policy file: logged and shown in the PR comment."""
    _warn(msg)
    config.policy_warnings.append(msg)


def _parse_float_input(name: str, default: float, minimum: float | None = None) -> float:
    raw = _get_input(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):  # "nan" and "inf" parse as floats, and compare false with any minimum
        _warn(f"Input `{name}` must be a number (got {raw[:40]!r}). Using {default}.")
        return default
    if minimum is not None and value < minimum:
        _warn(f"Input `{name}` must be >= {minimum}. Using {default}.")
        return default
    return value


def _parse_pricing_model(raw: str) -> str:
    value = (raw or "on_demand").strip().lower()
    if value not in ("on_demand", "editions"):
        _warn(f"bq_pricing_model must be 'on_demand' or 'editions' (got {raw[:40]!r}). Using on_demand.")
        return "on_demand"
    return value


def merge_group_pr_number(head_ref: str) -> int:
    """The PR number in a merge queue branch name (`gh-readonly-queue/<branch>/pr-<N>-<sha>`,
    the PR at the tip of the queue entry), or 0 when it can't be read."""
    match = _MERGE_GROUP_PR_RE.search(head_ref or "")
    return int(match.group(1)) if match else 0


def _get_pr_number() -> int:
    """Extract PR number from GITHUB_REF (refs/pull/<number>/merge) or event payload."""
    ref = os.environ.get("GITHUB_REF", "")
    if "/pull/" in ref:
        try:
            return int(ref.split("/pull/")[1].split("/")[0])
        except (ValueError, IndexError):
            pass

    # Fallback: read from event payload
    event = _read_event()
    pr = event.get("pull_request") or {}
    try:
        return int(pr.get("number") or 0)
    except (TypeError, ValueError):
        return 0


def _read_event() -> dict:
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path or not Path(event_path).exists():
        return {}
    try:
        import json

        with open(event_path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _parse_threshold(config: ActionConfig, raw, key: str, maximum: float) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        _policy_warn(config, f".diffcost.yml thresholds.{key} must be a number. Ignoring it.")
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        _policy_warn(config, f".diffcost.yml thresholds.{key} must be a number (got {str(raw)[:40]!r}). Ignoring it.")
        return None
    if not (0 <= value <= maximum):
        _policy_warn(config, f".diffcost.yml thresholds.{key} must be between 0 and {maximum:,.0f}. Ignoring it.")
        return None
    # The backend stores and evaluates thresholds with 4 decimal places; match it
    return round(value, 4)


def _load_diffcost_yml(config: ActionConfig) -> None:
    """Load .diffcost.yml from the BASE branch (not PR branch) to prevent bypass."""
    if not config.resolved_base_ref:
        # The run stops with "could not analyze" (the adapters raise git_error), so the
        # policy is never silently skipped.
        _warn(f"Skipping {config.config_path}: {config.git_error}")
        return

    yml_path = config.config_path
    base = config.base_label

    # Read from base branch to prevent developers from weakening thresholds in their PR
    try:
        result = run_git(config.workspace or ".", ["show", f"{config.resolved_base_ref}:{yml_path}"], timeout=10)
    except GitError as e:
        _policy_warn(config, f"Could not read {yml_path} from {base}: {e}. Its thresholds were not applied.")
        return

    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "does not exist" in stderr or "exists on disk, but not in" in stderr:
            return  # No config file on the base branch — that's fine
        _policy_warn(
            config,
            f"Could not read {yml_path} from {base}: {explain_git_failure(result.stderr)}. Its thresholds were not applied.",
        )
        return

    try:
        raw = yaml.safe_load(result.stdout)
    except yaml.YAMLError as e:
        _policy_warn(config, f"{yml_path} on {base} is not valid YAML ({str(e)[:120]}). Its thresholds were not applied.")
        return

    if raw is None:
        return
    if not isinstance(raw, dict):
        _policy_warn(config, f"{yml_path} must be a YAML mapping. Its thresholds were not applied.")
        return

    thresholds = raw.get("thresholds") or {}
    if isinstance(thresholds, dict):
        # Caps at or below the backend's (absolute 1,000,000, relative 100,000), so an
        # accepted value never makes the backend reject the comparison
        absolute = _parse_threshold(config, thresholds.get("absolute"), "absolute", 1_000_000)
        relative = _parse_threshold(config, thresholds.get("relative"), "relative", 100_000)
        config.yml_thresholds = ThresholdConfig(absolute_usd=absolute, relative_pct=relative)
        config.thresholds = ThresholdConfig(absolute_usd=absolute, relative_pct=relative)
    else:
        _policy_warn(config, f"{yml_path}: `thresholds` must be a mapping. Its thresholds were not applied.")

    ignored = raw.get("ignored_paths", [])
    if isinstance(ignored, list):
        config.ignored_paths = [str(p) for p in ignored]

    if "default_runs_per_month" in raw:
        runs = raw["default_runs_per_month"]
        try:
            runs_int = int(runs) if not isinstance(runs, bool) else None
        except (TypeError, ValueError):
            runs_int = None
        if runs_int is None or not (1 <= runs_int <= 100_000):
            _warn(f"{yml_path}: default_runs_per_month must be an integer between 1 and 100,000. Using {DEFAULT_RUNS_PER_MONTH}.")
        else:
            config.default_runs_per_month = runs_int

    webhook = raw.get("slack_webhook", "")
    if isinstance(webhook, str) and webhook:
        # Resolve env var references like ${{ SLACK_WEBHOOK }} from the step's `env:`
        resolved = _resolve_env_var(webhook)
        # Validate webhook URL format to prevent data exfiltration via misconfigured URLs
        if resolved and re.match(r"^https://hooks\.slack\.com/", resolved):
            config.slack_webhook = resolved
        elif resolved:
            _warn("slack_webhook must be a https://hooks.slack.com/ URL. Ignoring.")
        else:
            _warn(
                "slack_webhook resolved to an empty value. Set it with `${{ SLACK_WEBHOOK }}` in "
                ".diffcost.yml and pass `env: SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}` on the DiffCost step."
            )

    pricing_model = raw.get("bq_pricing_model", "")
    if isinstance(pricing_model, str) and pricing_model:
        config.bq_pricing_model = _parse_pricing_model(pricing_model)

    if "fail_on_error" in raw and isinstance(raw["fail_on_error"], bool):
        config.fail_on_error = raw["fail_on_error"]
    if "send_author" in raw and isinstance(raw["send_author"], bool):
        config.send_author = raw["send_author"]


# Env vars that may be resolved via ${{ VAR_NAME }} in .diffcost.yml.
# This prevents config files from exfiltrating secrets like GITHUB_TOKEN.
_ALLOWED_ENV_VARS = frozenset({
    "SLACK_WEBHOOK",
    "SLACK_WEBHOOK_URL",
    "DIFFCOST_SLACK_WEBHOOK",
})


def _resolve_env_var(value: str) -> str:
    """Resolve ${{ VAR_NAME }} patterns from environment variables.

    Only variables in the allowlist can be resolved, to prevent the base
    branch config from exfiltrating CI secrets (e.g. GITHUB_TOKEN).
    """

    def replacer(match: re.Match) -> str:
        var_name = match.group(1).strip()
        if var_name not in _ALLOWED_ENV_VARS:
            return ""  # Silently ignore disallowed vars
        return os.environ.get(var_name, "")

    return re.sub(r"\$\{\{\s*(\w+)\s*\}\}", replacer, value)
