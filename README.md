# DiffCost

[![Latest release](https://img.shields.io/github/v/release/diffcost/action?label=release)](https://github.com/diffcost/action/releases)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**See what a pull request does to your BigQuery bill before it merges.**

DiffCost is a GitHub Action for teams that keep BigQuery SQL or a dbt project in GitHub. On each pull request (PR) it dry-runs the base version and the PR version of every changed SQL file or dbt model, in your own Google Cloud project, and posts the cost difference per query run as a PR comment. A dry run is free and returns the bytes a query would scan, never rows.

[Website](https://diffcost.dev) · [Documentation](https://diffcost.dev/docs) · [Quickstart](https://diffcost.dev/docs/quickstart) · [Pricing](https://diffcost.dev/pricing)

## What it does

- **Prices every changed query.** For each changed `.sql` file or dbt model: the cost of one run before and after the change, the difference in dollars and percent, and the bytes scanned.
- **Shows the likely cause.** DiffCost compares the structure of the two versions and flags what usually drives cost: a removed partition filter or `WHERE` condition, an added `SELECT *`, a new join or cross join, a new table. It reads each table's partitioning and clustering from BigQuery, so a removed partition filter names the table and its size.
- **Says how far to trust the number.** BigQuery reports when a dry-run estimate isn't exact. A figure for a clustered table is marked `≤` (an upper bound). A query BigQuery can't estimate before it runs (an external table, row-level security, some scripts) counts as unknown cost, never as $0.
- **Works with plain SQL and dbt.** For dbt it compares compiled SQL, so a change to a macro, a variable or `dbt_project.yml` that alters a model is priced too.
- **Blocks expensive merges** (paid plans). Set a limit per run, in dollars or percent, on the worst single query. A PR over the limit gets a failing `diffcost/cost-check` commit status, which you make a required check. Slack alerts and a dashboard with history and trends come with it.
- **Keeps your SQL in your project.** SQL goes only to BigQuery, as dry-run jobs with your credentials. The DiffCost service receives numbers, never SQL, table names or file paths.

The PR comment needs no DiffCost account and no API key.

## Example

The comment on a PR that removes a partition filter, in a repository with a $5.00 per-run limit:

> 💰 **DiffCost** — BigQuery Cost Report (cost per run)
>
> 🚫 **Merge blocked** — this PR exceeds a DiffCost cost policy:
> - fct_daily_revenue: +$7.57/run exceeds the $5.00/run per-query threshold
>
> The policy that applied: `.diffcost.yml` on the base branch.
>
> | Query / Model | Current | This PR | Scan Delta | Cost Change |
> |:---|---:|---:|---:|:---|
> | `fct_daily_revenue` | $2.09 | $9.66 | +1.21 TiB | +$7.57 (↑362%) |
> | `stg_orders` | $0.031 | $0.031 | 0 B | — |
> | **TOTAL** | **$2.12** | **$9.69** | **+1.21 TiB** | **+$7.57 (↑357%)** |
>
> ⚠️ This PR increases query costs by **$7.57/run** (+357%, ≈$227.10/mo at 30 runs/mo).\
> Largest single-query increase: `fct_daily_revenue` **+$7.57/run** (↑362%).\
> 📊 **Data Scanned per run:** 347.50 GiB → 1.55 TiB (+1.21 TiB)
>
> > ⚠️ **High Scan Warning:** `fct_daily_revenue` no longer filters on the partition column of analytics.stg\_orders (9.28 TiB table, partitioned daily), so it reads every partition.
>
> <details open>
> <summary>🔍 Root cause analysis</summary>
>
> - 🔴 **`fct_daily_revenue`**: Partition filter removed: `order_date >= '2024-01-01'`
>   - order\_date is the partition column of analytics.stg\_orders (9.28 TiB table, partitioned daily). Without this filter the query reads every partition.
>
> </details>

Every dollar figure is the cost of **one run** of the query; the monthly figure is a projection. On the Free plan the report is the same, without the merge-blocking verdict. [Reading the PR comment](https://diffcost.dev/docs/pr-comment) explains every row and message.

## Quickstart

DiffCost needs BigQuery access in one of two ways: a service account (steps 1 and 2), or **direct access**, where Workload Identity Federation grants the two roles to the repository's principal and there is no service account. For direct access, follow [direct access](https://diffcost.dev/docs/bigquery-access#choose-direct-access-or-a-service-account) instead of steps 1 and 2, then continue at step 3.

1. **Create a service account** that can run dry-run jobs in a Google Cloud project (`roles/bigquery.jobUser` on the project) and read the datasets your SQL uses (`roles/bigquery.dataViewer`). DiffCost only submits dry runs and reads table metadata. [Commands](https://diffcost.dev/docs/quickstart#1-create-a-service-account-for-dry-runs).
2. **Choose how the workflow authenticates:** [Workload Identity Federation](https://diffcost.dev/docs/bigquery-access) (keyless, recommended), or a service account key stored as the `GCP_SA_KEY` secret.
3. **Add the workflow:**

   ```yaml
   # .github/workflows/diffcost.yml
   name: DiffCost
   on: pull_request   # every PR, no `paths:` filter

   permissions:
     contents: read        # git diff against the base branch
     pull-requests: write  # post and update the PR comment
     id-token: write       # only for Workload Identity Federation

   jobs:
     diffcost:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v7
           with:
             fetch-depth: 0  # required: a shallow clone has no base history

         # Keyless (Workload Identity Federation)
         - uses: google-github-actions/auth@v3
           if: github.event.pull_request.head.repo.full_name == github.repository  # fork PRs get no OIDC token
           with:
             workload_identity_provider: 'projects/123456789012/locations/global/workloadIdentityPools/github-pool/providers/github-provider'
             # Only through a service account; leave it out for direct access:
             # service_account: 'diffcost-ci@my-project.iam.gserviceaccount.com'

         - uses: diffcost/action@v1
           with:
             sql_paths: "queries/**/*.sql"
             gcp_project: "my-project"
             # or, instead of the auth step: gcp_credentials: ${{ secrets.GCP_SA_KEY }}
   ```

4. **Open a PR** that changes a `.sql` file. DiffCost comments when the workflow finishes, and updates the same comment on later pushes.

Good to know:

- **Plain SQL files** are dry-run as written, so each one must run as it is, with qualified table names and no templating or query parameters, and BigQuery must be able to estimate it before it runs (no external table, no table with row-level security, no script BigQuery stops estimating early). See [what DiffCost can price](https://diffcost.dev/docs/quickstart#what-diffcost-can-price).
- **dbt projects** need a compiled manifest and, for full results, a baseline manifest from your main branch. Follow [dbt projects](https://diffcost.dev/docs/dbt) for that workflow.
- **Run DiffCost once per PR.** Several runs on one PR would share one comment and one commit status.
- **PRs from forks** get no secrets and a read-only token: DiffCost writes its report to the job summary instead of a comment.
- **Merge queues** need the `merge_group` trigger and a different `if:` on the auth step. See [merge queues](https://diffcost.dev/docs/merge-blocking#merge-queues).
- **Without BigQuery credentials** DiffCost still runs and posts the structural flags, without dollar amounts.

## Merge blocking

Merge blocking needs a paid plan or the free Team trial. In short:

1. Sign in at [diffcost.dev](https://diffcost.dev) with GitHub. The trial starts when the first owner of your GitHub organization signs in.
2. Create an API key in the dashboard, store it as the `DIFFCOST_API_KEY` secret, and pass it to the step: `api_key: ${{ secrets.DIFFCOST_API_KEY }}`.
3. Install the DiffCost GitHub App on the repositories that run DiffCost. The DiffCost service posts the `diffcost/cost-check` status through it.
4. Set a threshold in the dashboard, or in a `.diffcost.yml` in the repository root:

   ```yaml
   thresholds:
     absolute: 5.00   # block when any one query costs more than $5.00/run extra
     relative: 200    # or when any one query's cost goes up by more than 200%
   ```

5. Make `diffcost/cost-check` a required status check, with the DiffCost GitHub App as its source.

DiffCost reads `.diffcost.yml` from the PR's base commit, so a PR can't loosen its own thresholds by editing the file. A PR can still edit the workflow itself, so protect `.github/workflows/` as well: [Merge blocking](https://diffcost.dev/docs/merge-blocking) has every step, and [Security and data](https://diffcost.dev/docs/security) says how to keep the check trustworthy. A blocked PR fails the commit status, not the workflow step.

## Inputs

Every input is optional. The [configuration reference](https://diffcost.dev/docs/configuration) describes each one in full, with the `.diffcost.yml` keys.

| Input | Default | Description |
|---|---|---|
| `sql_paths` | Auto-detect | Glob for SQL files, such as `queries/**/*.sql`; separate several globs with commas. Without it: the dbt project if the base branch has one, otherwise every changed `.sql` file. |
| `manifest_path` | Auto-detect | Path of the `manifest.json` that `dbt compile` writes in this run. Default: `target/manifest.json` next to the base branch's `dbt_project.yml`. |
| `gcp_project` | The key's project | Google Cloud project that runs the dry-run jobs. Set it when you use Workload Identity Federation. |
| `gcp_credentials` | — | Raw JSON of a service account key. Leave empty with Workload Identity Federation. |
| `api_key` | — | DiffCost API key: dashboard history on every plan; merge blocking, Slack alerts and trends on paid plans and the trial. |
| `github_token` | `${{ github.token }}` | Token that posts the PR comment. The job needs `pull-requests: write`. |
| `config_path` | `.diffcost.yml` | Path of the config file. DiffCost always reads it from the base branch. |
| `base_ref` | The PR's base | Compare against this instead: a branch name, a `refs/heads/` or `refs/remotes/` ref, or a full commit SHA. Never a tag. |
| `cost_per_tb` | `6.25` | USD per TiB for on-demand pricing. With an API key, the dashboard's value is used. |
| `bq_pricing_model` | `on_demand` | `on_demand` shows dollars; `editions` shows bytes scanned. With an API key, the dashboard's value is used. |
| `comment_on_clean` | `false` | Post a comment even when the PR changes no SQL. |
| `fail_on_error` | `false` | Fail the step when DiffCost can't finish. By default the step passes and reports the problem in the log, the comment and the job summary. |
| `send_author` | `false` | Send the PR author's GitHub login to DiffCost for cost attribution. |

## Outputs

| Output | Description |
|---|---|
| `analysis_status` | `ok`, `no_changes` (no SQL changes, or not a pull request) or `error` (DiffCost could not analyze the PR). |
| `total_cost_delta` | Net cost change in USD per run, across costed queries (positive = increase). |
| `monthly_cost_delta` | `total_cost_delta` × runs per month (a projection). |
| `max_query_cost_delta` | Largest single-query increase in USD per run: what absolute thresholds compare against. |
| `max_regression_pct` | Largest single-query percentage increase. |
| `files_analyzed` | Number of changed SQL files or models analyzed. |
| `items_unknown` | Number of queries whose cost could not be determined (excluded from totals). |
| `total_bytes_before` | Bytes scanned per run by the base versions. |
| `total_bytes_after` | Bytes scanned per run by the PR versions. |
| `scan_multiplier` | PR scan volume divided by base scan volume, such as `4.20`. |
| `policy_exceeded` | `true` when the cost policy was exceeded, enforced or not. |
| `blocked` | `true` when the policy was exceeded and the failing `diffcost/cost-check` status was posted. |

## Plans

The Action is free: PR comments with per-run cost and structural flags, on unlimited repositories, with no account. Starter and Team add merge blocking, Slack alerts, longer dashboard history and cost trends; Team also adds per-repository thresholds in the dashboard and cost attribution per PR author. Each organization gets a free Team trial with no credit card.

See [pricing](https://diffcost.dev/pricing) and [plans and billing](https://diffcost.dev/docs/plans-and-billing).

## What data leaves your runner

- **To BigQuery, in your project:** the SQL of each changed query, as a dry-run job with your credentials, and a metadata read (`tables.get`) of each table those dry runs list. Neither reads rows.
- **To the DiffCost service, only with an API key:** the repository name, PR number and commit SHA; per-run costs and bytes, and counts; runs per month; any `.diffcost.yml` thresholds; and the PR author's login only with `send_author: true`.
- **Never to the DiffCost service:** SQL, table, column or model names, file paths, row data, your BigQuery credentials or the workflow's `GITHUB_TOKEN`.
- **To GitHub, in the PR comment and the job summary:** the cost report, with query names, structural findings that quote your SQL, and, with BigQuery credentials, table names with their size and partitioning from the tables' metadata. In a public repository they are public.
- **To Slack, only if you configure it:** an alert from a `.diffcost.yml` webhook goes from your runner straight to Slack and names queries: the one with the largest increase, and those in the block reasons.

[Security and data](https://diffcost.dev/docs/security) has the full list, including what the service stores.

## Versions and updates

`diffcost/action@v1` follows the latest `v1.x.y` release, so it gets fixes and new features without breaking changes. To control updates yourself, pin a release (`@v1.2.3`) or a full commit SHA and let Dependabot propose updates.

GitHub builds the Action on the runner from its `Dockerfile`: the base image is pinned by digest and every Python package by hash, so pinning a release or a commit locks DiffCost's code, its base image and every Python package it installs. See [Update the Action](https://diffcost.dev/docs/maintenance#update-the-action) and the [releases](https://github.com/diffcost/action/releases).

## Documentation

| | |
|---|---|
| [Quickstart](https://diffcost.dev/docs/quickstart) | Set up DiffCost on a repository |
| [How DiffCost works](https://diffcost.dev/docs/how-it-works) | What is compared, priced and flagged |
| [BigQuery access and keyless auth](https://diffcost.dev/docs/bigquery-access) | Permissions and Workload Identity Federation |
| [dbt projects](https://diffcost.dev/docs/dbt) | Manifests, baselines and the main-branch artifact |
| [Merge blocking](https://diffcost.dev/docs/merge-blocking) | API key, GitHub App and the `diffcost/cost-check` required check |
| [Reading the PR comment](https://diffcost.dev/docs/pr-comment) | Every row, message and status |
| [Thresholds and alerts](https://diffcost.dev/docs/thresholds) | Cost policies and Slack |
| [Using the dashboard](https://diffcost.dev/docs/dashboard) | History, settings, team and billing |
| [Configuration reference](https://diffcost.dev/docs/configuration) | Inputs, outputs and `.diffcost.yml` |
| [Monitoring](https://diffcost.dev/docs/monitoring) and [Maintenance](https://diffcost.dev/docs/maintenance) | Check enforcement, update the Action, rotate keys |
| [Troubleshooting](https://diffcost.dev/docs/troubleshooting) | Error messages and their fixes |
| [Plans and billing](https://diffcost.dev/docs/plans-and-billing) | Free, Starter, Team and the trial |
| [Security and data](https://diffcost.dev/docs/security) | What leaves your runner and what is stored |

## Support

- **Something not working?** Start with [Troubleshooting](https://diffcost.dev/docs/troubleshooting): it lists each message DiffCost shows and its fix.
- **Bugs and questions:** [open an issue](https://github.com/diffcost/action/issues) or email [support@diffcost.dev](mailto:support@diffcost.dev). Leave out SQL, credentials and API keys.
- **Security issues:** email support@diffcost.dev with "Security" in the subject instead of opening an issue. See [SECURITY.md](SECURITY.md).

## Contributing

This repository is published from DiffCost's source repository at each release, so a commit made here is overwritten by the next one. Bug reports and suggestions are welcome as issues. If you open a pull request, we apply an accepted change to the source and it ships in the next release.

To run the tests (Python 3.11 or later):

```bash
pip install -r requirements.txt   # the hash-pinned lock the Docker image installs
pip install pytest ruff
pytest tests/
ruff check src/ tests/
```

## License

[Apache License 2.0](LICENSE). Copyright 2026 DiffCost.
