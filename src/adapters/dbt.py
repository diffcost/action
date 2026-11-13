"""dbt adapter — works with dbt manifest.json for richer context.

A model counts as changed when its source-file checksum differs OR its compiled
SQL differs. Compiled SQL catches changes outside the model file (macros, vars,
`dbt_project.yml`, schema yml, sources). Before comparing compiled SQL, relation
names of nodes present in both manifests are mapped from the PR manifest to the
production one, so manifests compiled against different targets (CI schema vs
production schema) do not mark every model as changed. The SQL that gets dry-run
is still each manifest's compiled SQL, so compile the PR manifest against the
production target (`dbt compile --target prod`).

Model paths in a manifest are relative to the dbt project; they are prefixed with
the project's directory (taken from the base commit's `dbt_project.yml`) so they
match repo-relative git paths and ignore globs.

Anything that could hide a change fails closed: a PR manifest or baseline that is
committed, symlinked or unreadable is an error (never a silent drop into
no-baseline mode); models compiled in only one manifest whose file changed are
reported as unknown cost; without a baseline, changes outside model files (macros,
yml, project config, local packages) are one unknown item; and changed `.sql` files
outside the analyzed dbt project are one "not analyzed" unknown item.

Each model also gets `changed_upstream` (a model, seed or snapshot it depends on,
transitively, is new or changed in this PR; ephemeral models count only through their
parents) and, in memory only, the relations upstream nodes create in this PR. A PR-side
"Unrecognized name" / "Not found" is "depends on objects this PR creates" only when these
explain it; otherwise the model's cost is unknown.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

from adapters import ChangedSQL
from adapters.sql_files import _git_show, _matches_glob, exempt_entry, git_changed_paths
from config import ActionConfig
from gitutil import INSTALLED_PACKAGE_DIRS, GitError, base_dbt_project_dirs, is_committed_copy, is_tracked, run_git


class ManifestError(RuntimeError):
    """The PR manifest cannot be used for analysis."""


# Baseline manifest locations, in priority order. The file must be written during the run
# (the base commit compiled in the job, or a downloaded artifact), never a file committed
# in the PR (the PR author controls those).
_BASELINE_CANDIDATES = (".diffcost-baseline/manifest.json", "target/prod-manifest.json")

# Node types whose `relation_name` can appear in another model's compiled SQL
_RELATION_TYPES = ("model", "seed", "snapshot")

# Project files whose changes alter compiled SQL without touching a model file
# (checked when there is no baseline, or when some models can't be compared).
_PROJECT_FILES = ("dbt_project.yml", "packages.yml", "dependencies.yml", "selectors.yml")
# dbt_project.yml path settings and their defaults (relative to the project directory).
# `source-paths` / `data-paths` are the pre-1.0 names of model-paths / seed-paths.
_MODEL_PATH_KEYS = ("model-paths", "source-paths")
_MODEL_PATH_DEFAULT = ("models",)
_OTHER_PATH_KEYS = {
    "macro-paths": ("macros",),
    "seed-paths": ("seeds",),
    "snapshot-paths": ("snapshots",),
    "analysis-paths": ("analyses", "analysis"),
}
_SEED_KEYS = ("seed-paths", "data-paths")
_TEST_PATH_KEYS = ("test-paths",)
_TEST_PATH_DEFAULT = ("tests",)
_NON_MODEL_ITEM = "non-model dbt files"
_OUTSIDE_ITEM = "SQL outside the analyzed dbt project"


@dataclass
class ProjectLayout:
    """Where the analyzed dbt project keeps its files, as repo-relative directories
    ("" = repo root), read from the base commit (the PR can't move them).

    Each path setting is the configured paths plus dbt's defaults, so a configured path
    that can't be read (an `env_var()` template, `..`) never drops a directory (fail closed).
    `model_dirs` / `config_dirs` decide what counts as a non-model change; `covered_dirs`
    (those plus the test and target paths) decide which `.sql` files the analysis covers.
    """

    project_dir: str = ""
    model_dirs: list[str] = field(default_factory=list)
    # macro, seed, snapshot and analysis paths
    config_dirs: list[str] = field(default_factory=list)
    # `local:` packages from packages.yml / dependencies.yml (never one holding the project)
    package_dirs: list[str] = field(default_factory=list)
    name: str | None = None  # `name:` in dbt_project.yml (the root package)
    covered_dirs: list[str] = field(default_factory=list)


@dataclass
class Manifest:
    models: dict[str, dict] = field(default_factory=dict)  # compiled models only
    all_model_ids: set[str] = field(default_factory=set)  # every model node, compiled or not
    relations: dict[str, str] = field(default_factory=dict)  # node / source id -> relation_name
    # Every model node, compiled or not: {"name", "path", "repo_path", "checksum"}. "path" is
    # the project directory joined with dbt's original_file_path (what the comment shows);
    # "repo_path" is that path for root-project nodes and None for package nodes, whose
    # original_file_path is relative to their package. Only "repo_path" is compared with
    # the files git says changed.
    info: dict[str, dict] = field(default_factory=dict)
    # Every model, seed and snapshot node: {"repo_path", "checksum", "parents"
    # (depends_on.nodes), "config" (materialized, database, schema, alias),
    # "ephemeral", "compiled" (snapshots only; models keep theirs in `models`)}
    nodes: dict[str, dict] = field(default_factory=dict)
    project_name: str | None = None  # the root project (manifest metadata, dbt >= 1.6)

    @property
    def uncompiled(self) -> int:
        return len(self.all_model_ids) - len(self.models)


def find_changed_dbt_models(config: ActionConfig) -> list[ChangedSQL]:
    """
    Find dbt models that changed between base branch and HEAD.
    Compares the PR manifest against the production manifest.
    """
    workspace = config.workspace or "."
    ignored = config.ignored_paths

    # The base commit is needed for .diffcost.yml and for git-change checks; never
    # analyze without it (a dropped cost policy would look like a passing run).
    if config.git_error:
        raise GitError(config.git_error)

    # Load PR manifest (from current checkout)
    pr_manifest_path = Path(workspace) / config.manifest_path
    if not pr_manifest_path.exists():
        raise ManifestError(
            f"Manifest not found at `{config.manifest_path}`. Run `dbt compile --target prod` before DiffCost."
        )
    if not _is_plain_file(workspace, config.manifest_path):
        # A committed symlink can point the manifest at the baseline file (dbt writes
        # through it), so both sides would be the same file and nothing would change.
        raise ManifestError(
            f"`{config.manifest_path}` is a symlink or inside a symlinked directory. The PR manifest must be the "
            "file `dbt compile --target prod` writes in this run: remove the symlink from the repository."
        )
    if _is_committed_manifest(workspace, config.manifest_path):
        raise ManifestError(
            f"`{config.manifest_path}` is the copy committed in the repository, not one compiled in this run, "
            "so it may not match this PR's SQL. Run `dbt compile --target prod` before DiffCost "
            "(and keep `target/` out of git)."
        )

    base_projects = config.base_dbt_project_dirs()
    project_dir = dbt_project_dir(workspace, config.manifest_path, base_projects)
    layout = project_layout(workspace, config.resolved_base_ref, project_dir)
    pr = _parse_manifest(pr_manifest_path, project_dir, layout.name)
    if not pr.all_model_ids:
        raise ManifestError(
            f"`{config.manifest_path}` contains no dbt models. Run `dbt compile --target prod` before DiffCost."
        )
    if not pr.models:
        raise ManifestError(
            f"`{config.manifest_path}` has {pr.uncompiled} model(s) but no compiled SQL. "
            "Run `dbt compile` (not `dbt parse`) before DiffCost."
        )
    if not any((Path(workspace) / m["path"]).is_file() for m in pr.models.values()):
        where = f"`{project_dir}/`" if project_dir else "the repository root"
        raise ManifestError(
            f"None of the models in `{config.manifest_path}` were found under {where}, so the dbt project "
            "directory could not be determined. DiffCost uses the directory with `dbt_project.yml` (on the base "
            "branch) at or above `manifest_path`: point `manifest_path` at `<project>/target/manifest.json`."
        )
    if pr.uncompiled:
        print(f"::warning::{pr.uncompiled} model(s) in the PR manifest have no compiled SQL.")

    changed_paths = git_changed_paths(workspace, config.base_rev)
    head_projects = base_dbt_project_dirs(workspace, "HEAD")  # includes projects the PR adds
    outside = _outside_project_item(changed_paths, layout, base_projects, ignored, head_projects)

    # Load the baseline manifest (written in the run, or the base branch's committed copy)
    prod = _load_production_manifest(workspace, config, project_dir, pr_manifest_path, pr.project_name or layout.name)

    if prod is None:
        results = _changed_without_baseline(changed_paths, pr, ignored, layout)
        changed, new_relations = _changed_nodes(pr, None, changed_paths, {}, {})
        return _mark_changed_upstream(results, pr, changed, new_relations) + outside

    relation_map = {
        pr.relations[node_id]: prod.relations[node_id]
        for node_id in pr.relations.keys() & prod.relations.keys()
        if pr.relations[node_id] != prod.relations[node_id]
    }

    # Compare manifests to find changed models
    results = []
    skipped_uncompiled = 0
    compared: dict[str, bool] = {}  # model id -> compiled SQL or checksum changed
    for model_id in sorted(pr.all_model_ids | prod.all_model_ids):
        pr_model = pr.models.get(model_id)
        prod_model = prod.models.get(model_id)
        info = pr.info.get(model_id) or prod.info.get(model_id)

        if any(_matches_glob(info["path"], p) for p in ignored):
            continue

        # Not compiled in a manifest that has the model (selective compile): not new and
        # not removed, just not comparable. If its file or checksum changed, the cost
        # change is unknown, never "no changes".
        uncompiled_in = [
            side for side, m, man in (("PR", pr_model, pr), ("baseline", prod_model, prod))
            if m is None and model_id in man.all_model_ids
        ]
        if uncompiled_in or (pr_model is None and prod_model is None):
            if model_id not in pr.all_model_ids:
                continue  # deleted in this PR: removing a model can't add cost
            if _changed_in_git_or_checksum(model_id, info["repo_path"], changed_paths, pr, prod):
                where = " and ".join(uncompiled_in) or "either"
                results.append(_unknown_model(
                    model_id, info, prod_model, pr_model,
                    f"the model is not compiled in the {where} manifest (selective compile?), "
                    "so its cost can't be compared",
                ))
            else:
                skipped_uncompiled += 1
            continue

        model = pr_model or prod_model
        metadata = {"unique_id": model_id}
        if pr_model and prod_model:
            compared[model_id] = _model_changed(pr_model, prod_model, relation_map)
        if pr_model and prod_model and not compared[model_id]:
            if not model["repo_path"] or model["repo_path"] not in changed_paths:
                continue
            # The model's file changed in git, but the manifest says nothing changed:
            # the PR manifest was not compiled from this checkout.
            metadata["unknown_reason"] = "the model file changed but the PR manifest was not recompiled"

        results.append(
            ChangedSQL(
                name=model["name"],
                path=model["path"],
                production_sql=prod_model["compiled_code"] if prod_model else None,
                pr_sql=pr_model["compiled_code"] if pr_model else None,
                adapter="dbt",
                metadata=metadata,
            )
        )

    if skipped_uncompiled:
        print(
            f"::warning::{skipped_uncompiled} model(s) whose files did not change are not compiled in one of the "
            "two manifests (selective compile?) and were not compared."
        )
        # Their compiled SQL can't be compared, so a macro, var or yml change could alter
        # them unseen: that change is unknown cost, never "no changes".
        others = _non_model_changes(changed_paths, pr, layout, ignored)
        if others:
            results.append(_non_model_item(
                others,
                f"can change the SQL of {skipped_uncompiled} model(s) that are not compiled in both manifests "
                "(selective compile?), so their cost can't be compared",
            ))
    changed, new_relations = _changed_nodes(pr, prod, changed_paths, relation_map, compared)
    return _mark_changed_upstream(results, pr, changed, new_relations) + outside


def _changed_nodes(
    pr: Manifest,
    prod: Manifest | None,
    changed_paths: dict[str, str],
    relation_map: dict[str, str],
    compared: dict[str, bool],
) -> tuple[set[str], set[str]]:
    """Models, seeds and snapshots this PR adds or changes (ignored paths included), and those
    whose relation this PR creates or moves.

    Changed: the node's file changed in git (root-project nodes only: see `repo_path` on
    `Manifest.info`) or, with a baseline, the node is new, gets a relation (an
    ephemeral model materialized), or its checksum, compiled SQL or relation config
    (materialized, database, schema, alias) differs. A relation name that differs only by
    target (CI vs production schema) is not a change. Without a baseline only git changes
    count: a macro or yml change can't be attributed to a node.

    New relation: with a baseline, the node had no relation in production or its relation
    config moved it; without one, the node's file was added in this PR.
    """
    changed: set[str] = set()
    new_relations: set[str] = set()
    for node_id, node in pr.nodes.items():
        git_status = changed_paths.get(node["repo_path"]) if node["repo_path"] else None
        if git_status:
            changed.add(node_id)
        if prod is None:
            if git_status == "A":
                new_relations.add(node_id)
            continue
        base = prod.nodes.get(node_id)
        moved = base is not None and node["config"][1:] != base["config"][1:]
        if node_id in pr.relations and (node_id not in prod.relations or moved):
            new_relations.add(node_id)
        if base is None or moved or node_id in new_relations or node["config"] != base["config"]:
            changed.add(node_id)
        elif node["checksum"] and base["checksum"] and node["checksum"] != base["checksum"]:
            changed.add(node_id)
        elif node_id in compared:
            if compared[node_id]:
                changed.add(node_id)
        elif node_id in pr.models and node_id in prod.models:
            if _model_changed(pr.models[node_id], prod.models[node_id], relation_map):
                changed.add(node_id)  # e.g. a model under ignored_paths
        elif node["compiled"] and base["compiled"]:
            if _sql_hash(_map_relations(node["compiled"], relation_map)) != _sql_hash(base["compiled"]):
                changed.add(node_id)  # a snapshot whose compiled SQL changed
    return changed, new_relations


def _mark_changed_upstream(
    items: list[ChangedSQL], pr: Manifest, changed: set[str], new_relations: set[str]
) -> list[ChangedSQL]:
    """Set on each model, from the nodes it depends on (directly or through other nodes):

    - `changed_upstream`: one of them is new or changed in this PR. Ephemeral models create
      no object, so they never count themselves, but their own parents do.
    - `new_upstream_relations`: the relation names of those whose relation this PR creates.
      Kept in memory only (never in the payload, the comment or the logs).

    Only then can a PR-side "Unrecognized name" / "Not found" mean the model reads something
    this PR creates.
    """
    counted = {n for n in changed if not pr.nodes[n]["ephemeral"]}
    created = {n for n in new_relations if n in pr.relations and not pr.nodes[n]["ephemeral"]}
    memo: dict[str, tuple[bool, frozenset]] = {}
    for item in items:
        model_id = (item.metadata or {}).get("unique_id")
        if model_id in pr.nodes:
            flag, relations = _upstream(model_id, pr, counted, created, memo)
            item.metadata["changed_upstream"] = flag
            item.metadata["new_upstream_relations"] = sorted(pr.relations[n] for n in relations)
    return items


def _upstream(
    start: str, manifest: Manifest, counted: set[str], created: set[str], memo: dict[str, tuple[bool, frozenset]]
) -> tuple[bool, frozenset]:
    """(any ancestor in `counted`, ancestors in `created`) for `start`, itself excluded.

    One iterative, memoized depth-first walk; a cycle (dbt refuses them) or a parent missing
    from the manifest just stops the walk there.
    """
    if start in memo:
        return memo[start]
    acc: dict[str, list] = {start: [False, set()]}
    on_path = {start}
    stack = [(start, iter(_parents(manifest, start)))]
    while stack:
        node, parents = stack[-1]
        pushed = False
        for parent in parents:
            if parent == node or parent in on_path:
                continue  # self-loop or cycle
            if parent in counted:
                acc[node][0] = True
            if parent in created:
                acc[node][1].add(parent)
            if parent in memo:
                acc[node][0] = acc[node][0] or memo[parent][0]
                acc[node][1] |= memo[parent][1]
                continue
            acc[parent] = [False, set()]
            on_path.add(parent)
            stack.append((parent, iter(_parents(manifest, parent))))
            pushed = True
            break
        if pushed:
            continue
        stack.pop()
        on_path.discard(node)
        memo[node] = (acc[node][0], frozenset(acc[node][1]))
        if stack:
            child = stack[-1][0]
            acc[child][0] = acc[child][0] or memo[node][0]
            acc[child][1] |= memo[node][1]
    return memo[start]


def _parents(manifest: Manifest, node_id: str) -> list[str]:
    node = manifest.nodes.get(node_id)
    return node["parents"] if node else []


def _relation_parts(name: str) -> tuple[str, ...]:
    """`project`.`dataset`.`table`, "db"."schema"."t" or project:dataset.table -> parts, lowercased."""
    cleaned = name.replace("`", "").replace('"', "").strip().lower()
    if ":" in cleaned:
        cleaned = cleaned.replace(":", ".", 1)
    return tuple(part for part in cleaned.split(".") if part)


def relation_created(missing: dict, relations: list[str]) -> bool:
    """Whether a "Not found" object (`{"kind": table|view|dataset, "name"}`, parsed from the raw
    BigQuery error) is one of `relations` (or, for a dataset, holds one of them)."""
    wanted = _relation_parts(str(missing.get("name") or ""))
    if not wanted:
        return False
    for relation in relations:
        parts = _relation_parts(relation)
        if missing.get("kind") == "dataset":
            parts = parts[:-1]
        if len(parts) >= len(wanted) and parts[-len(wanted):] == wanted:
            return True
    return False


def _changed_in_git_or_checksum(
    model_id: str, repo_path: str | None, changed_paths: dict[str, str], pr: Manifest, prod: Manifest
) -> bool:
    if repo_path and repo_path in changed_paths:
        return True
    pr_sum = (pr.info.get(model_id) or {}).get("checksum")
    prod_sum = (prod.info.get(model_id) or {}).get("checksum")
    return bool(pr_sum and prod_sum and pr_sum != prod_sum)


def _unknown_model(model_id: str, info: dict, prod_model: dict | None, pr_model: dict | None, reason: str) -> ChangedSQL:
    return ChangedSQL(
        name=info["name"],
        path=info["path"],
        production_sql=prod_model["compiled_code"] if prod_model else None,
        pr_sql=pr_model["compiled_code"] if pr_model else None,
        adapter="dbt",
        metadata={"unique_id": model_id, "unknown_reason": reason},
    )


def _join(base: str, rel: str) -> str | None:
    """`rel` (relative to directory `base`) as a repo-relative path; None if it leaves the repo."""
    if not isinstance(rel, str) or not rel.strip() or "{{" in rel or rel.startswith("/"):
        return None
    joined = posixpath.normpath(posixpath.join(base or ".", rel.strip()))
    if joined == ".." or joined.startswith("../"):
        return None
    return "" if joined == "." else joined


def _under(path: str, directory: str) -> bool:
    return directory == "" or path == directory or path.startswith(directory + "/")


def _read_base_yaml(workspace: str, base_rev: str, path: str) -> dict:
    """A YAML mapping from the base commit, or {} when it is missing or unreadable."""
    if not base_rev:
        return {}
    text = _git_show(workspace, base_rev, path)
    if text is None:
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _path_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def project_layout(workspace: str, base_rev: str, project_dir: str) -> ProjectLayout:
    """The analyzed project's paths (from the base commit's `dbt_project.yml`) and its
    `local:` packages. Any problem reading them falls back to dbt's defaults."""
    try:
        return _project_layout(workspace, base_rev, project_dir)
    except Exception:  # defensive: a malformed project file must never stop the analysis
        return _project_layout(workspace, "", project_dir)


def _project_layout(workspace: str, base_rev: str, project_dir: str) -> ProjectLayout:
    project = _read_base_yaml(workspace, base_rev, _join(project_dir, "dbt_project.yml") or "dbt_project.yml")

    def under_project(paths) -> list[str]:
        return [d for d in (_join(project_dir, p) for p in paths) if d is not None]

    def dirs(keys, default) -> list[str]:
        configured = [p for k in keys for p in _path_list(project.get(k))]
        return _dedupe(under_project(configured + list(default)))

    other = [(_SEED_KEYS, _OTHER_PATH_KEYS["seed-paths"])] + [
        ((k,), d) for k, d in _OTHER_PATH_KEYS.items() if k not in _SEED_KEYS
    ]
    layout = ProjectLayout(
        project_dir=project_dir,
        model_dirs=dirs(_MODEL_PATH_KEYS, _MODEL_PATH_DEFAULT),
        config_dirs=_dedupe([d for keys, default in other for d in dirs(keys, default)]),
        name=project.get("name") if isinstance(project.get("name"), str) else None,
    )
    layout.covered_dirs = _dedupe(
        layout.model_dirs + layout.config_dirs
        + dirs(_TEST_PATH_KEYS, _TEST_PATH_DEFAULT)
        + dirs(("target-path",), ("target",))
    )

    for name in ("packages.yml", "dependencies.yml"):
        spec = _read_base_yaml(workspace, base_rev, _join(project_dir, name) or name)
        packages = spec.get("packages")
        for pkg in packages if isinstance(packages, list) else []:
            if not isinstance(pkg, dict) or "local" not in pkg:
                continue
            local = _join(project_dir, pkg.get("local"))
            # `local: ../` (an integration_tests project) holds the analyzed project itself:
            # treating it as a package would cover every file in the repository.
            if local is not None and not _under(project_dir, local):
                layout.package_dirs.append(local)
    return layout


def _non_model_changes(
    changed_paths: dict[str, str], pr: Manifest, layout: ProjectLayout, ignored: list[str]
) -> list[str]:
    """Changed files outside models that can change compiled SQL: project config, files in
    the macro/seed/snapshot/analysis paths and local packages, and yml under model paths."""
    project_files = {_join(layout.project_dir, f) for f in _PROJECT_FILES}
    model_paths = {i["repo_path"] for i in pr.info.values() if i["repo_path"]}
    model_file_dirs = {str(PurePosixPath(p).parent) for p in model_paths}
    found = []
    for path in sorted(changed_paths):
        if path in model_paths or any(_matches_glob(path, p) for p in ignored):
            continue
        is_yml = path.endswith((".yml", ".yaml"))
        if (
            path in project_files
            or any(_under(path, d) for d in layout.config_dirs + layout.package_dirs)
            or (is_yml and (
                any(_under(path, d) for d in layout.model_dirs)
                or str(PurePosixPath(path).parent) in model_file_dirs
            ))
        ):
            found.append(path)
    return found


def _examples(paths: list[str]) -> str:
    return ", ".join(paths[:3]) + (f" and {len(paths) - 3} more" if len(paths) > 3 else "")


def _non_model_item(others: list[str], consequence: str) -> ChangedSQL:
    return ChangedSQL(
        name=_NON_MODEL_ITEM,
        path=others[0],
        production_sql=None,
        pr_sql=None,
        adapter="dbt",
        metadata={
            "unique_id": _NON_MODEL_ITEM,
            "unknown_reason": (
                f"{len(others)} changed file(s) outside model files ({_examples(others)}) {consequence}"
            ),
        },
    )


def _outside_project_item(
    changed_paths: dict[str, str],
    layout: ProjectLayout,
    base_projects: list[str] | None,
    ignored: list[str],
    head_projects: list[str] | None = None,
) -> list[ChangedSQL]:
    """Changed `.sql` files the analyzed dbt project doesn't cover become one unknown item.

    DiffCost analyzes one dbt project per PR, so SQL elsewhere (another dbt project,
    loose query files, even inside the project's directory) is not analyzed; reporting
    it as "no changes" would let it pass the cost policy unseen. A file is covered when
    it is under the project's model/macro/seed/snapshot/analysis/test or target paths,
    or in a package directory, and not inside another dbt project (on the base commit
    or added by the PR). Deleted files and `ignored_paths` are skipped, and the comment
    shows `ignored_paths` entries that opt the files out without hiding the project.
    """
    project_dir = layout.project_dir
    others = [d for d in _dedupe((base_projects or []) + (head_projects or [])) if d != project_dir]
    nested = [d for d in others if _under(d, project_dir)]
    found = []
    for path, status in sorted(changed_paths.items()):
        if status == "D" or not path.endswith(".sql"):
            continue
        if INSTALLED_PACKAGE_DIRS & set(path.split("/")[:-1]):
            continue
        if any(_matches_glob(path, p) for p in ignored):
            continue
        if any(_under(path, d) for d in layout.package_dirs):
            continue  # a local package of this project: its changes show in compiled SQL
        if (
            _under(path, project_dir)
            and not any(_under(path, d) for d in nested)
            and any(_under(path, d) for d in layout.covered_dirs)
        ):
            continue
        found.append(path)
    if not found:
        return []

    protected = [project_dir] + layout.covered_dirs + layout.package_dirs

    def safe(directory: str) -> bool:
        """`<directory>/**` matches none of the analyzed project's files."""
        if not directory or (project_dir and _under(project_dir, directory)):
            return False
        return not any(_under(d, directory) or _under(directory, d) for d in protected if d)

    globs: list[str] = []
    for path in found:
        candidates = [d for d in sorted(others, key=len, reverse=True) if d and _under(path, d)]
        if "/" in path:
            candidates.append(path.split("/")[0])
        glob = next((f"{exempt_entry(d)}/**" for d in candidates if safe(d)), exempt_entry(path))
        if glob not in globs:
            globs.append(glob)
    where = f"{project_dir}/" if project_dir else "the repository root"
    return [ChangedSQL(
        name=_OUTSIDE_ITEM,
        path=found[0],
        production_sql=None,
        pr_sql=None,
        adapter="dbt",
        metadata={
            "unique_id": _OUTSIDE_ITEM,
            "unknown_reason": (
                f"{len(found)} changed .sql file(s) outside the dbt project in {where} ({_examples(found)}) "
                "were not analyzed, so their cost is unknown"
            ),
            "ignore_globs": globs,
        },
    )]


def _changed_without_baseline(
    changed_paths: dict[str, str], pr: Manifest, ignored: list[str], layout: ProjectLayout
) -> list[ChangedSQL]:
    """No production manifest: analyze only root-project models whose source file changed in
    this PR (a changed local package is part of the non-model item below).

    Files added in the PR are new models (no production cost). Modified models have
    no production SQL to compare against, so their production cost is unknown and
    they are excluded from totals. Changes outside model files (macros, yml, project
    config) can't be attributed to models without a baseline: they become one
    unknown item, so a threshold still blocks.
    """
    print(
        "::warning::No baseline dbt manifest found. Only models whose files changed in this PR "
        "are analyzed, and modified models show 'cost unavailable'. Compile the PR's base commit in "
        "this job and copy its `target/manifest.json` to `.diffcost-baseline/manifest.json`, or "
        "download a main-branch manifest artifact there "
        "(see https://diffcost.dev/docs/dbt#the-pull-request-workflow)."
    )

    results = []
    for model_id in sorted(pr.all_model_ids):
        info = pr.info[model_id]
        status = changed_paths.get(info["repo_path"]) if info["repo_path"] else None
        if status is None:
            continue
        if any(_matches_glob(info["path"], p) for p in ignored):
            continue
        model = pr.models.get(model_id)
        if model is None:
            results.append(_unknown_model(
                model_id, info, None, None, "the model is not compiled in the PR manifest (selective compile?)"
            ))
            continue
        results.append(
            ChangedSQL(
                name=model["name"],
                path=model["path"],
                production_sql=None,
                pr_sql=model["compiled_code"],
                adapter="dbt",
                metadata={
                    "unique_id": model_id,
                    # Existing model without a production baseline: cost before is unknown
                    "baseline_unknown": status != "A",
                },
            )
        )

    others = _non_model_changes(changed_paths, pr, layout, ignored)
    if others:
        results.append(_non_model_item(
            others,
            "can change compiled SQL; their cost can't be determined without a production baseline manifest",
        ))
    return results


def _model_changed(pr_model: dict, prod_model: dict, relation_map: dict[str, str]) -> bool:
    pr_checksum = pr_model.get("checksum")
    prod_checksum = prod_model.get("checksum")
    if pr_checksum and prod_checksum and pr_checksum != prod_checksum:
        return True
    pr_sql = _map_relations(pr_model["compiled_code"], relation_map)
    return _sql_hash(pr_sql) != _sql_hash(prod_model["compiled_code"])


def _map_relations(sql: str, relation_map: dict[str, str]) -> str:
    """Rewrite PR-target relation names to their production names (longest first)."""
    for pr_name in sorted(relation_map, key=len, reverse=True):
        sql = sql.replace(pr_name, relation_map[pr_name])
    return sql


def dbt_project_dir(workspace: str, manifest_path: str, base_projects: list[str] | None = None) -> str:
    """The dbt project's directory relative to the workspace ("" for the repo root).

    The nearest directory at or above the manifest that has a dbt_project.yml on the
    base commit (`base_projects`), or the base commit's only dbt project. Without a
    match there: the nearest such directory in the checkout, then the parent of the
    default `target/` directory. The caller checks that the manifest's model files
    exist under the result, so a wrong guess fails instead of matching nothing.
    """
    manifest = PurePosixPath(manifest_path)
    parents = ["" if str(p) == "." else str(p) for p in manifest.parents]
    if base_projects:
        for parent in parents:
            if parent in base_projects:
                return parent
        if len(base_projects) == 1:
            return base_projects[0]
    for parent in manifest.parents:
        if (Path(workspace) / parent / "dbt_project.yml").is_file():
            return "" if str(parent) == "." else str(parent)
    if manifest.parent.name == "target":
        parent = manifest.parent.parent
        return "" if str(parent) == "." else str(parent)
    return ""


def _parse_manifest(manifest_path: Path, project_dir: str = "", root_package: str | None = None) -> Manifest:
    """Parse a dbt manifest.json. Model paths are made repo-relative with `project_dir`.

    The root package is the manifest's `project_name`, else `root_package` (`name:` in the
    base commit's dbt_project.yml). With neither, every node counts as the root project's.
    """
    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise ManifestError(f"Could not read `{manifest_path.name}`: {str(e)[:120]}") from e
    if not isinstance(data, dict):
        raise ManifestError(f"`{manifest_path.name}` is not a dbt manifest.")

    manifest = Manifest()
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("project_name"), str):
        manifest.project_name = metadata["project_name"]
    root = manifest.project_name or root_package
    nodes = data.get("nodes") or {}
    if not isinstance(nodes, dict):
        return manifest

    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        resource_type = node.get("resource_type")
        relation = node.get("relation_name")
        if resource_type in _RELATION_TYPES and isinstance(relation, str) and relation:
            manifest.relations[node_id] = relation
        if resource_type not in _RELATION_TYPES:
            continue

        checksum = node.get("checksum")
        if isinstance(checksum, dict):
            checksum = checksum.get("checksum")
        rel_path = str(node.get("original_file_path") or node.get("path") or "")
        path = f"{project_dir}/{rel_path}" if project_dir and rel_path else rel_path
        package = node.get("package_name") if isinstance(node.get("package_name"), str) else None
        repo_path = path if path and (root is None or package in (None, root)) else None
        depends_on = node.get("depends_on") or {}
        parents = depends_on.get("nodes", []) if isinstance(depends_on, dict) else []
        parents = [p for p in parents if isinstance(p, str)] if isinstance(parents, list) else []
        cfg = node.get("config") if isinstance(node.get("config"), dict) else {}
        node_config = tuple(
            None if cfg.get(k) is None else str(cfg.get(k)) for k in ("materialized", "database", "schema", "alias")
        )
        compiled = node.get("compiled_code") or node.get("compiled_sql") or ""
        manifest.nodes[node_id] = {
            "repo_path": repo_path,
            "checksum": checksum if isinstance(checksum, str) else None,
            "parents": parents,
            "config": node_config,
            "ephemeral": node_config[0] == "ephemeral",
            "compiled": compiled if resource_type == "snapshot" and isinstance(compiled, str) else "",
        }
        if resource_type != "model":
            continue
        manifest.all_model_ids.add(node_id)

        info = {
            "name": str(node.get("name") or node_id.split(".")[-1]),
            "path": path,
            "repo_path": repo_path,
            "checksum": checksum if isinstance(checksum, str) else None,
        }
        manifest.info[node_id] = info

        if not compiled:
            continue

        manifest.models[node_id] = {
            **info,
            "compiled_code": compiled,
        }

    sources = data.get("sources") or {}
    if isinstance(sources, dict):
        for source_id, source in sources.items():
            relation = source.get("relation_name") if isinstance(source, dict) else None
            if isinstance(relation, str) and relation:
                manifest.relations[source_id] = relation

    return manifest


def _is_committed_manifest(workspace: str, rel_path: str) -> bool:
    """The PR manifest is a committed file that was not regenerated in this run.

    The PR author controls a committed manifest (stale or hand-edited compiled SQL), so
    it can't decide what changed. Symlinks are followed to the file they point at.
    """
    root = Path(workspace).resolve()
    try:
        real = (root / rel_path).resolve().relative_to(root).as_posix()
    except ValueError:
        return False  # outside the repository, so not a committed file
    return is_committed_copy(workspace, real)


def _is_plain_file(workspace: str, rel_path: str) -> bool:
    """True if `rel_path` is a regular file reached without following any symlink."""
    current = Path(workspace)
    for part in PurePosixPath(rel_path).parts:
        current = current / part
        if current.is_symlink():
            return False
    return current.is_file()


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _load_production_manifest(
    workspace: str,
    config: ActionConfig,
    project_dir: str,
    pr_manifest_path: Path,
    root_package: str | None,
) -> Manifest | None:
    """
    Load the production manifest. Try these sources in order:
    1. A file written during the run at .diffcost-baseline/manifest.json or
       target/prod-manifest.json: the base commit compiled in the job, or a downloaded
       artifact. A candidate that exists but is committed in the repository, reached
       through a symlink, unreadable or uncompiled is an error: the PR author could plant
       it to turn off change detection, so it is never skipped silently.
    2. The base branch, via `git show <base>:<manifest_path>` (if the manifest is committed)
    """
    for candidate in _BASELINE_CANDIDATES:
        p = Path(workspace) / candidate
        if not p.exists() and not p.is_symlink():
            continue
        if not _is_plain_file(workspace, candidate):
            raise ManifestError(
                f"The baseline manifest `{candidate}` is a symlink or inside a symlinked directory. "
                "The baseline must be a file written in this run (the base commit compiled in the job, "
                "or a downloaded artifact): remove the symlink from the repository."
            )
        if is_tracked(workspace, candidate):
            raise ManifestError(
                f"The baseline manifest `{candidate}` is committed in the repository, so a pull request could "
                "edit it. The baseline must be written in this run (the base commit compiled in the job, "
                "or a downloaded artifact): remove it from git."
            )
        if _same_file(p, pr_manifest_path):
            raise ManifestError(
                f"The baseline manifest `{candidate}` is the same file as `{config.manifest_path}`, so every model "
                "would compare equal. Compile the PR manifest to `target/manifest.json`, and write the "
                "baseline to `.diffcost-baseline/manifest.json` (compile the base commit there, or "
                "download a main-branch manifest artifact)."
            )
        try:
            manifest = _parse_manifest(p, project_dir, root_package)
        except ManifestError as e:
            raise ManifestError(f"Could not use the baseline manifest `{candidate}`: {e}") from e
        if not manifest.models:
            raise ManifestError(
                f"The baseline manifest `{candidate}` has no compiled SQL ({manifest.uncompiled} uncompiled "
                "model(s)). Write it with `dbt compile --target prod`, not `dbt parse`."
            )
        if manifest.uncompiled:
            print(f"::warning::{manifest.uncompiled} model(s) in the baseline manifest have no compiled SQL.")
        return manifest

    if not config.resolved_base_ref:
        return None

    try:
        result = run_git(workspace, ["show", f"{config.resolved_base_ref}:{config.manifest_path}"], timeout=30)
    except GitError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(result.stdout)
        manifest = _parse_manifest(Path(tmp_path), project_dir, root_package)
        return manifest if manifest.models else None
    except ManifestError:
        return None
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _sql_hash(sql: str) -> str:
    """Hash SQL for quick equality comparison. Normalizes whitespace."""
    normalized = " ".join(sql.split())
    return hashlib.sha256(normalized.encode()).hexdigest()
