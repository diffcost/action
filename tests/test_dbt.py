"""Tests for the dbt manifest adapter."""

import json
import shutil

import pytest

from adapters.dbt import ManifestError, dbt_project_dir, find_changed_dbt_models
from adapters.sql_files import BASE_MOVED_REASON
from gitutil import GitError
from config import ActionConfig


def _node(name, compiled, checksum=None, path=None, relation=None, resource_type="model"):
    node = {
        "resource_type": resource_type,
        "name": name,
        "original_file_path": path or f"models/{name}.sql",
        "depends_on": {"nodes": []},
    }
    if compiled is not None:
        node["compiled_code"] = compiled
    if checksum is not None:
        node["checksum"] = {"name": "sha256", "checksum": checksum}
    if relation is not None:
        node["relation_name"] = relation
    return node


def _manifest(*nodes):
    return json.dumps({"nodes": {f"model.proj.{n['name']}": n for n in nodes}})


def _config(repo, **overrides) -> ActionConfig:
    defaults = dict(
        sql_paths="",
        manifest_path="target/manifest.json",
        gcp_credentials="",
        gcp_project="",
        api_key="",
        config_path=".diffcost.yml",
        base_ref="main",
        cost_per_tb=6.25,
        comment_on_clean=False,
        workspace=str(repo.path),
        resolved_base_ref="origin/main",
        pr_number=1,
    )
    defaults.update(overrides)
    return ActionConfig(**defaults)


class TestWithBaseline:
    def test_target_specific_relation_names_do_not_count_as_changes(self, checkout_repo):
        """Same model compiled against CI vs prod schemas must not count as changed."""
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(
            _node("x", "select 1", checksum="x", relation="`p`.`ci_pr_12`.`x`"),
            _node("a", "select * from `p`.`ci_pr_12`.`x`", checksum="same", relation="`p`.`ci_pr_12`.`a`"),
            _node("b", "select * from `p`.`ci_pr_12`.`x` where dt > '2026'", checksum="b2"),
        ))
        repo.write(".diffcost-baseline/manifest.json", _manifest(
            _node("x", "select 1", checksum="x", relation="`p`.`prod`.`x`"),
            _node("a", "select * from `p`.`prod`.`x`", checksum="same", relation="`p`.`prod`.`a`"),
            _node("b", "select * from `p`.`prod`.`x`", checksum="b1"),
        ))

        changed = find_changed_dbt_models(_config(repo))

        assert [c.name for c in changed] == ["b"]
        assert changed[0].metadata["unique_id"] == "model.proj.b"

    def test_macro_only_change_is_detected(self, checkout_repo):
        """A macro edit changes compiled SQL but not the model file's checksum."""
        repo = checkout_repo({"models/a.sql": "select {{ window() }}"})
        repo.write("target/manifest.json", _manifest(
            _node("a", "select * from `p`.`prod`.`events`", checksum="same"),
            _node("b", "select 2", checksum="b"),
        ))
        repo.write(".diffcost-baseline/manifest.json", _manifest(
            _node("a", "select * from `p`.`prod`.`events` where event_date > '2026-01-01'", checksum="same"),
            _node("b", "select 2", checksum="b"),
        ))
        assert [c.name for c in find_changed_dbt_models(_config(repo))] == ["a"]

    def test_compiled_hash_fallback_without_checksum(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("a", "select 2")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select   1")))
        assert [c.name for c in find_changed_dbt_models(_config(repo))] == ["a"]

    def test_selective_compile_is_not_removed_or_new(self, checkout_repo, capsys):
        """`dbt compile -s state:modified+` leaves other nodes uncompiled: not 'Removed'."""
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("a", "select 2", checksum="a2"), _node("b", None)))
        repo.write(".diffcost-baseline/manifest.json", _manifest(
            _node("a", "select 1", checksum="a1"), _node("b", "select 3", checksum="b"),
        ))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["a"]
        assert "not compiled in one of the two manifests" in capsys.readouterr().out

    def test_changed_model_compiled_in_one_manifest_is_unknown(self, checkout_repo):
        """A selective baseline must not hide an edited model as 'no changes'."""
        repo = checkout_repo({"models/a.sql": "select 1", "models/b.sql": "select 3"})
        repo.write("models/b.sql", "select * from big")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a"),
                                                     _node("b", "select * from big", checksum="b2")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a"),
                                                                 _node("b", None, checksum="b1")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["b"]
        assert "not compiled in the baseline manifest" in changed[0].metadata["unknown_reason"]

    def test_checksum_change_of_uncompiled_model_is_unknown(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a"), _node("b", None, checksum="b2")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a"),
                                                                 _node("b", "select 3", checksum="b1")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["b"]
        assert "not compiled in the PR manifest" in changed[0].metadata["unknown_reason"]

    def test_stale_pr_manifest_is_unknown_not_clean(self, checkout_repo):
        """The model file changed in git but the manifest still matches prod: not recompiled."""
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("models/a.sql", "select * from big")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="old")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="old")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["a"]
        assert "not recompiled" in changed[0].metadata["unknown_reason"]

    def test_committed_baseline_fails_closed(self, checkout_repo):
        """A baseline committed in the PR is attacker-controlled. Skipping it would drop into
        no-baseline mode, where a macro-only change passes as 'no changes'."""
        forged = _manifest(_node("a", "select * from big", checksum="pr"))
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("macros/m.sql", "{% macro m() %}x{% endmacro %}")
        repo.write(".diffcost-baseline/manifest.json", "{}")  # the downloaded artifact lands on a tracked path
        repo.commit("plant baseline")
        repo.write("target/manifest.json", forged)  # recompiled in this run
        repo.write(".diffcost-baseline/manifest.json", forged)  # overwritten by download-artifact

        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "committed in the repository" in str(exc.value)
        assert "base commit compiled in the job" in str(exc.value) and "downloaded artifact" in str(exc.value)

    def test_symlinked_baseline_directory_fails_closed(self, checkout_repo):
        """`.diffcost-baseline -> evil/` passes `git ls-files .diffcost-baseline/manifest.json`."""
        forged = _manifest(_node("a", "select * from big", checksum="pr"))
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("models/a.sql", "select * from big")
        repo.write("evil/manifest.json", forged)
        (repo.path / ".diffcost-baseline").symlink_to("evil")
        repo.commit("forge baseline via symlink")
        repo.write("target/manifest.json", forged)

        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "symlink" in str(exc.value)
        assert "base commit compiled in the job" in str(exc.value) and "downloaded artifact" in str(exc.value)

    def test_unreadable_baseline_fails_closed(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a")))
        repo.write(".diffcost-baseline/manifest.json", "{not json")
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "baseline manifest" in str(exc.value)

    def test_uncompiled_baseline_fails_closed(self, checkout_repo):
        """A baseline written by `dbt parse` has no compiled SQL to compare against."""
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", None, checksum="a")))
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        message = str(exc.value)
        assert "The baseline manifest `.diffcost-baseline/manifest.json` has no compiled SQL (1 uncompiled" in message
        assert "Write it with `dbt compile --target prod`, not `dbt parse`." in message


class TestSubdirectoryProject:
    def test_project_in_subdirectory_matches_git_paths(self, checkout_repo):
        """original_file_path is relative to the dbt project, git paths to the repo."""
        repo = checkout_repo({"transform/dbt_project.yml": "name: proj", "transform/models/a.sql": "select 1"})
        repo.write("transform/models/a.sql", "select 1 -- edited")
        repo.commit()
        repo.write("transform/target/manifest.json", _manifest(_node("a", "select 1 -- edited")))

        changed = find_changed_dbt_models(_config(repo, manifest_path="transform/target/manifest.json"))

        assert [c.path for c in changed] == ["transform/models/a.sql"]
        assert changed[0].metadata["baseline_unknown"] is True

    def test_ignore_globs_use_repo_relative_paths(self, checkout_repo):
        repo = checkout_repo({"transform/dbt_project.yml": "name: proj", "transform/models/legacy/a.sql": "x"})
        repo.write("transform/models/legacy/a.sql", "select 2")
        repo.commit()
        repo.write("transform/target/manifest.json", _manifest(_node("a", "select 2", path="models/legacy/a.sql")))
        config = _config(repo, manifest_path="transform/target/manifest.json", ignored_paths=["transform/models/legacy/**"])
        assert find_changed_dbt_models(config) == []

    def test_project_dir_detection(self, tmp_path):
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "dbt_project.yml").write_text("name: p")
        assert dbt_project_dir(str(tmp_path), "a/b/manifest.json") == "a"
        assert dbt_project_dir(str(tmp_path), "target/manifest.json") == ""
        assert dbt_project_dir(str(tmp_path), "x/target/manifest.json") == "x"
        # The base commit's projects win over the checkout
        assert dbt_project_dir(str(tmp_path), "a/b/manifest.json", base_projects=["", "a/b"]) == "a/b"
        assert dbt_project_dir(str(tmp_path), "artifacts/manifest.json", base_projects=["transform"]) == "transform"

    def test_custom_target_path_in_subdirectory(self, checkout_repo):
        """`--target-path artifacts` in transform/: model paths still get the transform/ prefix."""
        repo = checkout_repo({"transform/dbt_project.yml": "name: proj", "transform/models/a.sql": "select 1"})
        repo.write("transform/models/a.sql", "select 2")
        repo.commit()
        repo.write("artifacts/manifest.json", _manifest(_node("a", "select 2")))
        changed = find_changed_dbt_models(_config(repo, manifest_path="artifacts/manifest.json"))
        assert [c.path for c in changed] == ["transform/models/a.sql"]

    def test_pr_cannot_move_the_project_dir(self, checkout_repo):
        """A PR that commits target/dbt_project.yml must not re-prefix every model path."""
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/a.sql": "select 1"})
        repo.write("models/a.sql", "select 2")
        repo.write("target/dbt_project.yml", "name: decoy")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 2")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.path for c in changed] == ["models/a.sql"]

    def test_manifest_models_missing_from_checkout_raise(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("zzz", "select 2", path="elsewhere/zzz.sql")))
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "project" in str(exc.value) and "manifest_path" in str(exc.value)


class TestWithoutBaseline:
    def test_only_models_changed_in_git_are_analyzed(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1", "models/b.sql": "select 2"})
        repo.write("models/a.sql", "select 1 -- edited")
        repo.write("models/c.sql", "select 3")
        repo.commit()
        repo.write("target/manifest.json", _manifest(
            _node("a", "select 1 -- edited"), _node("b", "select 2"), _node("c", "select 3"),
        ))

        changed = {c.name: c for c in find_changed_dbt_models(_config(repo))}

        assert set(changed) == {"a", "c"}  # b is untouched: not listed as "New"
        assert changed["a"].metadata["baseline_unknown"] is True  # existed before: cost unknown
        assert changed["c"].metadata["baseline_unknown"] is False  # added in this PR: New
        assert changed["c"].production_sql is None

    def test_macro_change_without_baseline_is_unknown_not_clean(self, checkout_repo):
        """Without a baseline a macro edit can't be attributed to models: one unknown item."""
        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select {{ m() }}",
                              "macros/m.sql": "{% macro m() %}1{% endmacro %}"})
        repo.write("macros/m.sql", "{% macro m() %}* from big{% endmacro %}")
        repo.write("models/schema.yml", "version: 2")
        repo.write("README.md", "docs")  # not a dbt file
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1")))

        changed = find_changed_dbt_models(_config(repo))

        assert [c.name for c in changed] == ["non-model dbt files"]
        reason = changed[0].metadata["unknown_reason"]
        assert "2 changed file(s)" in reason and "macros/m.sql" in reason and "README" not in reason

    @pytest.mark.parametrize("path", [
        "tests/props.yml",  # dbt reads YAML under test-paths as properties (model config too)
        "package-lock.yml",  # pins the package versions
    ])
    def test_test_path_yaml_and_the_package_lock_are_non_model_files(self, checkout_repo, path):
        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select 1", path: "v1"})
        repo.write(path, "v2")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1")))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "non-model dbt files" and path in item.metadata["unknown_reason"]

    def test_changed_uncompiled_model_without_baseline_is_unknown(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1", "models/b.sql": "select 2"})
        repo.write("models/b.sql", "select * from big")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1"), _node("b", None)))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["b"]
        assert "not compiled in the PR manifest" in changed[0].metadata["unknown_reason"]


    def test_warning_names_both_ways_to_write_the_baseline(self, checkout_repo, capsys):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("models/a.sql", "select 2")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 2")))

        find_changed_dbt_models(_config(repo))

        out = capsys.readouterr().out
        assert "No baseline dbt manifest found" in out
        assert "base commit" in out and ".diffcost-baseline/manifest.json" in out and "artifact" in out
        assert "https://diffcost.dev/docs/dbt#the-pull-request-workflow" in out


class TestBaselineCompiledInTheJob:
    """The documented baseline step: `git worktree add` of the base commit outside the
    workspace, `dbt compile` there, and its manifest copied to .diffcost-baseline/."""

    def test_baseline_from_a_worktree_of_the_base_commit(self, checkout_repo, tmp_path):
        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select {{ m() }}",
                              "macros/m.sql": "{% macro m() %}1{% endmacro %}"})
        base = repo.git("rev-parse", "HEAD").strip()
        repo.write("macros/m.sql", "{% macro m() %}* from big{% endmacro %}")
        repo.commit()
        worktree = tmp_path / "runner_temp" / "base"
        repo.git("worktree", "add", "--force", "--detach", str(worktree), base)
        (worktree / "target").mkdir()
        (worktree / "target" / "manifest.json").write_text(_manifest(_node("a", "select 1", checksum="same")))
        repo.write(".diffcost-baseline/manifest.json", (worktree / "target" / "manifest.json").read_text())
        repo.write("target/manifest.json", _manifest(_node("a", "select * from big", checksum="same")))
        # The Action's container doesn't mount the runner's temp directory: the worktree stays
        # registered in .git but its directory is missing.
        shutil.rmtree(worktree)

        changed = find_changed_dbt_models(_config(repo))

        assert [c.name for c in changed] == ["a"]  # the macro change, attributed to its model
        assert changed[0].production_sql == "select 1"
        assert changed[0].pr_sql == "select * from big"


class TestManifestErrors:
    def test_missing_manifest_raises(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        with pytest.raises(ManifestError):
            find_changed_dbt_models(_config(repo))

    def test_parse_only_manifest_raises(self, checkout_repo):
        """`dbt parse` leaves compiled_code empty; that must not look like 'every model removed'."""
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("a", None), _node("b", None)))
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "dbt compile" in str(exc.value)

    def test_malformed_manifest_raises(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", "{not json")
        with pytest.raises(ManifestError):
            find_changed_dbt_models(_config(repo))

    def test_empty_manifest_raises(self, checkout_repo):
        """A `{}` manifest must not turn into 'no SQL changes' and a passing status."""
        repo = checkout_repo({"queries/a.sql": "select 1"})
        repo.write("queries/a.sql", "select * from big")
        repo.commit()
        repo.write("target/manifest.json", "{}")
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "no dbt models" in str(exc.value)

    def test_unresolved_base_raises_even_with_a_baseline(self, checkout_repo):
        """Without the base commit, .diffcost.yml thresholds were silently dropped."""
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("a", "select 2", checksum="a2")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a1")))
        with pytest.raises(GitError):
            find_changed_dbt_models(_config(repo, resolved_base_ref="", git_error="Base branch `main` was not found"))


class TestCommittedPrManifest:
    """The PR author controls a committed manifest, so it must not decide what changed."""

    def test_committed_manifest_not_recompiled_raises(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("macros/m.sql", "{% macro m() %}select * from big{% endmacro %}")
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a1")))
        repo.commit("stale manifest")
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a1")))
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "not one compiled in this run" in str(exc.value)

    def test_committed_manifest_recompiled_in_run_is_used(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a1")))
        repo.commit("manifest committed by habit")
        repo.write("target/manifest.json", _manifest(_node("a", "select 2", checksum="a1")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a1")))
        assert [c.name for c in find_changed_dbt_models(_config(repo))] == ["a"]

    def test_symlink_to_committed_manifest_raises(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("fake/manifest.json", _manifest(_node("a", "select 1", checksum="a1")))
        (repo.path / "target").symlink_to("fake")
        repo.commit("symlinked target")
        with pytest.raises(ManifestError):
            find_changed_dbt_models(_config(repo))


class TestSymlinkedPrManifest:
    """A committed symlink must not make the PR manifest and the baseline one file."""

    def _plant(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select 1",
                              "macros/m.sql": "{% macro m() %}1{% endmacro %}"})
        repo.write("macros/m.sql", "{% macro m() %}* from big{% endmacro %}")  # the real change
        repo.write(".diffcost-baseline/.keep", "")
        (repo.path / "target").mkdir()
        (repo.path / "target" / "manifest.json").symlink_to("../.diffcost-baseline/manifest.json")
        repo.commit("symlink the PR manifest to the baseline")
        return repo

    def test_compile_then_download_fails_closed(self, checkout_repo):
        repo = self._plant(checkout_repo)
        (repo.path / "target" / "manifest.json").write_text(_manifest(_node("a", "select * from big")))  # dbt
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1")))  # download-artifact
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "symlink" in str(exc.value)

    def test_download_then_compile_fails_closed(self, checkout_repo):
        repo = self._plant(checkout_repo)
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1")))  # download-artifact
        (repo.path / "target" / "manifest.json").write_text(_manifest(_node("a", "select * from big")))  # dbt
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo))
        assert "symlink" in str(exc.value)

    def test_manifest_path_pointing_at_the_baseline_fails_closed(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1")))
        with pytest.raises(ManifestError) as exc:
            find_changed_dbt_models(_config(repo, manifest_path=".diffcost-baseline/manifest.json"))
        assert "same file" in str(exc.value)
        assert "compile the base commit" in str(exc.value) and "artifact" in str(exc.value)


class TestProjectLayout:
    """Non-model changes come from the project's configured paths."""

    def test_custom_macro_path_without_baseline_is_unknown(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: p\nmacro-paths: ['dbt_macros']",
                              "models/a.sql": "select {{ f() }}",
                              "dbt_macros/filters.sql": "{% macro f() %}1 where d > 0{% endmacro %}"})
        repo.write("dbt_macros/filters.sql", "{% macro f() %}* from big{% endmacro %}")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["non-model dbt files"]
        assert "dbt_macros/filters.sql" in changed[0].metadata["unknown_reason"]

    def test_yml_under_a_custom_model_path_is_a_non_model_change(self, checkout_repo):
        repo = checkout_repo({"transform/dbt_project.yml": "name: p\nmodel-paths: ['src']",
                              "transform/src/marts/a.sql": "select 1"})
        repo.write("transform/src/sources.yml", "version: 2")  # no model file beside it
        repo.commit()
        repo.write("transform/target/manifest.json", _manifest(_node("a", "select 1", path="src/marts/a.sql")))
        changed = find_changed_dbt_models(_config(repo, manifest_path="transform/target/manifest.json"))
        assert [c.name for c in changed] == ["non-model dbt files"]
        assert "transform/src/sources.yml" in changed[0].metadata["unknown_reason"]

    def test_local_package_outside_the_project_is_a_non_model_change(self, checkout_repo):
        repo = checkout_repo({"transform/dbt_project.yml": "name: p",
                              "transform/packages.yml": "packages:\n  - local: ../shared",
                              "transform/models/a.sql": "select {{ shared() }}",
                              "shared/macros/s.sql": "{% macro shared() %}1{% endmacro %}"})
        repo.write("shared/macros/s.sql", "{% macro shared() %}* from big{% endmacro %}")
        repo.commit()
        repo.write("transform/target/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo, manifest_path="transform/target/manifest.json"))
        assert [c.name for c in changed] == ["non-model dbt files"]
        assert "shared/macros/s.sql" in changed[0].metadata["unknown_reason"]

    def test_unreadable_configured_paths_keep_the_defaults(self, checkout_repo):
        """An env_var() template or `..` path must not drop dbt's default directory."""
        repo = checkout_repo({"dbt_project.yml": "name: p\nmacro-paths: [\"{{ env_var('M') }}\", '../x']",
                              "models/a.sql": "select {{ m() }}", "macros/m.sql": "{% macro m() %}1{% endmacro %}"})
        repo.write("macros/m.sql", "{% macro m() %}* from big{% endmacro %}")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["non-model dbt files"]
        assert "macros/m.sql" in changed[0].metadata["unknown_reason"]

    def test_malformed_packages_file_falls_back_to_defaults(self, checkout_repo):
        """A malformed `packages: 5` must not crash the run (the comment would be lost)."""
        repo = checkout_repo({"dbt_project.yml": "name: p", "packages.yml": "packages: 5",
                              "models/a.sql": "select 1", "macros/m.sql": "x"})
        repo.write("macros/m.sql", "y")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1")))
        assert [c.name for c in find_changed_dbt_models(_config(repo))] == ["non-model dbt files"]

    def test_local_package_holding_the_project_is_ignored(self, checkout_repo):
        """A `local: ../` package (integration_tests layout) must not cover the whole repository."""
        repo = checkout_repo({"dbt_project.yml": "name: pkg", "macros/m.sql": "x",
                              "integration_tests/dbt_project.yml": "name: it",
                              "integration_tests/packages.yml": "packages:\n  - local: ../",
                              "integration_tests/models/a.sql": "select 1", "queries/q.sql": "select 1"})
        repo.write("queries/q.sql", "select 2")
        repo.commit()
        repo.write("integration_tests/target/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo, manifest_path="integration_tests/target/manifest.json"))
        assert [c.name for c in changed] == ["SQL outside the analyzed dbt project"]
        assert changed[0].metadata["ignore_globs"] == ["queries/**"]


class TestOutsideProject:
    """Changed .sql outside the analyzed dbt project is unknown cost."""

    def test_sql_outside_a_subdirectory_project_is_one_unknown_item(self, checkout_repo):
        repo = checkout_repo({"transform/dbt_project.yml": "name: p", "transform/models/a.sql": "select 1",
                              "marketing/dbt_project.yml": "name: m", "marketing/models/x.sql": "select 1",
                              "queries/q.sql": "select 1", "old/gone.sql": "select 1"})
        repo.write("marketing/models/x.sql", "select * from big")
        repo.write("queries/q.sql", "select * from big")
        repo.write("README.md", "docs")
        (repo.path / "old" / "gone.sql").unlink()  # deletions can't add cost
        repo.commit()
        repo.write("transform/target/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo, manifest_path="transform/target/manifest.json"))
        assert [c.name for c in changed] == ["SQL outside the analyzed dbt project"]
        meta = changed[0].metadata
        assert "2 changed .sql file(s)" in meta["unknown_reason"]
        assert "marketing/models/x.sql" in meta["unknown_reason"] and "gone" not in meta["unknown_reason"]
        assert meta["ignore_globs"] == ["marketing/**", "queries/**"]

    def test_ignored_paths_opt_out(self, checkout_repo):
        repo = checkout_repo({"transform/dbt_project.yml": "name: p", "transform/models/a.sql": "select 1",
                              "queries/q.sql": "select 1"})
        repo.write("queries/q.sql", "select 2")
        repo.commit()
        repo.write("transform/target/manifest.json", _manifest(_node("a", "select 1")))
        config = _config(repo, manifest_path="transform/target/manifest.json", ignored_paths=["queries/**"])
        assert find_changed_dbt_models(config) == []

    def test_nested_project_under_a_root_project_is_outside(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select 1",
                              "integration/dbt_project.yml": "name: i", "integration/models/t.sql": "select 1",
                              "scripts/backfill.sql": "select 1"})
        repo.write("integration/models/t.sql", "select 2")
        # Inside the root project's directory but under none of its dbt paths
        repo.write("scripts/backfill.sql", "select 2")
        repo.write("tests/assert_x.sql", "select 1")  # a dbt test: covered
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["SQL outside the analyzed dbt project"]
        assert "2 changed .sql file(s)" in changed[0].metadata["unknown_reason"]
        assert changed[0].metadata["ignore_globs"] == ["integration/**", "scripts/**"]

    def test_project_added_by_the_pr_in_a_root_project_is_outside(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select 1"})
        repo.write("newproj/dbt_project.yml", "name: n")  # added by this PR
        repo.write("newproj/models/x.sql", "select * from big")
        repo.write("models/legacy/dbt_project.yml", "name: l")  # a project inside the model path
        repo.write("models/legacy/q.sql", "select * from big")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["SQL outside the analyzed dbt project"]
        # A glob under `models/` could hide the analyzed project's models: the file's own path instead
        assert changed[0].metadata["ignore_globs"] == ["models/legacy/q.sql", "newproj/**"]

    def test_suggested_path_is_escaped(self, checkout_repo):
        """The file's own path, as suggested, matches only that file."""
        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select 1"})
        repo.write("models/legacy/dbt_project.yml", "name: l")
        repo.write("models/legacy/q[1],x.sql", "select * from big")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo))
        assert changed[0].metadata["ignore_globs"] == ["models/legacy/q[[]1][,]x.sql"]

    def test_hint_never_matches_the_analyzed_project(self, checkout_repo):
        """Analyzed dbt/core, edited dbt/marketing: `dbt/**` would hide every model."""
        repo = checkout_repo({"dbt/core/dbt_project.yml": "name: c", "dbt/core/models/a.sql": "select 1",
                              "dbt/marketing/dbt_project.yml": "name: m", "dbt/marketing/models/x.sql": "x",
                              "dbt/loose.sql": "select 1"})
        repo.write("dbt/marketing/models/x.sql", "select * from big")
        repo.write("dbt/loose.sql", "select 2")
        repo.commit()
        repo.write("dbt/core/target/manifest.json", _manifest(_node("a", "select 1")))
        changed = find_changed_dbt_models(_config(repo, manifest_path="dbt/core/target/manifest.json"))
        globs = changed[-1].metadata["ignore_globs"]
        assert globs == ["dbt/loose.sql", "dbt/marketing/**"]
        assert not any(g.startswith("dbt/**") or g == "dbt/core/**" for g in globs)


class TestSelectiveCompile:
    def test_macro_change_with_uncompiled_pr_models_is_unknown(self, checkout_repo):
        """A model the PR didn't compile can't show the macro's effect."""
        repo = checkout_repo({"dbt_project.yml": "name: p", "models/a.sql": "select {{ m() }}",
                              "models/b.sql": "select 2", "macros/m.sql": "{% macro m() %}1{% endmacro %}"})
        repo.write("macros/m.sql", "{% macro m() %}* from big{% endmacro %}")
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", None, checksum="a"), _node("b", "select 2", checksum="b")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a"),
                                                                 _node("b", "select 2", checksum="b")))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["non-model dbt files"]
        assert "not compiled in both manifests" in changed[0].metadata["unknown_reason"]

    def test_deleting_a_model_the_baseline_did_not_compile_is_not_unknown(self, checkout_repo):
        """Deleting a model can't add cost."""
        repo = checkout_repo({"models/a.sql": "select 1", "models/x.sql": "select 3"})
        (repo.path / "models" / "x.sql").unlink()
        repo.commit()
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a")))
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a"),
                                                                 _node("x", None, checksum="x")))
        assert find_changed_dbt_models(_config(repo)) == []


def _graph(*nodes):
    """A manifest from (unique_id, node, parents) tuples."""
    out = {}
    for unique_id, node, parents in nodes:
        node["depends_on"] = {"nodes": list(parents)}
        out[unique_id] = node
    return json.dumps({"nodes": out})


class TestChangedUpstream:
    """A PR-side "Unrecognized name" / "Not found" is exempt only
    when something upstream of the model is new or changed in this PR (`changed_upstream`)."""

    def _changed(self, repo):
        return {c.name: c.metadata for c in find_changed_dbt_models(_config(repo))}

    def test_new_model_reading_only_unchanged_models_has_no_changed_upstream(self, checkout_repo):
        """A column typo in a new model is not explained by this PR."""
        repo = checkout_repo({"models/stg.sql": "select 1"})
        repo.write("models/fct_new.sql", "select col_typo from {{ ref('stg') }}")
        repo.commit()
        stg = ("model.proj.stg", _node("stg", "select 1", checksum="s"), [])
        repo.write("target/manifest.json", _graph(
            stg, ("model.proj.fct_new", _node("fct_new", "select col_typo from stg", checksum="f"), ["model.proj.stg"]),
        ))
        repo.write(".diffcost-baseline/manifest.json", _graph(("model.proj.stg", _node("stg", "select 1", checksum="s"), [])))
        assert self._changed(repo)["fct_new"]["changed_upstream"] is False

    def test_new_model_reading_a_new_model_or_seed(self, checkout_repo):
        repo = checkout_repo({"models/stg.sql": "select 1"})
        repo.write("models/stg_new.sql", "select 2")
        repo.write("models/fct_new.sql", "select * from {{ ref('stg_new') }}")
        repo.write("models/fct_rates.sql", "select * from {{ ref('rates') }}")
        repo.write("seeds/rates.csv", "a,b\n1,2\n")
        repo.commit()
        repo.write("target/manifest.json", _graph(
            ("model.proj.stg", _node("stg", "select 1", checksum="s"), []),
            ("model.proj.stg_new", _node("stg_new", "select 2", checksum="n"), []),
            ("model.proj.fct_new", _node("fct_new", "select * from stg_new", checksum="f"), ["model.proj.stg_new"]),
            ("seed.proj.rates", _node("rates", None, checksum="r", path="seeds/rates.csv", resource_type="seed"), []),
            ("model.proj.fct_rates", _node("fct_rates", "select * from rates", checksum="fr"), ["seed.proj.rates"]),
        ))
        repo.write(".diffcost-baseline/manifest.json", _graph(("model.proj.stg", _node("stg", "select 1", checksum="s"), [])))
        changed = self._changed(repo)
        assert changed["fct_new"]["changed_upstream"] is True
        assert changed["fct_rates"]["changed_upstream"] is True
        assert changed["stg_new"]["changed_upstream"] is False  # nothing upstream of it

    def test_changed_upstream_through_an_unchanged_model(self, checkout_repo):
        """a (changed) -> b (unchanged) -> c (changed): c reads a column a adds, through b."""
        repo = checkout_repo({"models/a.sql": "select 1", "models/b.sql": "select * from a", "models/c.sql": "select x"})
        repo.write("models/a.sql", "select 1, 2 as discount")
        repo.write("models/c.sql", "select discount from b")
        repo.commit()
        repo.write("target/manifest.json", _graph(
            ("model.proj.a", _node("a", "select 1, 2 as discount", checksum="a2"), []),
            ("model.proj.b", _node("b", "select * from a", checksum="b"), ["model.proj.a"]),
            ("model.proj.c", _node("c", "select discount from b", checksum="c2"), ["model.proj.b"]),
        ))
        repo.write(".diffcost-baseline/manifest.json", _graph(
            ("model.proj.a", _node("a", "select 1", checksum="a1"), []),
            ("model.proj.b", _node("b", "select * from a", checksum="b"), ["model.proj.a"]),
            ("model.proj.c", _node("c", "select x", checksum="c1"), ["model.proj.b"]),
        ))
        changed = self._changed(repo)
        assert set(changed) == {"a", "c"}
        assert changed["c"]["changed_upstream"] is True
        assert changed["a"]["changed_upstream"] is False

    def test_changed_seed_checksum_counts(self, checkout_repo):
        repo = checkout_repo({"models/m.sql": "select a from rates"})
        repo.write("models/m.sql", "select a, b from rates")
        repo.commit()
        seed = ("seed.proj.rates", _node("rates", None, checksum="r2", path="seeds/rates.csv", resource_type="seed"), [])
        repo.write("target/manifest.json", _graph(
            seed, ("model.proj.m", _node("m", "select a, b from rates", checksum="m2"), ["seed.proj.rates"]),
        ))
        repo.write(".diffcost-baseline/manifest.json", _graph(
            ("seed.proj.rates", _node("rates", None, checksum="r1", path="seeds/rates.csv", resource_type="seed"), []),
            ("model.proj.m", _node("m", "select a from rates", checksum="m1"), ["seed.proj.rates"]),
        ))
        assert self._changed(repo)["m"]["changed_upstream"] is True

    def test_without_baseline_only_git_changes_count(self, checkout_repo):
        repo = checkout_repo({"models/stg.sql": "select 1"})
        repo.write("models/stg_new.sql", "select 2")
        repo.write("models/fct_new.sql", "select * from {{ ref('stg_new') }}")
        repo.write("models/fct_typo.sql", "select col_typo from {{ ref('stg') }}")
        repo.commit()
        repo.write("target/manifest.json", _graph(
            ("model.proj.stg", _node("stg", "select 1"), []),
            ("model.proj.stg_new", _node("stg_new", "select 2"), []),
            ("model.proj.fct_new", _node("fct_new", "select * from stg_new"), ["model.proj.stg_new"]),
            ("model.proj.fct_typo", _node("fct_typo", "select col_typo from stg"), ["model.proj.stg"]),
        ))
        changed = self._changed(repo)
        assert changed["fct_new"]["changed_upstream"] is True
        assert changed["fct_typo"]["changed_upstream"] is False


def _with(node, package=None, **config):
    if config:
        node["config"] = config
    if package:
        node["package_name"] = package
    return node


class TestChangedUpstreamEdges:
    """What counts as a changed upstream, and the walk's edge cases."""

    def _changed(self, repo, **overrides):
        return {c.name: c.metadata for c in find_changed_dbt_models(_config(repo, **overrides))}

    def _base_and_pr(self, repo, base_nodes, pr_nodes):
        repo.write("target/manifest.json", _graph(*pr_nodes))
        repo.write(".diffcost-baseline/manifest.json", _graph(*base_nodes))

    def _new_x(self, repo, parents):
        """A new model x (committed) reading `parents`."""
        repo.write("models/x.sql", "select 1")
        repo.commit()
        return ("model.proj.x", _node("x", "select new from p", checksum="x"), parents)

    def test_target_specific_relation_names_alone_are_not_a_change(self, checkout_repo):
        """CI-target relations: every relation name differs, nothing else does."""
        repo = checkout_repo({"models/a.sql": "select 1"})
        x = self._new_x(repo, ["model.proj.a"])
        self._base_and_pr(
            repo,
            [("model.proj.a", _node("a", "select 1", checksum="a", relation="`p`.`prod`.`a`"), [])],
            [("model.proj.a", _node("a", "select 1", checksum="a", relation="`p`.`ci_pr_12`.`a`"), []), x],
        )
        meta = self._changed(repo)["x"]
        assert meta["changed_upstream"] is False
        assert meta["new_upstream_relations"] == []

    def test_cycles_self_loops_missing_and_malformed_parents(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1", "models/b.sql": "select 2", "models/c.sql": "select 3",
                              "models/d.sql": "select 4", "models/n.sql": "select 5"})
        x = self._new_x(repo, ["model.proj.x", "model.proj.a", "model.proj.n", "model.proj.gone", "source.proj.raw.t"])
        pr_nodes = [
            ("model.proj.a", _node("a", "select 1", checksum="a"), ["model.proj.b"]),
            ("model.proj.b", _node("b", "select 2", checksum="b"), ["model.proj.a", "model.proj.c"]),
            ("model.proj.c", _node("c", "select 3", checksum="c"), []),
            ("model.proj.d", _node("d", "select 4", checksum="d"), []),
            ("model.proj.n", _node("n", "select 5", checksum="n"), []),
            x,
        ]
        manifest = json.loads(_graph(*pr_nodes))
        manifest["nodes"]["model.proj.c"]["depends_on"] = "model.proj.d"  # malformed: a string
        manifest["nodes"]["model.proj.d"]["depends_on"] = {"nodes": {"model.proj.a": 1}}  # malformed: a dict
        manifest["nodes"]["model.proj.b"]["depends_on"] = {"nodes": ["model.proj.a", None, 3, "model.proj.c"]}
        manifest["nodes"]["model.proj.n"]["depends_on"] = None
        repo.write("target/manifest.json", json.dumps(manifest))
        repo.write(".diffcost-baseline/manifest.json", _graph(*[n for n in pr_nodes if n[0] != "model.proj.x"]))
        assert self._changed(repo)["x"]["changed_upstream"] is False  # x itself (new) doesn't count

        manifest["nodes"]["model.proj.c"]["checksum"] = {"name": "sha256", "checksum": "c2"}  # c changes
        repo.write("target/manifest.json", json.dumps(manifest))
        assert self._changed(repo)["x"]["changed_upstream"] is True  # x -> a -> b -> c, through the a/b cycle

    def test_macro_only_upstream_change_counts(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select {{ m() }}"})
        x = self._new_x(repo, ["model.proj.a"])
        self._base_and_pr(
            repo,
            [("model.proj.a", _node("a", "select 1", checksum="same"), [])],
            [("model.proj.a", _node("a", "select 1, 2 as discount", checksum="same"), []), x],
        )
        assert self._changed(repo)["x"]["changed_upstream"] is True

    def test_upstream_under_ignored_paths_counts(self, checkout_repo):
        repo = checkout_repo({"models/legacy/a.sql": "select 1"})
        x = self._new_x(repo, ["model.proj.a"])
        self._base_and_pr(
            repo,
            [("model.proj.a", _node("a", "select 1", checksum="a1", path="models/legacy/a.sql"), [])],
            [("model.proj.a", _node("a", "select 2", checksum="a2", path="models/legacy/a.sql"), []), x],
        )
        changed = self._changed(repo, ignored_paths=["models/legacy/**"])
        assert "a" not in changed
        assert changed["x"]["changed_upstream"] is True

    def test_ephemeral_ancestor_does_not_count_but_its_parents_do(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1", "models/e.sql": "select * from a"})
        x = self._new_x(repo, ["model.proj.e"])
        e_base = ("model.proj.e", _with(_node("e", "select * from a", checksum="e1"), materialized="ephemeral"),
                  ["model.proj.a"])
        e_pr = ("model.proj.e", _with(_node("e", "select *, 1 from a", checksum="e2"), materialized="ephemeral"),
                ["model.proj.a"])
        a = ("model.proj.a", _node("a", "select 1", checksum="a"), [])
        self._base_and_pr(repo, [a, e_base], [a, e_pr, x])
        assert self._changed(repo)["x"]["changed_upstream"] is False  # only the ephemeral model changed

        a2 = ("model.proj.a", _node("a", "select 1, 2", checksum="a2"), [])
        self._base_and_pr(repo, [a, e_base], [a2, e_pr, x])
        assert self._changed(repo)["x"]["changed_upstream"] is True  # through the ephemeral model to a

    def test_config_only_materialization_creates_a_relation(self, checkout_repo):
        """Ephemeral e becomes a table via yml: same file and SQL, but e now creates an object."""
        repo = checkout_repo({"models/e.sql": "select 1"})
        x = self._new_x(repo, ["model.proj.e"])
        self._base_and_pr(
            repo,
            [("model.proj.e", _with(_node("e", "select 1", checksum="e"), materialized="ephemeral"), [])],
            [("model.proj.e", _with(_node("e", "select 1", checksum="e", relation="`p`.`prod`.`e`"),
                                    materialized="table"), []), x],
        )
        meta = self._changed(repo)["x"]
        assert meta["changed_upstream"] is True
        assert meta["new_upstream_relations"] == ["`p`.`prod`.`e`"]

    def test_schema_config_change_moves_a_relation(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        x = self._new_x(repo, ["model.proj.a"])
        self._base_and_pr(
            repo,
            [("model.proj.a", _with(_node("a", "select 1", checksum="a", relation="`p`.`prod`.`a`"),
                                    materialized="table"), [])],
            [("model.proj.a", _with(_node("a", "select 1", checksum="a", relation="`p`.`prod_marts`.`a`"),
                                    materialized="table", schema="marts"), []), x],
        )
        assert self._changed(repo)["x"]["new_upstream_relations"] == ["`p`.`prod_marts`.`a`"]

    def test_package_node_paths_never_match_root_git_changes(self, checkout_repo):
        """A package model's path is relative to the package, not to the root project."""
        repo = checkout_repo({"models/stg.sql": "select 1"})
        repo.write("models/stg.sql", "select 2")  # the root project's file, same relative path
        repo.write("models/x.sql", "select 1")
        repo.commit()
        manifest = json.loads(_graph(
            ("model.utils.stg", _with(_node("stg", "select 1"), package="utils"), []),
            ("model.proj.x", _with(_node("x", "select new from stg"), package="proj"), ["model.utils.stg"]),
        ))
        manifest["metadata"] = {"project_name": "proj"}
        repo.write("target/manifest.json", json.dumps(manifest))
        assert self._changed(repo)["x"]["changed_upstream"] is False

    def test_snapshot_whose_compiled_sql_changed_counts(self, checkout_repo):
        repo = checkout_repo({"snapshots/s.sql": "select 1", "models/m.sql": "select 1"})
        x = self._new_x(repo, ["snapshot.proj.s"])
        snap = {"path": "snapshots/s.sql", "resource_type": "snapshot", "checksum": "s"}
        m = ("model.proj.m", _node("m", "select 1", checksum="m"), [])
        self._base_and_pr(
            repo,
            [m, ("snapshot.proj.s", _node("s", "select 1", **snap), [])],
            [m, ("snapshot.proj.s", _node("s", "select 1, 2 as discount", **snap), []), x],
        )
        assert self._changed(repo)["x"]["changed_upstream"] is True


class TestPackageModelPaths:
    """A package model's original_file_path is relative to its package: a root-project file at
    the same relative path changing must not list the package model as changed."""

    def _repo(self, checkout_repo, project_yml="name: proj"):
        repo = checkout_repo({"dbt_project.yml": project_yml, "models/stg.sql": "select 1"})
        repo.write("models/stg.sql", "select 2")
        repo.commit()
        return repo

    def _manifest(self, project_name="proj", root_checksum="s2"):
        manifest = json.loads(_graph(
            ("model.proj.stg", _with(_node("stg", "select 2", checksum=root_checksum), package="proj"), []),
            ("model.utils.pkg_stg", _with(_node("pkg_stg", "select 9", checksum="u", path="models/stg.sql"),
                                          package="utils"), []),
        ))
        if project_name:
            manifest["metadata"] = {"project_name": project_name}
        return json.dumps(manifest)

    def test_without_a_baseline_only_the_root_model_is_listed(self, checkout_repo):
        repo = self._repo(checkout_repo)
        repo.write("target/manifest.json", self._manifest())
        assert [c.name for c in find_changed_dbt_models(_config(repo))] == ["stg"]

    def test_root_package_from_dbt_project_yml_when_the_manifest_has_no_project_name(self, checkout_repo):
        repo = self._repo(checkout_repo)
        repo.write("target/manifest.json", self._manifest(project_name=None))
        assert [c.name for c in find_changed_dbt_models(_config(repo))] == ["stg"]

    def test_with_a_baseline_an_unchanged_package_model_is_not_listed(self, checkout_repo):
        """Not "the model file changed but the PR manifest was not recompiled" either."""
        repo = self._repo(checkout_repo)
        repo.write("target/manifest.json", self._manifest())
        repo.write(".diffcost-baseline/manifest.json", self._manifest(root_checksum="s1"))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["stg"]
        assert "unknown_reason" not in changed[0].metadata

    def test_changed_local_package_is_one_unknown_item_without_a_baseline(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "packages.yml": "packages:\n  - local: pkgs/utils",
                              "models/a.sql": "select 1", "pkgs/utils/dbt_project.yml": "name: utils",
                              "pkgs/utils/models/stg.sql": "select 1"})
        repo.write("pkgs/utils/models/stg.sql", "select * from big")
        repo.commit()
        manifest = json.loads(_graph(
            ("model.proj.a", _with(_node("a", "select 1"), package="proj"), []),
            ("model.utils.stg", _with(_node("stg", "select * from big"), package="utils"), []),
        ))
        manifest["metadata"] = {"project_name": "proj"}
        repo.write("target/manifest.json", json.dumps(manifest))
        changed = find_changed_dbt_models(_config(repo))
        assert [c.name for c in changed] == ["non-model dbt files"]
        assert "pkgs/utils/models/stg.sql" in changed[0].metadata["unknown_reason"]


class TestRelationCreated:
    def test_forms_and_kinds(self):
        from adapters.dbt import relation_created

        rels = ["`my-project`.`marts`.`fct_new`"]
        assert relation_created({"kind": "table", "name": "my-project:marts.fct_new"}, rels)
        assert relation_created({"kind": "view", "name": "my-project.marts.fct_new"}, rels)
        assert relation_created({"kind": "dataset", "name": "my-project:marts"}, rels)
        assert relation_created({"kind": "table", "name": "marts.fct_new"}, rels)  # no project in the error
        assert not relation_created({"kind": "table", "name": "my-project:ci.fct_new"}, rels)
        assert not relation_created({"kind": "dataset", "name": "my-project:ci_pr_12"}, rels)
        assert not relation_created({"kind": "table", "name": ""}, rels)
        assert not relation_created({"kind": "table", "name": "my-project:marts.fct_new"}, [])

    def test_domain_scoped_projects(self):
        """A project id such as `example.com:my-proj` holds a dot and a colon; BigQuery's error
        writes the table as `example.com:my-proj:marts.fct_new`."""
        from adapters.dbt import relation_created

        rels = ["`example.com:my-proj`.`marts`.`fct_new`"]
        assert relation_created({"kind": "table", "name": "example.com:my-proj:marts.fct_new"}, rels)
        assert relation_created({"kind": "view", "name": "example.com:my-proj.marts.fct_new"}, rels)
        assert relation_created({"kind": "dataset", "name": "example.com:my-proj:marts"}, rels)
        assert not relation_created({"kind": "table", "name": "other-proj:marts.fct_new"}, rels)
        assert not relation_created({"kind": "table", "name": "example.com:my-proj:ci.fct_new"}, rels)
        assert not relation_created({"kind": "dataset", "name": "example.com:other:marts"}, rels)


def _with_sources(nodes, sources):
    """A manifest from (unique_id, node, parents) tuples plus {source id: relation_name}."""
    manifest = json.loads(_graph(*nodes))
    manifest["sources"] = {sid: {"relation_name": rel, "resource_type": "source"} for sid, rel in sources.items()}
    return json.dumps(manifest)


class TestSourcesAndRelationNames:
    """Relation names are mapped only for nodes dbt builds: a source that names another table
    changes what its readers scan."""

    def test_a_source_pointed_at_another_table_changes_its_readers(self, checkout_repo):
        repo = checkout_repo({"models/stg.sql": "select id from {{ source('raw', 'events') }}",
                              "models/sources.yml": "identifier: events_sample"})
        repo.write("models/sources.yml", "identifier: events_full_history")
        repo.commit()
        src = "source.proj.raw.events"
        def stg(table):
            sql = f"select id from `p`.`raw`.`{table}`"
            return ("model.proj.stg", _node("stg", sql, checksum="same", relation="`p`.`prod`.`stg`"), [src])

        repo.write("target/manifest.json", _with_sources([stg("events_full_history")],
                                                         {src: "`p`.`raw`.`events_full_history`"}))
        repo.write(".diffcost-baseline/manifest.json", _with_sources([stg("events_sample")],
                                                                     {src: "`p`.`raw`.`events_sample`"}))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "stg"
        assert item.production_sql == "select id from `p`.`raw`.`events_sample`"
        assert item.pr_sql == "select id from `p`.`raw`.`events_full_history`"
        # A source isn't built by this PR: neither "Unrecognized name" nor "Not found" on it is
        # explained by it
        assert item.metadata["new_upstream_relations"] == []
        assert item.metadata["changed_upstream"] is False

    def test_a_model_moved_by_its_own_config_keeps_its_readers_unchanged(self, checkout_repo):
        """dbt builds the moved model from the same SQL, so its readers read the same data."""
        repo = checkout_repo({"models/up.sql": "select 1", "models/fct.sql": "select * from {{ ref('up') }}",
                              "models/schema.yml": "v1"})
        repo.write("models/schema.yml", "v2")
        repo.commit()
        up_base = ("model.proj.up", _with(_node("up", "select 1", checksum="u", relation="`p`.`prod`.`up`"),
                                          materialized="table"), [])
        up_pr = ("model.proj.up", _with(_node("up", "select 1", checksum="u", relation="`p`.`prod`.`orders_v2`"),
                                        materialized="table", alias="orders_v2"), [])
        def fct(rel):
            return ("model.proj.fct", _node("fct", f"select * from {rel}", checksum="f"), ["model.proj.up"])

        repo.write("target/manifest.json", _graph(up_pr, fct("`p`.`prod`.`orders_v2`")))
        repo.write(".diffcost-baseline/manifest.json", _graph(up_base, fct("`p`.`prod`.`up`")))
        assert find_changed_dbt_models(_config(repo)) == []

    def test_relation_names_that_swap_map_in_one_pass(self, checkout_repo):
        """Two models that exchanged names: each reader still compares equal to production."""
        repo = checkout_repo({"models/a.sql": "select 1", "models/b.sql": "select 2", "models/r.sql": "select 3"})
        def a(rel):
            return ("model.proj.a", _node("a", "select 1", checksum="a", relation=rel), [])

        def b(rel):
            return ("model.proj.b", _node("b", "select 2", checksum="b", relation=rel), [])

        def r(sql):
            return ("model.proj.r", _node("r", sql, checksum="r"), ["model.proj.a", "model.proj.b"])

        repo.write(".diffcost-baseline/manifest.json", _graph(
            a("`p`.`d`.`a`"), b("`p`.`d`.`b`"), r("select * from `p`.`d`.`a` join `p`.`d`.`b` using (id)"),
        ))
        repo.write("target/manifest.json", _graph(
            a("`p`.`d`.`b`"), b("`p`.`d`.`a`"), r("select * from `p`.`d`.`b` join `p`.`d`.`a` using (id)"),
        ))
        assert find_changed_dbt_models(_config(repo)) == []

    def test_one_pass_mapping(self):
        from adapters.dbt import _RelationMap

        swap = _RelationMap({"`p`.`d`.`a`": "`p`.`d`.`b`", "`p`.`d`.`b`": "`p`.`d`.`a`"})
        assert swap.apply("SELECT * FROM `p`.`d`.`a` JOIN `p`.`d`.`b`") == "SELECT * FROM `p`.`d`.`b` JOIN `p`.`d`.`a`"
        # A replacement is never replaced again, even when it is another key
        chain = _RelationMap({"`p`.`prod`.`aa`": "`p`.`d`.`b`", "`p`.`d`.`b`": "`p`.`d`.`c`"})
        assert chain.apply("SELECT * FROM `p`.`prod`.`aa`") == "SELECT * FROM `p`.`d`.`b`"
        # The longest key wins where one name starts another (unquoted names)
        longest = _RelationMap({"p.d.ab": "p.d.x", "p.d.a": "p.d.y"})
        assert longest.apply("p.d.ab p.d.a") == "p.d.x p.d.y"


def _snapshot(name, compiled, checksum, path=None, relation=None):
    return _node(name, compiled, checksum=checksum, path=path or f"snapshots/{name}.sql", relation=relation,
                 resource_type="snapshot")


class TestSnapshotsWithABaseline:
    """With a baseline, a snapshot is compared like a model and priced as its SELECT."""

    def _repo(self, checkout_repo, edit=True):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/m.sql": "select 1",
                              "snapshots/s.sql": "{% snapshot s %}select 1{% endsnapshot %}"})
        if edit:
            repo.write("snapshots/s.sql", "{% snapshot s %}select * from big{% endsnapshot %}")
            repo.commit()
        return repo

    def _write(self, repo, base_nodes, pr_nodes):
        m = ("model.proj.m", _node("m", "select 1", checksum="m"), [])
        repo.write(".diffcost-baseline/manifest.json", _graph(m, *base_nodes))
        repo.write("target/manifest.json", _graph(m, *pr_nodes))

    def test_a_changed_snapshot_is_a_row(self, checkout_repo):
        repo = self._repo(checkout_repo)
        self._write(repo, [("snapshot.proj.s", _snapshot("s", "select 1", "s1"), [])],
                    [("snapshot.proj.s", _snapshot("s", "select * from big", "s2"), [])])
        [item] = find_changed_dbt_models(_config(repo))
        assert (item.name, item.path) == ("s", "snapshots/s.sql")
        assert (item.production_sql, item.pr_sql) == ("select 1", "select * from big")
        assert item.metadata["unique_id"] == "snapshot.proj.s"
        assert item.metadata["changed_upstream"] is False

    def test_a_macro_change_to_a_snapshot_is_a_row(self, checkout_repo):
        repo = self._repo(checkout_repo, edit=False)
        self._write(repo, [("snapshot.proj.s", _snapshot("s", "select 1", "same"), [])],
                    [("snapshot.proj.s", _snapshot("s", "select 1, 2 as x", "same"), [])])
        assert [c.name for c in find_changed_dbt_models(_config(repo))] == ["s"]

    def test_an_unchanged_snapshot_is_not_listed(self, checkout_repo):
        repo = self._repo(checkout_repo, edit=False)
        node = ("snapshot.proj.s", _snapshot("s", "select 1", "same"), [])
        self._write(repo, [node], [node])
        assert find_changed_dbt_models(_config(repo)) == []

    def test_a_snapshot_whose_config_changed_but_not_its_select(self, checkout_repo):
        """Its checksum covers the whole file (for one defined in YAML, every edit to that file):
        a snapshot is priced as its SELECT, so only that decides."""
        repo = self._repo(checkout_repo)
        self._write(repo, [("snapshot.proj.s", _snapshot("s", "select 1", "s1"), [])],
                    [("snapshot.proj.s", _snapshot("s", "select 1", "s2"), [])])
        assert find_changed_dbt_models(_config(repo)) == []

    def test_a_compared_snapshot_file_is_not_a_non_model_change(self, checkout_repo):
        """An uncompiled model makes changed non-model files one unknown item; a snapshot's own
        `.sql` file is compared through its node, so it isn't one of them."""
        repo = self._repo(checkout_repo)
        u = ("model.proj.u", _node("u", None, checksum="u", path="models/u.sql"), [])  # its file is unchanged
        self._write(repo, [u, ("snapshot.proj.s", _snapshot("s", "select 1", "s1"), [])],
                    [u, ("snapshot.proj.s", _snapshot("s", "select * from big", "s2"), [])])
        assert [c.name for c in find_changed_dbt_models(_config(repo))] == ["s"]

    def test_a_macro_change_still_counts_next_to_an_uncompiled_model(self, checkout_repo):
        repo = self._repo(checkout_repo)
        repo.write("macros/m.sql", "{% macro m() %}{% endmacro %}")
        repo.commit()
        u = ("model.proj.u", _node("u", None, checksum="u", path="models/u.sql"), [])
        self._write(repo, [u, ("snapshot.proj.s", _snapshot("s", "select 1", "s1"), [])],
                    [u, ("snapshot.proj.s", _snapshot("s", "select * from big", "s2"), [])])
        items = {c.name: c for c in find_changed_dbt_models(_config(repo))}
        assert set(items) == {"s", "non-model dbt files"}
        assert "macros/m.sql" in items["non-model dbt files"].metadata["unknown_reason"]
        assert "snapshots/s.sql" not in items["non-model dbt files"].metadata["unknown_reason"]

    def test_new_and_removed_snapshots(self, checkout_repo):
        repo = self._repo(checkout_repo, edit=False)
        self._write(repo, [("snapshot.proj.old", _snapshot("old", "select 2", "o"), [])],
                    [("snapshot.proj.new", _snapshot("new", "select 3", "n"), [])])
        items = {c.name: c for c in find_changed_dbt_models(_config(repo))}
        assert (items["new"].production_sql, items["new"].pr_sql) == (None, "select 3")
        assert (items["old"].production_sql, items["old"].pr_sql) == ("select 2", None)

    def test_a_snapshot_compiled_in_one_manifest_only_is_unknown(self, checkout_repo):
        repo = self._repo(checkout_repo)
        self._write(repo, [("snapshot.proj.s", _snapshot("s", None, "s1"), [])],
                    [("snapshot.proj.s", _snapshot("s", "select * from big", "s2"), [])])
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "s"
        assert item.metadata["unknown_reason"] == (
            "the snapshot is not compiled in the baseline manifest (selective compile?), so its cost can't be compared"
        )

    def test_a_snapshot_reading_a_column_an_upstream_model_adds(self, checkout_repo):
        """The Not costed path works for snapshots: their upstream is tracked like a model's."""
        repo = self._repo(checkout_repo)
        repo.write("models/m.sql", "select 1, 2 as discount")
        repo.commit()
        m_base = ("model.proj.m", _node("m", "select 1", checksum="m1"), [])
        m_pr = ("model.proj.m", _node("m", "select 1, 2 as discount", checksum="m2"), [])
        repo.write(".diffcost-baseline/manifest.json", _graph(m_base, ("snapshot.proj.s", _snapshot("s", "select 1", "s1"), [])))
        repo.write("target/manifest.json", _graph(
            m_pr, ("snapshot.proj.s", _snapshot("s", "select discount from m", "s2"), ["model.proj.m"]),
        ))
        items = {c.name: c for c in find_changed_dbt_models(_config(repo))}
        assert items["s"].metadata["changed_upstream"] is True

    def test_without_a_baseline_snapshots_stay_in_the_non_model_row(self, checkout_repo):
        repo = self._repo(checkout_repo)
        repo.write("target/manifest.json", _graph(
            ("model.proj.m", _node("m", "select 1", checksum="m"), []),
            ("snapshot.proj.s", _snapshot("s", "select * from big", "s2"), []),
        ))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "non-model dbt files"
        assert "snapshots/s.sql" in item.metadata["unknown_reason"]

    def test_tests_and_analyses_are_not_costed(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/m.sql": "select 1",
                              "tests/assert_x.sql": "select 1", "analyses/a.sql": "select 1"})
        repo.write("tests/assert_x.sql", "select * from big")
        repo.write("analyses/a.sql", "select * from big")
        repo.commit()
        m = ("model.proj.m", _node("m", "select 1", checksum="m"), [])
        def t(sql):
            node = _node("assert_x", sql, checksum=sql, path="tests/assert_x.sql", resource_type="test")
            return ("test.proj.assert_x", node, ["model.proj.m"])

        repo.write(".diffcost-baseline/manifest.json", _graph(m, t("select 1")))
        repo.write("target/manifest.json", _graph(m, t("select * from big")))
        assert find_changed_dbt_models(_config(repo)) == []


class TestComparedWhereTheBranchStarted:
    def test_a_committed_baseline_manifest_is_read_where_the_branch_starts(self, checkout_repo):
        """The committed copy that describes the branch's starting point, not the base's tip."""
        from tests.conftest import _git

        start = _manifest(_node("a", "select 1", checksum="a1"))
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/a.sql": "select 1",
                              "target/manifest.json": start})
        repo.write("models/a.sql", "select 2")
        repo.git("rm", "-q", "target/manifest.json")  # the PR stops committing it
        repo.commit("pr")
        (repo.upstream / "models/a.sql").write_text("select 3")
        (repo.upstream / "target/manifest.json").write_text(_manifest(_node("a", "select 3", checksum="a3")))
        _git(repo.upstream, "add", "-A")
        _git(repo.upstream, "commit", "-q", "-m", "base moves on")
        repo.git("fetch", "-q", "origin")
        repo.write("target/manifest.json", _manifest(_node("a", "select 2", checksum="a2")))  # compiled in the run
        tip = repo.git("rev-parse", "origin/main").strip()
        [item] = find_changed_dbt_models(_config(repo, resolved_base_ref=tip))
        assert (item.production_sql, item.pr_sql) == ("select 1", "select 2")
        # The base changed models/a.sql too since the branch started
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON

    def _base_moves_on(self, checkout_repo, base_change):
        """The PR changes models/a.sql; then the base commits `base_change` (a path -> content)."""
        from tests.conftest import _git

        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/a.sql": "select 1", "models/b.sql": "select 5"})
        repo.write("models/a.sql", "select 2")
        repo.commit("pr")
        path, content = base_change
        (repo.upstream / path).write_text(content)
        _git(repo.upstream, "add", "-A")
        _git(repo.upstream, "commit", "-q", "-m", "base moves on")
        repo.git("fetch", "-q", "origin")
        b = _node("b", "select 5", checksum="b")
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a1"), b))
        repo.write("target/manifest.json", _manifest(_node("a", "select 2", checksum="a2"), b))
        return repo, repo.git("rev-parse", "origin/main").strip()

    def test_a_model_the_base_also_changed_later_is_unknown(self, checkout_repo):
        repo, tip = self._base_moves_on(checkout_repo, ("models/a.sql", "select 3"))
        [item] = find_changed_dbt_models(_config(repo, resolved_base_ref=tip))
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON

    def test_a_model_only_the_pr_changed_is_priced(self, checkout_repo):
        repo, tip = self._base_moves_on(checkout_repo, ("models/b.sql", "select 6"))
        [item] = find_changed_dbt_models(_config(repo, resolved_base_ref=tip))
        assert (item.name, item.production_sql, item.pr_sql) == ("a", "select 1", "select 2")
        assert "unknown_reason" not in item.metadata

    def _macro_on_both_sides(self, checkout_repo, macro_path="macros/m.sql", files=None):
        """The base put a saving into a macro after the branch started; the PR has that saving
        too (cherry-picked) plus an increase of its own in the same file."""
        from tests.conftest import _git

        before = "{% macro f() %}\nwhere true\n-- joins\n\n\n\n\n{% endmacro %}\n"
        saving = before.replace("where true", "where dt = current_date()")
        both = saving.replace("-- joins", "cross join huge")
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/r.sql": "select * from t {{ f() }}",
                              macro_path: before, **(files or {})})
        repo.write(macro_path, both)
        repo.commit("pr")
        (repo.upstream / macro_path).write_text(saving)
        _git(repo.upstream, "commit", "-q", "-am", "base moves on")
        repo.git("fetch", "-q", "origin")
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("r", "select * from t where true", checksum="r")))
        repo.write("target/manifest.json", _manifest(
            _node("r", "select * from t where dt = current_date() cross join huge", checksum="r")))
        return repo, repo.git("rev-parse", "origin/main").strip()

    def test_a_macro_the_base_also_changed_later_is_unknown(self, checkout_repo):
        """Priced from the branch point, the base's saving would offset the PR's increase."""
        repo, tip = self._macro_on_both_sides(checkout_repo)
        items = {c.name: c.metadata for c in find_changed_dbt_models(_config(repo, resolved_base_ref=tip))}
        assert set(items) == {"r", "non-model dbt files"}
        assert items["non-model dbt files"]["unknown_reason"] == (
            "1 changed file(s) outside model files were also changed by the base branch since this PR's branch "
            "started: update the branch (merge or rebase the base branch into it) (macros/m.sql)"
        )

    def test_the_comment_keeps_what_to_do_with_long_paths(self, checkout_repo):
        from adapters.dbt import _base_moved_non_model_reason
        from comment import REASON_LIMIT, shorten

        paths = [f"dbt/very/deeply/nested/directory/number_{i}/with_a_long_file_name_{i}.yml" for i in range(5)]
        assert "update the branch" in shorten(_base_moved_non_model_reason(paths), REASON_LIMIT)
        assert "update the branch" in shorten(
            f"{BASE_MOVED_REASON}; no production baseline (no baseline manifest in this run)", REASON_LIMIT)

    def test_a_renamed_macro_the_base_also_changed_later_is_unknown(self, checkout_repo):
        """The PR moved the file (still a rename) and edited it; the base edited it at the old path."""
        from tests.conftest import _git

        body = "".join(f"-- line {i}\n" for i in range(20))
        macro = "{% macro f() %}\nwhere true\n" + body + "{% endmacro %}\n"
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/r.sql": "select * from t {{ f() }}",
                              "macros/m.sql": macro})
        repo.git("mv", "macros/m.sql", "macros/m2.sql")
        repo.write("macros/m2.sql", macro.replace("where true", "where true cross join huge"))
        repo.commit("pr")
        assert repo.git("diff", "--name-status", "-M", "origin/main...HEAD").split()[0].startswith("R")
        (repo.upstream / "macros/m.sql").write_text(macro.replace("-- line 19", "-- line 19, base"))
        _git(repo.upstream, "commit", "-q", "-am", "base moves on")
        repo.git("fetch", "-q", "origin")
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("r", "select * from t where true", checksum="r")))
        repo.write("target/manifest.json", _manifest(_node("r", "select * from t where true cross join huge", checksum="r")))
        tip = repo.git("rev-parse", "origin/main").strip()
        items = {c.name: c.metadata for c in find_changed_dbt_models(_config(repo, resolved_base_ref=tip))}
        assert "(macros/m2.sql)" in items["non-model dbt files"]["unknown_reason"]

    @pytest.mark.parametrize("path", ["analyses/a.sql", "tests/assert_x.sql"])
    def test_a_test_or_analysis_sql_both_changed_is_not_counted(self, checkout_repo, path):
        """Neither changes a model, and a PR that changes only them reports no SQL changes."""
        repo, tip = self._macro_on_both_sides(checkout_repo, path)
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("r", "select 1", checksum="r")))
        repo.write("target/manifest.json", _manifest(_node("r", "select 1", checksum="r")))
        assert find_changed_dbt_models(_config(repo, resolved_base_ref=tip)) == []

    def test_test_path_yaml_both_changed_still_counts(self, checkout_repo):
        repo, tip = self._macro_on_both_sides(checkout_repo, "tests/props.yml")
        items = {c.name: c.metadata for c in find_changed_dbt_models(_config(repo, resolved_base_ref=tip))}
        assert "(tests/props.yml)" in items["non-model dbt files"]["unknown_reason"]

    def test_one_non_model_item_when_a_model_is_also_uncompiled(self, checkout_repo):
        """The base-moved reason joins the selective-compile one, first."""
        repo, tip = self._macro_on_both_sides(checkout_repo)
        u = _node("u", None, checksum="u")
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("r", "select * from t where true", checksum="r"), u))
        repo.write("target/manifest.json", _manifest(
            _node("r", "select * from t where dt = current_date() cross join huge", checksum="r"), u))
        items = [c for c in find_changed_dbt_models(_config(repo, resolved_base_ref=tip)) if c.name == "non-model dbt files"]
        assert len(items) == 1
        reason = items[0].metadata["unknown_reason"]
        assert reason.startswith("1 changed file(s) outside model files were also changed by the base branch")
        assert "; 1 changed file(s) outside model files (macros/m.sql) can change the SQL of 1 model(s)" in reason

    def test_a_local_package_file_the_base_also_changed_later_is_unknown(self, checkout_repo):
        repo, tip = self._macro_on_both_sides(
            checkout_repo, "pkgs/shared/macros/m.sql", {"packages.yml": "packages:\n  - local: pkgs/shared\n"})
        items = {c.name: c.metadata for c in find_changed_dbt_models(_config(repo, resolved_base_ref=tip))}
        assert "pkgs/shared/macros/m.sql" in items["non-model dbt files"]["unknown_reason"]

    def test_a_renamed_model_whose_old_file_the_base_changed_is_unknown(self, checkout_repo):
        from tests.conftest import _git

        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/old.sql": "select 1\n-- 1\n-- 2\n-- 3\n"})
        repo.git("mv", "models/old.sql", "models/new.sql")
        repo.write("models/new.sql", "select 2\n-- 1\n-- 2\n-- 3\n")
        repo.commit("pr")
        (repo.upstream / "models/old.sql").write_text("select 1\n-- 1\n-- 2\n-- 3, base\n")
        _git(repo.upstream, "commit", "-q", "-am", "base moves on")
        repo.git("fetch", "-q", "origin")
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("old", "select 1", checksum="o")))
        repo.write("target/manifest.json", _manifest(_node("new", "select 2", checksum="n", path="models/new.sql")))
        tip = repo.git("rev-parse", "origin/main").strip()
        [item] = find_changed_dbt_models(_config(repo, resolved_base_ref=tip))
        assert (item.name, item.production_sql) == ("new", "select 1")
        assert item.metadata["unknown_reason"] == BASE_MOVED_REASON

    def test_without_a_baseline_both_reasons_are_kept(self, checkout_repo):
        repo, tip = self._base_moves_on(checkout_repo, ("models/a.sql", "select 3"))
        (repo.path / ".diffcost-baseline/manifest.json").unlink()
        [item] = find_changed_dbt_models(_config(repo, resolved_base_ref=tip))
        assert item.metadata["unknown_reason"] == (
            f"{BASE_MOVED_REASON}; no production baseline (no baseline manifest in this run)"
        )

    def test_a_failed_diff_of_the_base_fails_the_run(self, checkout_repo, monkeypatch):
        repo, tip = self._base_moves_on(checkout_repo, ("models/b.sql", "select 6"))
        def broken(*_):
            raise GitError("Could not diff: boom")

        monkeypatch.setattr("config.paths_changed_between", broken)
        with pytest.raises(GitError, match="boom"):
            find_changed_dbt_models(_config(repo, resolved_base_ref=tip))


class TestRenamedModels:
    """A model file git reports as renamed is compared with its old version in one row."""

    def _renamed(self, checkout_repo, old_sql="select * from big", new_sql=None):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/old.sql": "select * from {{ ref('big') }}",
                              "models/reader.sql": "select * from {{ ref('old') }}"})
        repo.git("mv", "models/old.sql", "models/new.sql")
        repo.write("models/reader.sql", "select * from {{ ref('new') }}")
        repo.commit()
        old = ("model.proj.old", _node("old", old_sql, checksum="c", relation="`p`.`prod`.`old`"), [])
        new = ("model.proj.new", _node("new", new_sql or old_sql, checksum="c", path="models/new.sql",
                                       relation="`p`.`prod`.`new`"), [])
        def reader(rel, parent):
            node = _node("reader", f"select * from {rel}", checksum=rel, relation="`p`.`prod`.`reader`")
            return ("model.proj.reader", node, [parent])

        repo.write(".diffcost-baseline/manifest.json", _graph(old, reader("`p`.`prod`.`old`", "model.proj.old")))
        repo.write("target/manifest.json", _graph(new, reader("`p`.`prod`.`new`", "model.proj.new")))
        return repo

    def test_a_pure_rename_and_its_readers_are_not_listed(self, checkout_repo):
        """Neither a Removed + New pair for the model, nor its reader (only `ref()` changed)."""
        assert find_changed_dbt_models(_config(self._renamed(checkout_repo))) == []

    def test_a_renamed_model_whose_sql_changed_is_one_row(self, checkout_repo):
        repo = self._renamed(checkout_repo, new_sql="select * from big where dt = current_date()")
        [item] = find_changed_dbt_models(_config(repo))
        assert (item.name, item.path) == ("new", "models/new.sql")
        assert item.metadata["unique_id"] == "model.proj.new"
        assert item.production_sql == "select * from big"
        assert item.pr_sql == "select * from big where dt = current_date()"

    def test_without_rename_detection_it_stays_new_and_removed(self, checkout_repo):
        """A file deleted and another added with different content: no pairing."""
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/old.sql": "select 1 from a, b, c, d"})
        repo.git("rm", "-q", "models/old.sql")
        repo.write("models/new.sql", "select * from somewhere_else")
        repo.commit()
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("old", "select 1", checksum="o")))
        repo.write("target/manifest.json", _manifest(_node("new", "select 2", checksum="n")))
        items = {c.name: c for c in find_changed_dbt_models(_config(repo))}
        assert (items["old"].pr_sql, items["new"].production_sql) == (None, None)

    def _renamed_seed(self, checkout_repo, reader_sql_pr):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "seeds/countries.csv": "code,name\nAU,Australia\n",
                              "models/reader.sql": "select * from {{ ref('countries') }}"})
        repo.git("mv", "seeds/countries.csv", "seeds/country_codes.csv")
        repo.write("models/reader.sql", "select * from {{ ref('country_codes') }}")
        repo.commit()
        def seed(name):
            return (f"seed.proj.{name}", _node(name, None, checksum="csv", path=f"seeds/{name}.csv",
                                               relation=f"`p`.`prod`.`{name}`", resource_type="seed"), [])

        def reader(sql, parent):
            return ("model.proj.reader", _node("reader", sql, checksum=sql), [parent])

        repo.write(".diffcost-baseline/manifest.json", _graph(
            seed("countries"), reader("select * from `p`.`prod`.`countries`", "seed.proj.countries")))
        repo.write("target/manifest.json", _graph(
            seed("country_codes"), reader(reader_sql_pr, "seed.proj.country_codes")))
        return repo

    def test_a_reader_of_a_renamed_seed_is_not_listed(self, checkout_repo):
        repo = self._renamed_seed(checkout_repo, "select * from `p`.`prod`.`country_codes`")
        assert find_changed_dbt_models(_config(repo)) == []

    def test_a_reader_of_a_renamed_seed_is_analyzed_under_the_old_name(self, checkout_repo):
        repo = self._renamed_seed(checkout_repo, "select code from `p`.`prod`.`country_codes`")
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "reader"
        assert item.metadata["structural_pr_sql"] == "select code from `p`.`prod`.`countries`"

    def test_renames_are_detected_whatever_the_repository_setting(self, checkout_repo):
        repo = self._renamed(checkout_repo, new_sql="select 2")
        repo.git("config", "diff.renames", "false")
        [item] = find_changed_dbt_models(_config(repo))
        assert (item.name, item.production_sql) == ("new", "select * from big")


def _table(name, compiled="select 1", materialized="table", **config):
    return _with(_node(name, compiled, checksum=name, relation=f"`p`.`prod`.`{name}`"), materialized=materialized, **config)


class TestPartitioningAndClustering:
    """Removing or changing a table's partitioning or clustering changes what its readers scan
    once it is rebuilt: no dry run before the merge can price that."""

    def _changed(self, checkout_repo, base_config, pr_config, base_mat="table", pr_mat="table", compiled=None):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/t.sql": "select 1",
                              "models/schema.yml": "v1"})
        repo.write("models/schema.yml", "v2")
        repo.commit()
        repo.write(".diffcost-baseline/manifest.json",
                   _graph(("model.proj.t", _table("t", materialized=base_mat, **base_config), [])))
        repo.write("target/manifest.json",
                   _graph(("model.proj.t", _table("t", compiled or "select 1", materialized=pr_mat, **pr_config), [])))
        return {c.name: c.metadata for c in find_changed_dbt_models(_config(repo))}

    DAY = {"field": "event_date", "data_type": "date"}

    @pytest.mark.parametrize("base, pr, expected", [
        ({"partition_by": DAY}, {}, "partitioning (`partition_by`) removed"),
        ({"partition_by": DAY}, {"partition_by": {"field": "created_at", "data_type": "date"}},
         "partitioning (`partition_by`) changed"),
        ({"partition_by": DAY}, {"partition_by": {**DAY, "granularity": "month"}}, "partitioning (`partition_by`) changed"),
        ({"cluster_by": ["a", "b"]}, {"cluster_by": ["b", "a"]}, "clustering (`cluster_by`) changed"),
        ({"cluster_by": "a"}, {}, "clustering (`cluster_by`) removed"),
        ({"partition_by": DAY, "cluster_by": "a"}, {"cluster_by": "b"},
         "partitioning (`partition_by`) removed and clustering (`cluster_by`) changed"),
    ])
    def test_removed_or_changed_is_an_unknown_cost_row(self, checkout_repo, base, pr, expected):
        meta = self._changed(checkout_repo, base, pr)["t"]
        assert meta["unknown_reason"].startswith(f"{expected} in this PR")

    @pytest.mark.parametrize("base, pr", [
        ({}, {"partition_by": {"field": "event_date", "data_type": "date"}, "cluster_by": "a"}),  # added
        ({"partition_by": {"field": "event_date", "data_type": "date"}},
         {"partition_by": {"Field": "EVENT_DATE", "data_type": "DATE", "granularity": "day"}}),  # same, spelled out
        ({"cluster_by": "a"}, {"cluster_by": ["A"]}),
        ({"partition_by": {"field": "n", "data_type": "int64", "range": {"start": 0, "end": 10, "interval": 1}}},
         {"partition_by": {"data_type": "int64", "field": "n", "range": {"interval": 1, "end": 10, "start": 0}}}),
    ])
    def test_added_or_unchanged_is_not_listed(self, checkout_repo, base, pr):
        assert self._changed(checkout_repo, base, pr) == {}

    def test_a_view_that_was_a_partitioned_table(self, checkout_repo):
        meta = self._changed(checkout_repo, {"partition_by": self.DAY}, {}, pr_mat="view")["t"]
        assert meta["unknown_reason"].startswith("partitioning (`partition_by`) removed")

    def test_an_ephemeral_model_is_priced_through_its_readers(self, checkout_repo):
        assert self._changed(checkout_repo, {"partition_by": self.DAY}, {}, pr_mat="ephemeral") == {}

    def test_a_view_never_had_partitioning_to_lose(self, checkout_repo):
        assert self._changed(checkout_repo, {"partition_by": self.DAY}, {}, base_mat="view", pr_mat="view") == {}

    def test_the_reason_goes_on_a_row_that_changed_anyway(self, checkout_repo):
        items = self._changed(checkout_repo, {"partition_by": self.DAY}, {}, compiled="select 2")
        assert list(items) == ["t"]
        assert items["t"]["unknown_reason"].startswith("partitioning (`partition_by`) removed")

    def test_a_renamed_model_that_loses_its_partitioning(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/old.sql": "select 1"})
        repo.git("mv", "models/old.sql", "models/new.sql")
        repo.commit()
        old = ("model.proj.old", _table("old", partition_by=self.DAY), [])
        new = ("model.proj.new", _with(_node("new", "select 1", checksum="old", path="models/new.sql",
                                             relation="`p`.`prod`.`new`"), materialized="table"), [])
        repo.write(".diffcost-baseline/manifest.json", _graph(old))
        repo.write("target/manifest.json", _graph(new))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "new" and item.production_sql == "select 1"
        assert item.metadata["unknown_reason"].startswith("partitioning (`partition_by`) removed in this PR")

    def test_an_uncompiled_model_that_loses_its_partitioning(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/t.sql": "select 1", "models/m.sql": "select 2",
                              "models/schema.yml": "v1"})
        repo.write("models/schema.yml", "v2")  # where the config is set
        repo.commit()
        m = ("model.proj.m", _node("m", "select 2", checksum="m"), [])
        repo.write(".diffcost-baseline/manifest.json", _graph(m, ("model.proj.t", _table("t", partition_by=self.DAY), [])))
        repo.write("target/manifest.json", _graph(m, ("model.proj.t", _table("t", compiled=None), [])))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "t"
        assert item.metadata["unknown_reason"].startswith("partitioning (`partition_by`) removed in this PR")

    def _not_set_by_this_pr(self, checkout_repo, edit=None):
        """The baseline has partitioning the PR manifest doesn't, though the PR changed no
        file that could set it (a baseline compiled from another commit)."""
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/t.sql": "select 1", "models/m.sql": "select 2",
                              "README.md": "v1"})
        repo.write(*(edit or ("README.md", "v2")))
        repo.commit()
        m = ("model.proj.m", _node("m", "select 2", checksum="m"), [])
        repo.write(".diffcost-baseline/manifest.json", _graph(m, ("model.proj.t", _table("t", partition_by=self.DAY), [])))
        pr_t = _table("t")
        if edit and edit[0] == "models/t.sql":
            pr_t["checksum"]["checksum"] = "t-edited"  # compiled from the edited file
        repo.write("target/manifest.json", _graph(m, ("model.proj.t", pr_t, [])))
        return repo

    def test_a_difference_no_changed_file_can_set_is_only_a_warning(self, checkout_repo, capsys):
        repo = self._not_set_by_this_pr(checkout_repo)
        assert find_changed_dbt_models(_config(repo)) == []
        out = capsys.readouterr().out
        assert "::warning::1 model(s) or snapshot(s) have different partitioning or clustering" in out
        assert "dbt#how-the-baseline-step-works" in out

    @pytest.mark.parametrize("edit", [
        ("models/t.sql", "{{ config(materialized='table') }} select 1"),  # the model's own file
        ("dbt_project.yml", "name: proj\nmodels:\n  +partition_by: null\n"),
        ("macros/partitions.sql", "{% macro p() %}{% endmacro %}"),
    ])
    def test_a_change_to_a_file_that_can_set_it_counts(self, checkout_repo, edit):
        repo = self._not_set_by_this_pr(checkout_repo, edit)
        items = {c.name: c.metadata for c in find_changed_dbt_models(_config(repo))}
        assert items["t"]["unknown_reason"].startswith("partitioning (`partition_by`) removed in this PR")

    @pytest.mark.parametrize("base, pr", [
        ({"cluster_by": "a,b"}, {"cluster_by": ["a", "b"]}),
        ({"cluster_by": ["a", "b"]}, {"cluster_by": "a, b"}),
        ({"cluster_by": ["customer_id"]}, {"cluster_by": ["customer_id", "order_date"]}),  # columns added after
    ])
    def test_the_same_or_extended_clustering_is_not_listed(self, checkout_repo, base, pr):
        assert self._changed(checkout_repo, base, pr) == {}

    @pytest.mark.parametrize("base, pr", [
        ({"cluster_by": ["a", "b"]}, {"cluster_by": ["a"]}),  # a column removed
        ({"cluster_by": ["a"]}, {"cluster_by": ["b", "a"]}),  # a column put first
    ])
    def test_shortened_or_reordered_clustering_is_listed(self, checkout_repo, base, pr):
        meta = self._changed(checkout_repo, base, pr)["t"]
        assert meta["unknown_reason"].startswith("clustering (`cluster_by`) changed in this PR")

    def _turned_into_a_view(self, checkout_repo, base_mat="table", readers=True, **base_config):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/t.sql": "select 1", "models/r.sql": "select 2"})
        repo.write("models/t.sql", "{{ config(materialized='view') }} select 1")
        repo.commit()
        reader = [("model.proj.r", _node("r", "select * from `p`.`prod`.`t`", checksum="r"), ["model.proj.t"])]
        base = _table("t", materialized=base_mat, **base_config)
        pr = _with(_node("t", "select 1", checksum="t2", relation="`p`.`prod`.`t`"), materialized="view")
        repo.write(".diffcost-baseline/manifest.json", _graph(("model.proj.t", base, []), *(reader if readers else [])))
        repo.write("target/manifest.json", _graph(("model.proj.t", pr, []), *(reader if readers else [])))
        return {c.name: c.metadata for c in find_changed_dbt_models(_config(repo))}

    @pytest.mark.parametrize("base_mat", ["table", "incremental"])
    def test_a_table_turned_into_a_view_that_is_read_is_unknown(self, checkout_repo, base_mat):
        meta = self._turned_into_a_view(checkout_repo, base_mat)["t"]
        assert meta["unknown_reason"] == (
            "materialized as a view (its readers will run its query) in this PR: what the models that read it scan "
            "changes after the merge, which a dry run can't price before it"
        )

    def test_a_partitioned_table_turned_into_a_read_view_gets_one_reason(self, checkout_repo):
        meta = self._turned_into_a_view(checkout_repo, partition_by=self.DAY)["t"]
        assert meta["unknown_reason"].startswith("materialized as a view (its readers will run its query) in this PR")
        assert "partitioning" not in meta["unknown_reason"]

    def test_a_table_turned_into_a_view_nothing_reads_is_priced(self, checkout_repo):
        meta = self._turned_into_a_view(checkout_repo, readers=False)["t"]
        assert "unknown_reason" not in meta


class TestMovedAndRenamedReaders:
    def test_a_model_moved_to_another_directory_is_not_listed(self, checkout_repo):
        """Same name, so the same unique_id: the PR manifest records the new path."""
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/a.sql": "select 1"})
        (repo.path / "models/staging").mkdir()
        repo.git("mv", "models/a.sql", "models/staging/a.sql")
        repo.commit()
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a")))
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a", path="models/staging/a.sql")))
        assert find_changed_dbt_models(_config(repo)) == []

    def test_a_moved_model_with_a_stale_manifest_is_unknown(self, checkout_repo):
        """The PR manifest still has the old path: it wasn't compiled from this checkout."""
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/a.sql": "select 1", "models/m.sql": "select 2"})
        (repo.path / "models/staging").mkdir()
        repo.git("mv", "models/a.sql", "models/staging/a.sql")
        repo.commit()
        m = _node("m", "select 2", checksum="m")
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a"), m))
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a"), m))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "a"
        assert item.metadata["unknown_reason"] == "the model file changed but the PR manifest was not recompiled"

    def test_a_moved_snapshot_is_not_listed(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/m.sql": "select 1",
                              "snapshots/s.sql": "{% snapshot s %}select 1{% endsnapshot %}"})
        (repo.path / "snapshots/core").mkdir()
        repo.git("mv", "snapshots/s.sql", "snapshots/core/s.sql")
        repo.commit()
        m = ("model.proj.m", _node("m", "select 1", checksum="m"), [])
        repo.write(".diffcost-baseline/manifest.json", _graph(m, ("snapshot.proj.s", _snapshot("s", "select 1", "s"), [])))
        repo.write("target/manifest.json", _graph(
            m, ("snapshot.proj.s", _snapshot("s", "select 1", "s", path="snapshots/core/s.sql"), []),
        ))
        assert find_changed_dbt_models(_config(repo)) == []

    def test_a_reader_of_a_renamed_model_is_analyzed_under_the_old_name(self, checkout_repo):
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/old.sql": "select 1 as id",
                              "models/reader.sql": "select id from {{ ref('old') }}"})
        repo.git("mv", "models/old.sql", "models/new.sql")
        repo.write("models/reader.sql", "select id, 2 as discount from {{ ref('new') }}")
        repo.commit()
        old = ("model.proj.old", _node("old", "select 1 as id", checksum="o", relation="`p`.`prod`.`old`"), [])
        new = ("model.proj.new", _node("new", "select 1 as id", checksum="o", path="models/new.sql",
                                       relation="`p`.`prod`.`new`"), [])
        base_reader = ("model.proj.reader", _node("reader", "select id from `p`.`prod`.`old`", checksum="r1"),
                       ["model.proj.old"])
        pr_reader = ("model.proj.reader", _node("reader", "select id, 2 as discount from `p`.`prod`.`new`",
                                                checksum="r2"), ["model.proj.new"])
        repo.write(".diffcost-baseline/manifest.json", _graph(old, base_reader))
        repo.write("target/manifest.json", _graph(new, pr_reader))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "reader"
        assert item.pr_sql == "select id, 2 as discount from `p`.`prod`.`new`"  # what the dry run prices
        assert item.metadata["structural_pr_sql"] == "select id, 2 as discount from `p`.`prod`.`old`"
        assert item.metadata["new_upstream_relations"] == ["`p`.`prod`.`new`"]

    def test_no_mapped_copy_when_nothing_maps(self, checkout_repo):
        repo = checkout_repo({"models/a.sql": "select 1"})
        repo.write("models/a.sql", "select 2")
        repo.commit()
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a1")))
        repo.write("target/manifest.json", _manifest(_node("a", "select 2", checksum="a2")))
        [item] = find_changed_dbt_models(_config(repo))
        assert "structural_pr_sql" not in item.metadata

    def test_a_macro_change_with_an_uncompiled_snapshot_is_unknown(self, checkout_repo):
        """A snapshot compiled in one manifest only can't show the macro's effect either."""
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/m.sql": "select 1",
                              "snapshots/s.sql": "{% snapshot s %}select {{ m() }}{% endsnapshot %}",
                              "macros/m.sql": "{% macro m() %}1{% endmacro %}"})
        repo.write("macros/m.sql", "{% macro m() %}* from big{% endmacro %}")
        repo.commit()
        m = ("model.proj.m", _node("m", "select 1", checksum="m"), [])
        repo.write(".diffcost-baseline/manifest.json", _graph(m, ("snapshot.proj.s", _snapshot("s", "select 1", "s"), [])))
        repo.write("target/manifest.json", _graph(m, ("snapshot.proj.s", _snapshot("s", None, "s"), [])))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.name == "non-model dbt files"
        assert "1 model(s) or snapshot(s) that are not compiled in both manifests" in item.metadata["unknown_reason"]

    def test_a_path_only_the_baseline_has_proves_nothing(self, checkout_repo):
        """A baseline saved before the base branch moved the file: the PR edits it in its new
        place, and a PR manifest compiled from the base branch still has the old SQL."""
        repo = checkout_repo({"dbt_project.yml": "name: proj", "models/staging/a.sql": "select 1"})
        repo.write("models/staging/a.sql", "select 1, 2")
        repo.commit()
        repo.write(".diffcost-baseline/manifest.json", _manifest(_node("a", "select 1", checksum="a")))
        repo.write("target/manifest.json", _manifest(_node("a", "select 1", checksum="a", path="models/staging/a.sql")))
        [item] = find_changed_dbt_models(_config(repo))
        assert item.metadata["unknown_reason"] == "the model file changed but the PR manifest was not recompiled"
