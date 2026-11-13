"""Shared test fixtures and configuration."""

import subprocess
import sys
from pathlib import Path

import pytest

# Add src/ to Python path so tests can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


class CheckoutRepo:
    """A repo laid out the way actions/checkout leaves a pull_request run.

    Branches exist only as `origin/<name>` and HEAD is detached (no local `main`).
    """

    def __init__(self, root: Path):
        self.root = root
        self.upstream = root / "upstream"
        self.path = root / "work"

    def write(self, rel: str, content: str) -> None:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def commit(self, message: str = "change") -> None:
        _git(self.path, "add", "-A")
        _git(self.path, "commit", "-q", "-m", message)

    def git(self, *args: str) -> str:
        return _git(self.path, *args)


@pytest.fixture
def checkout_repo(tmp_path):
    """Factory: checkout_repo({path: content}) -> CheckoutRepo with those files on origin/main."""

    def make(base_files: dict[str, str], base_branch: str = "main") -> CheckoutRepo:
        repo = CheckoutRepo(tmp_path)
        repo.upstream.mkdir()
        _git(repo.upstream, "init", "-q", "-b", base_branch)
        for rel, content in base_files.items():
            p = repo.upstream / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        if not base_files:
            (repo.upstream / "README.md").write_text("x")
        _git(repo.upstream, "add", "-A")
        _git(repo.upstream, "commit", "-q", "-m", "base")

        _git(tmp_path, "clone", "-q", str(repo.upstream), str(repo.path))
        # Mimic actions/checkout on pull_request: detached HEAD, no local base branch
        _git(repo.path, "checkout", "-q", "--detach")
        _git(repo.path, "branch", "-q", "-D", base_branch)
        return repo

    return make
