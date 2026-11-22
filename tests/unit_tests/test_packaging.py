"""pyproject.toml and the workflows agree: the CI lowest leg pins every floor this package declares."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

# The workflows live in the repository; the sdist ships the tests without them.
pytestmark = pytest.mark.skipif(
    not (ROOT / ".github" / "workflows").is_dir(), reason="no .github/workflows (an unpacked sdist)"
)


def workflow(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))
    return data


def floors() -> dict[str, str]:
    """``{package: lowest version}`` from the runtime dependencies and the agents extra."""
    found = {}
    for spec in [
        *PYPROJECT["project"]["dependencies"],
        *PYPROJECT["project"]["optional-dependencies"]["agents"],
    ]:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)>=([0-9.]+),<[0-9.]+", spec)
        assert match, f"every dependency has a floor and a ceiling: {spec}"
        found[match.group(1)] = match.group(2)
    return found


def test_the_lowest_leg_pins_every_floor() -> None:
    steps = workflow("ci.yml")["jobs"]["test"]["steps"]
    pin = next(s for s in steps if s.get("name") == "Pin the lowest supported versions")
    pinned = dict(arg.split("==") for arg in shlex.split(pin["run"]) if "==" in arg)
    assert pinned == floors()


def test_pytest_plugins_the_workflows_rely_on_are_direct_dev_dependencies() -> None:
    dev = {
        re.split(r"[<>=!~ ]", spec, maxsplit=1)[0]
        for spec in PYPROJECT["project"]["optional-dependencies"]["dev"]
    }
    texts = [
        (ROOT / ".github" / "workflows" / n).read_text(encoding="utf-8") for n in ("ci.yml", "release.yml")
    ]
    if any("--disable-socket" in text for text in texts):
        assert "pytest-socket" in dev  # not only through langchain-tests
    assert {"pytest", "pytest-asyncio"} <= dev


# -- release.yml -------------------------------------------------------------------------------


def _release_step(name: str) -> str:
    steps = [s for job in workflow("release.yml")["jobs"].values() for s in job.get("steps", [])]
    script: str = next(s["run"] for s in steps if s.get("name") == name)
    return script


@pytest.mark.parametrize(
    ("tag", "heading", "passes"),
    [
        ("v0.1.0", "## [0.1.0] - 2026-11-22", True),
        ("v0.1.0", "## [0.1.0] - unreleased", False),
        ("v0.1.0", "## [0.1.0] - TBD", False),
        ("v0.1.0", "## [0.1.0]", False),
        ("v0.1.0", "## [0.1.0] - 2026-11-22 (yanked)", False),
        ("v0.1.0", "## [0x1x0] - 2026-11-22", False),
        ("v0.1.1", "## [0.1.1] - 2026-11-22", False),  # __version__ is 0.1.0
    ],
)
def test_the_release_gate_wants_the_version_and_a_dated_heading(
    tmp_path: Path, tag: str, heading: str, passes: bool
) -> None:
    (tmp_path / "src" / "langchain_mnemoverse").mkdir(parents=True)
    version_file = ROOT / "src" / "langchain_mnemoverse" / "_version.py"
    (tmp_path / "src" / "langchain_mnemoverse" / "_version.py").write_text(version_file.read_text())
    (tmp_path / "CHANGELOG.md").write_text(f"# Changelog\n\n{heading}\n\n### Added\n\n- Something.\n")
    script = _release_step("Tag must match __version__ and a dated CHANGELOG heading")
    env = {"GITHUB_REF_NAME": tag, "PATH": os.environ["PATH"]}
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is passes, result.stdout + result.stderr


def test_the_release_gate_refuses_the_changelog_as_committed() -> None:
    # Until the release PR dates it, the heading says unreleased; a tag now must not publish.
    script = _release_step("Tag must match __version__ and a dated CHANGELOG heading")
    env = {"GITHUB_REF_NAME": "v0.1.0", "PATH": os.environ["PATH"]}
    result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "date the heading first" in result.stdout


def test_the_published_files_come_from_a_job_that_runs_no_dev_dependency() -> None:
    jobs = workflow("release.yml")["jobs"]
    builders = [
        name for name, job in jobs.items() if any("python -m build" in s.get("run", "") for s in job["steps"])
    ]
    assert builders == ["build"]
    runs = [s.get("run", "") for s in jobs["build"]["steps"]]
    installs = [line for run in runs for line in run.splitlines() if "pip install" in line]
    assert installs and all("-e" not in shlex.split(line) and ".[dev]" not in line for line in installs)
    for line in installs:
        packages = [
            a for a in shlex.split(line) if not a.startswith("-") and a not in {"python", "pip", "install"}
        ]
        assert packages and all("==" in p for p in packages), line
    assert not any("pytest" in run or "ruff" in run or "mypy" in run for run in runs)
    publish = jobs["publish"]
    assert set(publish["needs"]) == {"test", "build", "verify"}
    assert [name for name, job in jobs.items() if "id-token" in job.get("permissions", {})] == ["publish"]


@pytest.mark.parametrize("name", ["ci.yml", "release.yml"])
def test_no_checkout_keeps_the_job_token_on_disk(name: str) -> None:
    checkouts = [
        step
        for job in workflow(name)["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert checkouts
    assert all(step.get("with", {}).get("persist-credentials") is False for step in checkouts)
