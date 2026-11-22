"""pyproject.toml and the workflows agree: the CI lowest leg pins every floor this package declares."""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


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
