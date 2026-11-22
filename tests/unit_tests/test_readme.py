"""README statements that can be checked against the code. The README is also the PyPI page."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_mnemoverse._common import MAX_CONTENT, MAX_DOMAIN, MAX_QUERY

README = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")


def test_every_link_is_absolute() -> None:
    # On PyPI a relative link (examples/x.py, LICENSE, #settings) resolves against pypi.org.
    targets = re.findall(r"\]\(([^)\s]+)\)", README)
    assert targets
    relative = [t for t in targets if not t.startswith("https://")]
    assert relative == []


def test_the_limits_it_states_are_the_ones_the_code_applies() -> None:
    assert f"last {MAX_QUERY:,} characters" in README
    assert f"over {MAX_CONTENT:,} characters" in README
    assert f"up to {MAX_DOMAIN} characters" in README


def test_cost_counts_queries_not_requests() -> None:
    # The SDK retries failed calls, so a call can be several HTTP requests; the plan counts
    # only successful ones (docs/api/pricing.md).
    cost = README.split("## Cost", 1)[1].split("\n## ", 1)[0]
    assert "at most one request" not in cost
    assert "counts as at most one query against your plan" in cost
