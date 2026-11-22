"""LangChain's standard integration suites against the live Mnemoverse API.

Skipped unless both are set:

    MNEMOVERSE_API_KEY=mk_live_...   a key of a dedicated test account
    MNEMOVERSE_LIVE_TESTS=1

Run with ``python -m pytest tests/integration_tests -q``. Each run writes three memories into a
fresh domain ``test:langchain-mnemoverse:<uuid4>`` and cannot remove them (the API has no
delete), and the save-tool suite writes four more. A run makes 21 requests (3 seed writes,
6 retriever reads, 12 tool calls), all counted against the account's daily limit.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_tests.integration_tests import RetrieversIntegrationTests, ToolsIntegrationTests

LIVE = bool(os.environ.get("MNEMOVERSE_API_KEY")) and os.environ.get("MNEMOVERSE_LIVE_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE, reason="live API tests: set MNEMOVERSE_API_KEY and MNEMOVERSE_LIVE_TESTS=1 to run them"
)

DOMAIN = f"test:langchain-mnemoverse:{uuid.uuid4()}"
SEED = [
    "Ana takes the window seat on every flight.",
    "Ana drinks tea, not coffee, in the morning.",
    "Ana's team settled on Postgres for the ledger service.",
]
SEEDED_IDS: list[str] = []


@pytest.fixture(scope="module", autouse=True)
def seeded() -> Iterator[None]:
    from mnemoverse import MnemoClient

    with MnemoClient() as client:
        for content in SEED:
            result = client.write(content, domain=DOMAIN, metadata={"platform": "langchain", "role": "test"})
            assert result.stored, result.reason
            SEEDED_IDS.append(str(result.atom_id))
    yield


def _retriever_params() -> dict[str, Any]:
    return {"domain": DOMAIN, "min_relevance": 0.0}


class TestLiveRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[Any]:
        from langchain_mnemoverse import MnemoverseRetriever

        return MnemoverseRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return _retriever_params()

    @property
    def retriever_query_example(self) -> str:
        return "What does Ana prefer?"


class TestLiveSearchTool(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[Any]:
        from langchain_mnemoverse import MnemoverseSearchTool

        return MnemoverseSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"domain": DOMAIN}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "Which seat does Ana take?", "limit": 2}


class TestLiveSaveTool(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[Any]:
        from langchain_mnemoverse import MnemoverseSaveTool

        return MnemoverseSaveTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"domain": DOMAIN}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"content": "Ana prefers aisle seats on night trains."}


class TestLiveRateTool(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[Any]:
        from langchain_mnemoverse import MnemoverseRateTool

        return MnemoverseRateTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"domain": DOMAIN}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"memory_ids": SEEDED_IDS[:1], "helpful": True}
