"""LangChain's standard test suites, run against the fake clients.

The integration suites (``RetrieversIntegrationTests``, ``ToolsIntegrationTests``) need no network
here because every call goes to a fake; ``tests/integration_tests`` runs them against the live API.
"""

from __future__ import annotations

from typing import Any

from conftest import AsyncFakeMnemo, FakeMnemo, atom_id
from langchain_tests.integration_tests import RetrieversIntegrationTests, ToolsIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_mnemoverse import (
    MnemoverseRateTool,
    MnemoverseRetriever,
    MnemoverseSaveTool,
    MnemoverseSearchTool,
)

ENV = ({"MNEMOVERSE_API_KEY": "mk_live_test"}, {"domain": "user:ana"}, {"domain": "user:ana"})


def _params() -> dict[str, Any]:
    return {"domain": "user:ana", "client": FakeMnemo(), "async_client": AsyncFakeMnemo()}


class TestRetrieverStandard(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[MnemoverseRetriever]:
        return MnemoverseRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def retriever_query_example(self) -> str:
        return "Which seat does Ana take?"


class TestSearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[MnemoverseSearchTool]:
        return MnemoverseSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "seat preference", "limit": 2}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return ENV


class TestSearchToolStandard(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[MnemoverseSearchTool]:
        return MnemoverseSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "seat preference", "limit": 2}


class TestSaveToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[MnemoverseSaveTool]:
        return MnemoverseSaveTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"content": "I take the window seat."}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return ENV


class TestSaveToolStandard(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[MnemoverseSaveTool]:
        return MnemoverseSaveTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"content": "I take the window seat."}


class TestRateToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[MnemoverseRateTool]:
        return MnemoverseRateTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"memory_ids": [str(atom_id(0))], "helpful": True}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return ENV


class TestRateToolStandard(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[MnemoverseRateTool]:
        return MnemoverseRateTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _params()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"memory_ids": [str(atom_id(0))], "helpful": True}
