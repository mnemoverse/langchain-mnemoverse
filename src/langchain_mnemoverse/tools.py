"""Mnemoverse as LangChain tools: ``mnemoverse_search``, ``mnemoverse_save``, ``mnemoverse_rate``,
and :class:`MnemoverseToolkit`, which builds them around one pair of SDK clients."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from mnemoverse import AsyncMnemoClient, FeedbackResponse, MnemoClient, ReadResponse, WriteResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, field_validator

from ._common import (
    DEFAULT_BASE_URL,
    MAX_CONTENT,
    MAX_QUERY,
    PLATFORM_METADATA,
    Clients,
    check_domain,
    describe_error,
    memory_lines,
    resolve_domain,
)

SEARCH_DESCRIPTION = "Search the user's long-term Mnemoverse memory. Returns memories, each with its id."
SEARCH_RATE_HINT = " Pass an id to mnemoverse_rate once you know whether the memory helped."
SAVE_DESCRIPTION = (
    "Save something to the user's Mnemoverse memory. Call this only when the user asks you to "
    "remember something or plainly means it; never save your own conclusions."
)
RATE_DESCRIPTION = (
    "Rate recalled memories as helpful or misleading. The rating changes what Mnemoverse recalls "
    "next for similar questions."
)


class _MnemoverseTool(BaseTool):
    """Shared settings and plumbing of the three tools.

    Each call reads or writes one domain: ``domain`` when it is set, otherwise
    ``config["configurable"]["mnemoverse_domain"]`` of the run. With neither, the call fails
    closed: the model gets an error message and nothing is read or written. The domain is never
    an argument the model can set.

    Every failure reaches the model as a ``ToolMessage`` with ``status="error"`` and the
    server's reason (``handle_tool_error=True``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle_tool_error: bool = True
    domain: str | None = None
    """The domain every call uses. Leave it out in an app with several users and set
    ``config["configurable"]["mnemoverse_domain"]`` per run instead."""
    api_key: SecretStr | None = None
    """A Mnemoverse API key. Falls back to the ``MNEMOVERSE_API_KEY`` environment variable."""
    base_url: str = DEFAULT_BASE_URL
    client: MnemoClient | None = Field(default=None, exclude=True)
    """A ready sync client, used by ``invoke``. Never closed by the tool."""
    async_client: AsyncMnemoClient | None = Field(default=None, exclude=True)
    """A ready async client, used by ``ainvoke``. Never closed by the tool."""

    _clients: Clients = PrivateAttr()

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, value: str | None) -> str | None:
        return None if value is None else check_domain(value)

    def model_post_init(self, context: Any, /) -> None:
        key = self.api_key.get_secret_value() if self.api_key else None
        self._clients = Clients(self.client, self.async_client, key, self.base_url)

    def _domain(self, config: RunnableConfig | None) -> str:
        try:
            return resolve_domain(self.domain, config)
        except ValueError as exc:
            raise ToolException(f"Mnemoverse is not available: {exc}.") from exc


def _failed(exc: Exception) -> ToolException:
    return ToolException(describe_error(exc))


class SearchInput(BaseModel):
    """Arguments of ``mnemoverse_search``."""

    query: str = Field(description="What to look for, in natural language.")
    limit: int = Field(default=5, ge=1, le=50, description="How many memories, 1-50.")


class MnemoverseSearchTool(_MnemoverseTool):
    """Search the user's Mnemoverse memory; returns one line per memory with its id.

    .. code-block:: python

        from langchain_mnemoverse import MnemoverseSearchTool

        tool = MnemoverseSearchTool(domain="user:ana")
        tool.invoke({"query": "seat preference", "limit": 3})
    """

    name: str = "mnemoverse_search"
    description: str = SEARCH_DESCRIPTION + SEARCH_RATE_HINT
    args_schema: type[BaseModel] = SearchInput

    @staticmethod
    def _answer(result: ReadResponse, limit: int) -> str:
        lines = memory_lines(list(result.items), limit)
        if not lines:
            return "No memories in the user's Mnemoverse memory matched."
        return "\n".join(["Memories from the user's Mnemoverse memory:", *lines])

    def _run(
        self,
        query: str,
        limit: int = 5,
        *,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        domain = self._domain(config)
        try:
            result = self._clients.sync.read(query[-MAX_QUERY:], top_k=limit, domain=domain)
        except Exception as exc:  # the model gets the error as a ToolMessage
            raise _failed(exc) from exc
        return self._answer(result, limit)

    async def _arun(
        self,
        query: str,
        limit: int = 5,
        *,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        domain = self._domain(config)
        try:
            result = await self._clients.aio.read(query[-MAX_QUERY:], top_k=limit, domain=domain)
        except Exception as exc:  # the model gets the error as a ToolMessage
            raise _failed(exc) from exc
        return self._answer(result, limit)


class SaveInput(BaseModel):
    """Arguments of ``mnemoverse_save``."""

    content: str = Field(description="The fact to keep, in the user's own words. One fact per call.")


class MnemoverseSaveTool(_MnemoverseTool):
    """Save one memory to the user's Mnemoverse memory.

    A memory holds up to 10,000 characters; a longer one is refused before any request. Saved
    memories carry ``metadata={"platform": "langchain", "role": "user"}``.
    """

    name: str = "mnemoverse_save"
    description: str = SAVE_DESCRIPTION
    args_schema: type[BaseModel] = SaveInput

    @staticmethod
    def _check(content: str) -> None:
        if len(content) > MAX_CONTENT:
            raise ToolException(
                f"Mnemoverse did not store it: a memory holds up to {MAX_CONTENT:,} characters and "
                f"this one has {len(content):,}. Save the facts that matter, one per call."
            )

    @staticmethod
    def _answer(result: WriteResponse) -> str:
        if not result.stored:
            reason = (result.reason or "").strip().rstrip(".")
            return "Mnemoverse did not store it" + (f": {reason}." if reason else ".")
        return f"Saved to Mnemoverse (id {result.atom_id}); it can be recalled now."

    def _run(
        self,
        content: str,
        *,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        self._check(content)
        domain = self._domain(config)
        try:
            result = self._clients.sync.write(content, domain=domain, metadata=dict(PLATFORM_METADATA))
        except Exception as exc:  # the model gets the error as a ToolMessage
            raise _failed(exc) from exc
        return self._answer(result)

    async def _arun(
        self,
        content: str,
        *,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        self._check(content)
        domain = self._domain(config)
        try:
            result = await self._clients.aio.write(content, domain=domain, metadata=dict(PLATFORM_METADATA))
        except Exception as exc:  # the model gets the error as a ToolMessage
            raise _failed(exc) from exc
        return self._answer(result)


class RateInput(BaseModel):
    """Arguments of ``mnemoverse_rate``."""

    memory_ids: list[str] = Field(description="Ids of recalled memories, from mnemoverse_search.")
    helpful: bool = Field(description="True if the memories helped answer, False if they misled.")


class MnemoverseRateTool(_MnemoverseTool):
    """Rate recalled memories as helpful (outcome 1.0) or misleading (outcome -1.0).

    The model passes the ids it got from ``mnemoverse_search`` (or from the middleware's
    memory block). The tool keeps no state between calls, so one instance can serve many users.
    """

    name: str = "mnemoverse_rate"
    description: str = RATE_DESCRIPTION
    args_schema: type[BaseModel] = RateInput

    @staticmethod
    def _ids(memory_ids: list[str]) -> list[UUID | str]:
        ids: list[UUID | str] = [i.strip() for i in memory_ids if i.strip()]
        if not ids:
            raise ToolException("No memories to rate: pass the ids from mnemoverse_search.")
        return ids

    @staticmethod
    def _answer(result: FeedbackResponse, count: int, helpful: bool) -> str:
        word = "helpful" if helpful else "misleading"
        return f"Rated {result.updated_count} of {count} memories as {word}."

    def _run(
        self,
        memory_ids: list[str],
        helpful: bool,
        *,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        ids = self._ids(memory_ids)
        domain = self._domain(config)
        try:
            result = self._clients.sync.feedback(ids, 1.0 if helpful else -1.0, domain=domain)
        except Exception as exc:  # the model gets the error as a ToolMessage
            raise _failed(exc) from exc
        return self._answer(result, len(ids), helpful)

    async def _arun(
        self,
        memory_ids: list[str],
        helpful: bool,
        *,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        ids = self._ids(memory_ids)
        domain = self._domain(config)
        try:
            result = await self._clients.aio.feedback(ids, 1.0 if helpful else -1.0, domain=domain)
        except Exception as exc:  # the model gets the error as a ToolMessage
            raise _failed(exc) from exc
        return self._answer(result, len(ids), helpful)


class MnemoverseToolkit(BaseToolkit):
    """The three Mnemoverse tools around one pair of SDK clients.

    .. code-block:: python

        from langchain_mnemoverse import MnemoverseToolkit

        tools = MnemoverseToolkit(domain="user:ana").get_tools()
        # [mnemoverse_search, mnemoverse_save, mnemoverse_rate]

    ``save=False`` leaves out ``mnemoverse_save``, ``rate=False`` leaves out ``mnemoverse_rate``.
    There is no delete tool.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: str | None = None
    """The domain every tool call uses; see :class:`MnemoverseSearchTool` for the per-run form."""
    api_key: SecretStr | None = None
    base_url: str = DEFAULT_BASE_URL
    client: MnemoClient | None = Field(default=None, exclude=True)
    async_client: AsyncMnemoClient | None = Field(default=None, exclude=True)
    save: bool = True
    """Include ``mnemoverse_save``."""
    rate: bool = True
    """Include ``mnemoverse_rate``."""

    _clients: Clients = PrivateAttr()

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, value: str | None) -> str | None:
        return None if value is None else check_domain(value)

    def model_post_init(self, context: Any, /) -> None:
        key = self.api_key.get_secret_value() if self.api_key else None
        self._clients = Clients(self.client, self.async_client, key, self.base_url)

    def get_tools(self) -> list[BaseTool]:
        """``mnemoverse_search``, then ``mnemoverse_save`` and ``mnemoverse_rate`` unless switched
        off. Every tool shares this toolkit's clients."""
        settings: dict[str, Any] = {
            "domain": self.domain,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "client": self.client,
            "async_client": self.async_client,
        }
        description = SEARCH_DESCRIPTION + (SEARCH_RATE_HINT if self.rate else "")
        tools: list[_MnemoverseTool] = [MnemoverseSearchTool(description=description, **settings)]
        if self.save:
            tools.append(MnemoverseSaveTool(**settings))
        if self.rate:
            tools.append(MnemoverseRateTool(**settings))
        for tool in tools:
            tool._clients = self._clients  # exactly the toolkit's clients
        return list(tools)


__all__ = [
    "MnemoverseRateTool",
    "MnemoverseSaveTool",
    "MnemoverseSearchTool",
    "MnemoverseToolkit",
]
