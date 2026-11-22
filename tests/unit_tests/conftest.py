"""Offline fixtures: fake sync and async Mnemoverse clients that record every call and answer with
the SDK's own response models, and a scripted chat model for ``create_agent``.

The fakes subclass ``MnemoClient`` and ``AsyncMnemoClient`` so the components' type checks
accept them, but never call the SDK constructors: nothing here opens a connection or needs a key.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from mnemoverse import (
    AsyncMnemoClient,
    FeedbackResponse,
    MemoryItem,
    MnemoAuthError,
    MnemoClient,
    MnemoError,
    MnemoRateLimitError,
    MnemoUnavailableError,
    Provenance,
    ReadResponse,
    WriteResponse,
)
from pydantic import Field

import langchain_mnemoverse._common as common

ITEM_COUNT = 8
"""Every read answers with this many memories, more than any test asks for."""

SAVED_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
REPLACEMENT_ID = uuid.UUID("44444444-4444-4444-8444-444444444444")
CREATED = datetime(2026, 11, 1, 12, 30, tzinfo=UTC)


def atom_id(n: int) -> uuid.UUID:
    return uuid.UUID(int=n + 1)


def item(n: int, domain: str = "user:ana", content: str | None = None) -> MemoryItem:
    """Memory ``n``; memory 0 carries every optional field so the mapping can be checked."""
    extra: dict[str, Any] = {}
    if n == 0:
        extra = {
            "created_at": CREATED,
            "provenance": Provenance(agent="claude-code", is_external=False),
            "metadata": {"platform": "test"},
            "superseded_by": REPLACEMENT_ID,
        }
    return MemoryItem(
        atom_id=atom_id(n),
        content=f"memory {n}" if content is None else content,
        relevance=0.9 - n / 100,
        similarity=0.8,
        valence=0.1,
        importance=0.5,
        source="semantic",
        concepts=["seat"] if n == 0 else [],
        domain=domain,
        **extra,
    )


def _read_response(items: list[MemoryItem]) -> ReadResponse:
    return ReadResponse(
        items=items, episodic_hit=False, query_concepts=[], expanded_concepts=[], search_time_ms=1.0
    )


class _Recorder:
    """What both fakes share: the call log, scripted failures and answers."""

    def _setup(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.items: list[MemoryItem] = [item(n) for n in range(ITEM_COUNT)]
        # call name -> error raised on every such call
        self.fail: dict[str, BaseException] = {}
        # (call name, domain) -> error: a refusal that belongs to one domain only
        self.fail_in: dict[tuple[str, str], BaseException] = {}
        self.delay = 0.0
        self.stored = True
        self.closed = False
        self.completed: list[str] = []

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))

    def _finish(self, name: str, kwargs: dict[str, Any]) -> None:
        self.completed.append(name)
        if name in self.fail:
            raise self.fail[name]
        key = (name, kwargs.get("domain"))
        if key in self.fail_in:
            raise self.fail_in[key]

    def named(self, name: str) -> list[dict[str, Any]]:
        return [kw for n, kw in self.calls if n == name]

    def _write_response(self) -> WriteResponse:
        if not self.stored:
            return WriteResponse(stored=False, reason="duplicate of an existing memory")
        return WriteResponse(stored=True, atom_id=SAVED_ID, importance=0.5)


class FakeMnemo(_Recorder, MnemoClient):
    """Stands in for ``MnemoClient``: same method names and keywords, no network."""

    def __init__(self) -> None:  # the SDK constructor is skipped on purpose
        self._setup()

    def _enter(self, name: str, **kwargs: Any) -> None:
        self._record(name, **kwargs)
        if self.delay:
            time.sleep(self.delay)
        self._finish(name, kwargs)

    def read(self, query: str, **kwargs: Any) -> ReadResponse:  # type: ignore[override]
        self._enter("read", query=query, **kwargs)
        return _read_response(list(self.items))

    def write(self, content: str, **kwargs: Any) -> WriteResponse:  # type: ignore[override]
        self._enter("write", content=content, **kwargs)
        return self._write_response()

    def feedback(self, atom_ids: list[Any], outcome: float, **kwargs: Any) -> FeedbackResponse:  # type: ignore[override]
        self._enter("feedback", atom_ids=list(atom_ids), outcome=outcome, **kwargs)
        return FeedbackResponse(updated_count=len(atom_ids), avg_valence=outcome)

    def close(self) -> None:
        self.closed = True


class AsyncFakeMnemo(_Recorder, AsyncMnemoClient):
    """Stands in for ``AsyncMnemoClient``: same method names and keywords, no network."""

    def __init__(self) -> None:  # the SDK constructor is skipped on purpose
        self._setup()

    async def _enter(self, name: str, **kwargs: Any) -> None:
        self._record(name, **kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        self._finish(name, kwargs)

    async def read(self, query: str, **kwargs: Any) -> ReadResponse:  # type: ignore[override]
        await self._enter("read", query=query, **kwargs)
        return _read_response(list(self.items))

    async def write(self, content: str, **kwargs: Any) -> WriteResponse:  # type: ignore[override]
        await self._enter("write", content=content, **kwargs)
        return self._write_response()

    async def feedback(self, atom_ids: list[Any], outcome: float, **kwargs: Any) -> FeedbackResponse:  # type: ignore[override]
        await self._enter("feedback", atom_ids=list(atom_ids), outcome=outcome, **kwargs)
        return FeedbackResponse(updated_count=len(atom_ids), avg_valence=outcome)

    async def close(self) -> None:
        self.closed = True


def _forbidden() -> MnemoAuthError:
    exc = MnemoAuthError("Key is not a member of this room")
    exc.status = 403  # the SDK keeps the status the server sent
    return exc


# (error, a phrase the description must contain)
ERRORS: dict[str, tuple[BaseException, str]] = {
    "auth": (MnemoAuthError("Invalid API key"), "refused the call (401): Invalid API key"),
    "forbidden": (_forbidden(), "refused the call (403): Key is not a member of this room"),
    "rate": (
        MnemoRateLimitError(retry_after=12.0),
        "rate limiting this account right now (429). Try again in 12 s",
    ),
    "quota": (
        MnemoRateLimitError("Daily query limit reached", retryable=False),
        "refused the call (429): Daily query limit reached",
    ),
    "unavailable": (MnemoUnavailableError("circuit open"), "could not be reached"),
    "server": (MnemoError("server error", status=500), "Mnemoverse error (500): server error"),
}


@pytest.fixture(autouse=True)
def _fresh_opened_clients() -> Iterator[None]:
    """Every test starts with no clients opened by the package."""
    common._OPENED.clear()
    yield
    common._OPENED.clear()


class Opened:
    """What the package opened itself: every ``MnemoClient`` and ``AsyncMnemoClient`` it built."""

    def __init__(self) -> None:
        self.sync: list[OpenedSync] = []
        self.aio: list[OpenedAsync] = []


class OpenedSync(FakeMnemo):
    """A fake built where the package would build a ``MnemoClient``; records its arguments."""

    log: Opened

    def __init__(self, api_key: str | None = None, base_url: str = "", **kwargs: Any) -> None:
        super().__init__()
        self.api_key, self.base_url = api_key, base_url
        self.log.sync.append(self)


class OpenedAsync(AsyncFakeMnemo):
    """A fake built where the package would build an ``AsyncMnemoClient``. Like the SDK 0.5.2
    client, it belongs to the event loop that first used it: on another loop it fails."""

    log: Opened

    def __init__(self, api_key: str | None = None, base_url: str = "", **kwargs: Any) -> None:
        super().__init__()
        self.api_key, self.base_url = api_key, base_url
        self.loop: asyncio.AbstractEventLoop | None = None
        self.log.aio.append(self)

    async def _enter(self, name: str, **kwargs: Any) -> None:
        loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = loop
        elif self.loop is not loop:
            raise RuntimeError("Event loop is closed")
        await super()._enter(name, **kwargs)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    """Clients the package opens are fakes, recorded here; ``MNEMOVERSE_API_KEY`` is set."""
    log = Opened()
    monkeypatch.setattr(OpenedSync, "log", log, raising=False)
    monkeypatch.setattr(OpenedAsync, "log", log, raising=False)
    monkeypatch.setattr(common, "MnemoClient", OpenedSync)
    monkeypatch.setattr(common, "AsyncMnemoClient", OpenedAsync)
    monkeypatch.setenv("MNEMOVERSE_API_KEY", "mk_live_env")
    return log


@pytest.fixture
def mem() -> FakeMnemo:
    return FakeMnemo()


@pytest.fixture
def amem() -> AsyncFakeMnemo:
    return AsyncFakeMnemo()


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MNEMOVERSE_API_KEY", raising=False)


class ScriptedChatModel(GenericFakeChatModel):
    """A chat model with no model behind it: replays scripted messages and records every prompt."""

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:  # type: ignore[override]
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def scripted(*replies: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(list(replies) or [AIMessage(content="ok")]), seen=[])


def say(text: str) -> AIMessage:
    return AIMessage(content=text)


def call_tool(name: str, args: dict[str, Any], call_id: str = "t1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def user(text: str) -> dict[str, Any]:
    """Agent input with one user message."""
    return {"messages": [{"role": "user", "content": text}]}
