"""Shared plumbing: which Mnemoverse domain a call reads and writes, the SDK clients every call goes
through, time-boxed calls, and SDK errors turned into sentences a person can act on."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Mapping
from typing import Any, Protocol, TypeVar
from uuid import UUID

from langchain_core.documents import Document
from mnemoverse import (
    AsyncMnemoClient,
    FeedbackResponse,
    MemoryItem,
    MnemoAuthError,
    MnemoClient,
    MnemoError,
    MnemoRateLimitError,
    MnemoUnavailableError,
    ReadResponse,
    WriteResponse,
)

logger = logging.getLogger("langchain_mnemoverse")

T = TypeVar("T")

# API limits (https://mnemoverse.com/docs/api/reference): a read query of 1-5,000 characters,
# a memory of 1-10,000 characters, a domain of up to 100 characters, top_k of 1-500.
MAX_QUERY = 5000
MAX_CONTENT = 10000
MAX_DOMAIN = 100
MAX_TOP_K = 500

CONFIG_KEY = "mnemoverse_domain"
"""The key a run names its domain under: ``config["configurable"]`` or the agent's ``context``."""

DEFAULT_BASE_URL = "https://core.mnemoverse.com"
KEYS_PAGE = "https://console.mnemoverse.com/dashboard/keys"
PLATFORM_METADATA: dict[str, Any] = {"platform": "langchain", "role": "user"}


class NoDomainError(ValueError):
    """No fixed domain, and the run names none in its config or context."""


def check_domain(domain: object) -> str:
    """A domain stripped of surrounding space: non-empty and at most 100 characters."""
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string")
    name = domain.strip()
    if len(name) > MAX_DOMAIN:
        raise ValueError(f"domain must be at most {MAX_DOMAIN} characters, got {len(name)}")
    return name


def check_top_k(value: int, name: str = "k") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_TOP_K:
        raise ValueError(f"{name} must be an integer from 1 to {MAX_TOP_K}")
    return value


def _lookup(source: object, key: str) -> object:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def resolve_domain(fixed: str | None, config: Mapping[str, Any] | None, context: object = None) -> str:
    """The fixed domain; else ``context["mnemoverse_domain"]`` (a mapping key or an attribute);
    else ``config["configurable"]["mnemoverse_domain"]``. Never a default: with none of them,
    :class:`NoDomainError`."""
    if fixed is not None:
        return fixed
    value = _lookup(context, CONFIG_KEY)
    if value is None and config is not None:
        value = _lookup(config.get("configurable"), CONFIG_KEY)
    if value is None:
        raise NoDomainError(
            "no Mnemoverse domain: pass domain= for a single-user app, or set "
            f"config['configurable']['{CONFIG_KEY}'] per run on the server"
        )
    return check_domain(value)


class AsyncMemory(Protocol):
    """The part of ``AsyncMnemoClient`` this package calls."""

    async def read(
        self, query: str, *, top_k: int = ..., domain: str | None = ..., min_relevance: float = ...
    ) -> ReadResponse: ...

    async def write(
        self, content: str, *, domain: str = ..., metadata: dict[str, Any] | None = ...
    ) -> WriteResponse: ...

    async def feedback(
        self, atom_ids: list[UUID | str], outcome: float, *, domain: str = ...
    ) -> FeedbackResponse: ...


class _SyncInExecutor:
    """Async calls served by a sync ``MnemoClient`` in the event loop's default executor; used
    only when the caller passed a sync client and no async one."""

    def __init__(self, client: MnemoClient) -> None:
        self._client = client

    async def read(
        self, query: str, *, top_k: int = 10, domain: str | None = None, min_relevance: float = 0.3
    ) -> ReadResponse:
        call = functools.partial(
            self._client.read, query, top_k=top_k, domain=domain, min_relevance=min_relevance
        )
        return await asyncio.get_running_loop().run_in_executor(None, call)

    async def write(
        self, content: str, *, domain: str = "general", metadata: dict[str, Any] | None = None
    ) -> WriteResponse:
        call = functools.partial(self._client.write, content, domain=domain, metadata=metadata)
        return await asyncio.get_running_loop().run_in_executor(None, call)

    async def feedback(
        self, atom_ids: list[UUID | str], outcome: float, *, domain: str = "general"
    ) -> FeedbackResponse:
        call = functools.partial(self._client.feedback, atom_ids, outcome, domain=domain)
        return await asyncio.get_running_loop().run_in_executor(None, call)


class Clients:
    """One sync and one async SDK client: the ones passed in, or opened from the key.

    With neither client passed, an ``AsyncMnemoClient`` is opened right away, so a missing key
    fails at construction (the SDK raises ``ValueError``) rather than at the first call. The
    sync client is opened on first sync use. Clients passed in are never closed here; clients
    opened here stay open until the process exits.
    """

    def __init__(
        self,
        client: MnemoClient | None,
        async_client: AsyncMnemoClient | None,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._sync = client
        self._api_key = api_key
        self._base_url = base_url
        self._async: AsyncMemory | None = async_client
        self._opened_async: AsyncMnemoClient | None = None
        if client is None and async_client is None:
            # MNEMOVERSE_API_KEY applies when no key is passed; the SDK raises ValueError here
            # when neither is set.
            self._opened_async = AsyncMnemoClient(api_key=api_key, base_url=base_url)
            self._async = self._opened_async

    def aio_if_open(self) -> AsyncMnemoClient | None:
        """The async client opened here at construction, if any (never one passed in)."""
        return self._opened_async

    @property
    def sync(self) -> MnemoClient:
        if self._sync is None:
            try:
                self._sync = MnemoClient(api_key=self._api_key, base_url=self._base_url)
            except ValueError as exc:
                raise ValueError(
                    "only an async client was passed (async_client=) and no API key is set, so "
                    "sync calls have no client: pass client=MnemoClient(...) too, set "
                    "MNEMOVERSE_API_KEY, or use the async methods"
                ) from exc
        return self._sync

    @property
    def aio(self) -> AsyncMemory:
        if self._async is None:
            # Only a sync client was passed: serve async calls with it in an executor thread.
            self._async = _SyncInExecutor(self.sync)
        return self._async


async def within(call: Awaitable[T], timeout: float, what: str) -> T:
    """Wait for an SDK call at most ``timeout`` seconds without cancelling it.

    The call runs on to the SDK's own timeout and reports its outcome to the client's circuit
    breaker; cancelling it instead could leave the breaker stuck half-open, after which every
    call on that client fails. A late failure is logged.
    """
    task = asyncio.ensure_future(call)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout)
    except TimeoutError:
        task.add_done_callback(functools.partial(log_late_failure, what))
        raise


def log_late_failure(what: str, future: asyncio.Future[Any] | Any) -> None:
    """Done-callback for a call that ran past its budget: log it if it failed."""
    if future.cancelled():
        return
    exc = future.exception()
    if exc is not None:
        logger.warning(
            "Mnemoverse: the %s that ran past its budget later failed: %s", what, describe_error(exc)
        )


def _server_says(exc: MnemoError) -> str:
    return (getattr(exc, "message", "") or str(exc)).strip().rstrip(".")


def describe_error(exc: BaseException) -> str:
    """One actionable message for an SDK failure, with the server's own reason when it gave one."""
    if isinstance(exc, MnemoAuthError):
        status = exc.status if exc.status in (401, 403) else "401 or 403"
        return (
            f"Mnemoverse refused the call ({status}): {_server_says(exc)}. A missing, mistyped or "
            f"revoked key is fixed at {KEYS_PAGE}; a 403 can also mean this key may not use this "
            "domain, such as a room it is not a member of."
        )
    if isinstance(exc, MnemoRateLimitError):
        if exc.retryable is False:
            return f"Mnemoverse refused the call (429): {_server_says(exc)}."
        wait = f" Try again in {exc.retry_after:.0f} s." if exc.retry_after else ""
        return f"Mnemoverse is rate limiting this account right now (429).{wait}"
    if isinstance(exc, MnemoUnavailableError):
        return "Mnemoverse could not be reached. Try again shortly."
    if isinstance(exc, MnemoError):
        status = f" ({exc.status})" if exc.status else ""
        return f"Mnemoverse error{status}: {_server_says(exc)}."
    return f"Mnemoverse call failed: {exc!r}"


# "<" and ">" become the single guillemets U+2039 and U+203A.
_ANGLE_BRACKETS = str.maketrans({"<": chr(0x2039), ">": chr(0x203A)})


def one_line(text: str) -> str:
    """Memory text as it is shown to a model: one line, angle brackets replaced by single
    guillemets (U+2039, U+203A).

    Memory text is written by other people and other tools (shared rooms, every AI tool on the
    account, capture). Kept on one line, it cannot start a line of its own that looks like
    another memory (``- [<id>] ...``) or a heading; without ``<`` and ``>`` it cannot close
    the ``<mnemoverse_memory>`` block it is shown in, in any spelling.
    """
    return " ".join(text.split()).translate(_ANGLE_BRACKETS)


def memory_lines(items: list[MemoryItem], limit: int) -> list[str]:
    """One line per recalled memory with its id, at most ``limit``; empty memories skipped.

    Each memory's text goes through :func:`one_line`, so the ids on these lines are exactly
    the ids the server returned, one per line."""
    lines: list[str] = []
    for item in items:
        if len(lines) >= limit:
            break
        content = one_line(item.content or "")
        if content:
            lines.append(f"- [{item.atom_id}] {content}")
    return lines


def to_document(item: MemoryItem) -> Document:
    """A recalled memory as a LangChain ``Document``; the atom id is the document id."""
    atom_id = str(item.atom_id)
    return Document(
        id=atom_id,
        page_content=item.content,
        metadata={
            "atom_id": atom_id,
            "relevance": item.relevance,
            "similarity": item.similarity,
            "valence": item.valence,
            "importance": item.importance,
            "domain": item.domain,
            "concepts": list(item.concepts),
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "source": item.source,
            "provenance": item.provenance.model_dump(exclude_none=True) if item.provenance else None,
            "metadata": dict(item.metadata),
            "superseded_by": str(item.superseded_by) if item.superseded_by else None,
        },
    )
