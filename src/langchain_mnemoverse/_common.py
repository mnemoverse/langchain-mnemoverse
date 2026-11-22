"""Shared plumbing: which Mnemoverse domain a call reads and writes, the SDK clients every call goes
through, time-boxed calls, and SDK errors turned into sentences a person can act on."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import threading
import weakref
from collections.abc import Awaitable, Callable, Mapping
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


class DomainConflictError(ValueError):
    """The run's context and its config name two different domains."""


def check_domain(domain: object) -> str:
    """A domain stripped of surrounding space: non-empty and at most 100 characters."""
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain must be a non-empty string")
    name = domain.strip()
    if len(name) > MAX_DOMAIN:
        raise ValueError(f"domain must be at most {MAX_DOMAIN} characters, got {len(name)}")
    return name


def cut_query(query: str) -> str:
    """A query as it is sent: stripped, then cut to its last 5,000 characters. Empty when the
    query is blank; callers send no request then."""
    return query.strip()[-MAX_QUERY:].strip()


def domain_ref(domain: str) -> str:
    """How a domain appears in logs: ``#`` and the first 12 hex digits of its SHA-256. Domains
    often carry a user id (``user:<id>``, sometimes an email), which does not belong in logs."""
    return "#" + hashlib.sha256(domain.encode()).hexdigest()[:12]


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
    """The domain a call uses: the fixed one; else the one the run names in its ``context``
    (``context["mnemoverse_domain"]``, a mapping key or an attribute) or its
    ``config["configurable"]["mnemoverse_domain"]``.

    Never a default: with none of them, :class:`NoDomainError`. When the context and the config
    both name one and they differ, :class:`DomainConflictError`: the run is ambiguous, and
    picking either could read one user's memory and write another's.
    """
    if fixed is not None:
        return fixed
    from_context = _lookup(context, CONFIG_KEY)
    from_config = _lookup(config.get("configurable"), CONFIG_KEY) if config is not None else None
    if from_context is None and from_config is None:
        raise NoDomainError(
            "no Mnemoverse domain: pass domain= for a single-user app, or name it per run in "
            f"config['configurable']['{CONFIG_KEY}'] or the agent's context"
        )
    if from_context is not None and from_config is not None:
        name = check_domain(from_context)
        if name != check_domain(from_config):
            raise DomainConflictError(
                f"the run's context and its config['configurable'] name different values for "
                f"'{CONFIG_KEY}'; name the domain in one place"
            )
        return name
    return check_domain(from_context if from_context is not None else from_config)


@functools.cache
def _runtime_getter() -> Callable[[], object] | None:
    try:
        from langgraph.runtime import get_runtime
    except ImportError:  # the base install has no LangGraph, and so no runs with a context
        return None
    return get_runtime


def run_context() -> object:
    """The ``context`` of the LangGraph run the caller is part of (``create_agent(...).invoke(...,
    context=...)``); ``None`` outside a run or without LangGraph."""
    get_runtime = _runtime_getter()
    if get_runtime is None:
        return None
    try:
        runtime = get_runtime()
    except (RuntimeError, KeyError):  # not inside a runnable, or a config without the runtime
        return None
    return getattr(runtime, "context", None)


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


API_KEY_ENV = "MNEMOVERSE_API_KEY"

ONLY_ASYNC_CLIENT = (
    "only an async client was passed (async_client=), so sync calls have no client: pass "
    "client=MnemoClient(...) too, pass api_key= to let sync calls open their own client, or use "
    "the async methods (ainvoke)"
)


class NoSyncClientError(ValueError):
    """A sync call on a component that was given only an async client and no ``api_key``."""


class OpenedClients:
    """The SDK clients this package opens from one key for one address.

    Every component built with the same key and address shares one instance (see
    :func:`open_clients`), so copies of a component (``configurable_fields`` builds one per
    call) and the tools of a toolkit do not open clients of their own.

    * One sync ``MnemoClient``, opened on first sync use, under a lock: threads that arrive
      together open one client, not one each.
    * One ``AsyncMnemoClient`` per running event loop. SDK 0.5.2 keeps one HTTP connection pool
      per async client, bound to the event loop that first used it; carried into a second loop
      (a second ``asyncio.run``), it fails with "Event loop is closed".
    """

    def __init__(self, key: str, base_url: str) -> None:
        self._key = key
        self._base_url = base_url
        self._lock = threading.Lock()
        self._sync: MnemoClient | None = None
        self._by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncMnemoClient] = (
            weakref.WeakKeyDictionary()
        )

    def sync(self) -> MnemoClient:
        """The shared sync client."""
        with self._lock:
            if self._sync is None:
                self._sync = MnemoClient(api_key=self._key, base_url=self._base_url)
            return self._sync

    def own_sync(self) -> MnemoClient:
        """A new sync client that nobody else uses."""
        return MnemoClient(api_key=self._key, base_url=self._base_url)

    def aio(self) -> AsyncMnemoClient:
        """The async client of the running event loop."""
        loop = asyncio.get_running_loop()
        with self._lock:
            client = self._by_loop.get(loop)
            if client is None:
                client = AsyncMnemoClient(api_key=self._key, base_url=self._base_url)
                self._by_loop[loop] = client
            return client


# Live only while a component holds them: when the last component built with a key is gone,
# its clients go too.
_OPENED: weakref.WeakValueDictionary[tuple[str, str], OpenedClients] = weakref.WeakValueDictionary()
_OPENED_LOCK = threading.Lock()


def open_clients(api_key: str | None, base_url: str) -> OpenedClients:
    """The clients opened for ``api_key`` (else ``MNEMOVERSE_API_KEY``) and ``base_url``.

    Raises ``ValueError`` when there is no key, at construction rather than at the first call.
    """
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(f"No API key: pass api_key= or set the {API_KEY_ENV} environment variable")
    # A digest, so the key itself is not a dictionary key anybody can list.
    cache_key = (hashlib.sha256(key.encode()).hexdigest(), base_url.rstrip("/"))
    with _OPENED_LOCK:
        opened = _OPENED.get(cache_key)
        if opened is None:
            opened = OpenedClients(key, base_url)
            _OPENED[cache_key] = opened
        return opened


class Clients:
    """The clients one component calls: the ones passed in, or ones opened from the key.

    * ``client`` and ``async_client`` passed in are used as given and never closed here.
    * Neither passed: the clients are opened from ``api_key``, else ``MNEMOVERSE_API_KEY``; with
      neither, ``ValueError`` at construction.
    * Only ``client`` passed: async calls run it in the event loop's default executor.
    * Only ``async_client`` passed: sync calls open their own client only from an ``api_key``
      passed explicitly, at ``base_url``; otherwise they raise :class:`NoSyncClientError`. They
      never fall back to ``MNEMOVERSE_API_KEY``, which may belong to another account than the
      client passed in.
    """

    def __init__(
        self,
        client: MnemoClient | None,
        async_client: AsyncMnemoClient | None,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._sync = client
        self._async: AsyncMemory | None = async_client
        self.opened: OpenedClients | None = None
        if client is None and async_client is None:
            self.opened = open_clients(api_key, base_url)
        elif client is None and api_key:
            self.opened = open_clients(api_key, base_url)

    @property
    def sync(self) -> MnemoClient:
        if self._sync is not None:
            return self._sync
        if self.opened is None:
            raise NoSyncClientError(ONLY_ASYNC_CLIENT)
        return self.opened.sync()

    @property
    def aio(self) -> AsyncMemory:
        if self._async is not None:
            return self._async
        if self.opened is not None:
            return self.opened.aio()
        assert self._sync is not None  # one of the three is always there
        return _SyncInExecutor(self._sync)


async def within(call: Awaitable[T], timeout: float, what: str) -> T:
    """Wait for an SDK call at most ``timeout`` seconds without cancelling it.

    The call runs on to the SDK's own timeout and reports its outcome to the client's circuit
    breaker; cancelling it instead could leave the breaker stuck half-open, after which every
    call on that client fails. It goes on only while its event loop runs: ``asyncio.run()``
    cancels it when it returns. A late failure or cancellation is logged.
    """
    task = asyncio.ensure_future(call)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout)
    except TimeoutError:
        task.add_done_callback(functools.partial(log_late_failure, what))
        raise


def log_late_failure(what: str, future: asyncio.Future[Any] | Any) -> None:
    """Done-callback for a call that ran past its budget: log it if it failed or was cancelled."""
    if future.cancelled():
        logger.warning(
            "Mnemoverse: the %s that ran past its budget was cancelled before it finished, "
            "usually because its event loop closed (asyncio.run returned); it did not complete.",
            what,
        )
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
