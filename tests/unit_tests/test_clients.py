"""Which SDK clients a component calls, and how many the package opens itself."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from conftest import AsyncFakeMnemo, FakeMnemo, Opened, OpenedSync, user
from langchain.agents import create_agent
from langchain_core.runnables import ConfigurableField

from langchain_mnemoverse import (
    MnemoverseMiddleware,
    MnemoverseRetriever,
    MnemoverseSearchTool,
    MnemoverseToolkit,
)
from langchain_mnemoverse._common import NoSyncClientError

SEARCH = {"type": "tool_call", "id": "1", "name": "mnemoverse_search", "args": {"query": "q"}}


# -- only an async client passed ---------------------------------------------------------------


def test_only_an_async_client_never_falls_back_to_the_environment_key(opened: Opened) -> None:
    # MNEMOVERSE_API_KEY is set (fixture), and it may belong to another account than the client.
    passed = AsyncFakeMnemo()
    retriever = MnemoverseRetriever(domain="user:ana", async_client=passed)
    with pytest.raises(NoSyncClientError, match="client="):
        retriever.invoke("q")
    msg = MnemoverseSearchTool(domain="user:ana", async_client=passed).invoke(SEARCH)
    assert msg.status == "error"
    assert "client=" in str(msg.content)
    assert opened.sync == []  # no client opened from the environment key
    assert passed.calls == []


def test_only_an_async_client_with_an_explicit_key_opens_sync_from_that_key(opened: Opened) -> None:
    retriever = MnemoverseRetriever(
        domain="user:ana",
        async_client=AsyncFakeMnemo(),
        api_key="mk_live_explicit",
        base_url="https://h.test",
    )
    retriever.invoke("q")
    assert [(c.api_key, c.base_url) for c in opened.sync] == [("mk_live_explicit", "https://h.test")]


async def test_only_an_async_client_still_serves_async_calls(opened: Opened) -> None:
    passed = AsyncFakeMnemo()
    await MnemoverseRetriever(domain="user:ana", async_client=passed).ainvoke("q")
    assert len(passed.named("read")) == 1
    assert opened.aio == []


# -- clients the package opens are shared ------------------------------------------------------


def test_configurable_fields_copies_share_the_opened_clients(opened: Opened) -> None:
    retriever = MnemoverseRetriever(domain="user:default").configurable_fields(
        domain=ConfigurableField(id="mnemoverse_domain")
    )
    for uid in range(5):
        retriever.invoke("q", config={"configurable": {"mnemoverse_domain": f"user:{uid}"}})
    # langchain-core builds a new retriever for each call; all five use one sync client.
    assert len(opened.sync) == 1
    assert [kw["domain"] for kw in opened.sync[0].named("read")] == [f"user:{i}" for i in range(5)]
    assert opened.aio == []  # no async call, no async client


def test_components_with_the_same_key_and_address_share_clients(opened: Opened) -> None:
    retriever = MnemoverseRetriever(domain="d")
    search = MnemoverseSearchTool(domain="d")
    toolkit = MnemoverseToolkit(domain="d")
    retriever.invoke("q")
    search.invoke(SEARCH)
    for tool in toolkit.get_tools():
        tool.invoke(
            {
                **SEARCH,
                "name": tool.name,
                "args": {"query": "q", "content": "x", "memory_ids": ["a"], "helpful": True},
            }
        )
    assert len(opened.sync) == 1
    assert [n for n, _ in opened.sync[0].calls] == ["read", "read", "read", "write", "feedback"]
    others = [
        MnemoverseRetriever(domain="d", api_key="mk_live_other"),
        MnemoverseRetriever(domain="d", base_url="https://other.test"),
    ]
    for other in others:
        other.invoke("q")
    assert [(c.api_key, c.base_url) for c in opened.sync] == [
        ("mk_live_env", "https://core.mnemoverse.com"),
        ("mk_live_other", "https://core.mnemoverse.com"),
        ("mk_live_env", "https://other.test"),
    ]


def test_opened_clients_go_when_the_last_component_does(opened: Opened) -> None:
    import gc

    MnemoverseRetriever(domain="d").invoke("q")
    gc.collect()
    MnemoverseRetriever(domain="d").invoke("q")
    assert len(opened.sync) == 2  # nothing keeps the clients of a component nobody holds


def test_threads_arriving_together_open_one_sync_client(
    opened: Opened, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow_init = OpenedSync.__init__

    def slow(self: OpenedSync, *args: Any, **kwargs: Any) -> None:
        time.sleep(0.05)  # widen the window two threads could both open one in
        slow_init(self, *args, **kwargs)

    monkeypatch.setattr(OpenedSync, "__init__", slow)
    retriever = MnemoverseRetriever(domain="d")
    threads = [threading.Thread(target=retriever.invoke, args=("q",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(opened.sync) == 1
    assert len(opened.sync[0].named("read")) == 4


# -- one async client per event loop -----------------------------------------------------------


def test_a_component_works_from_several_asyncio_runs(opened: Opened) -> None:
    tool = MnemoverseSearchTool(domain="d")
    retriever = MnemoverseRetriever(domain="d")
    for _ in range(3):
        assert asyncio.run(tool.ainvoke(SEARCH)).status == "success"
        assert len(asyncio.run(retriever.ainvoke("q"))) == 4
    # One client per loop, never one carried into a loop it does not belong to.
    assert len(opened.aio) == 6
    assert all(len(c.calls) == 1 for c in opened.aio)


def test_the_middleware_recalls_in_every_asyncio_run(opened: Opened) -> None:
    from conftest import scripted

    mw = MnemoverseMiddleware(domain="d")
    for _ in range(2):
        model = scripted()
        asyncio.run(create_agent(model=model, middleware=[mw]).ainvoke(user("q")))
        assert "<mnemoverse_memory>" in str(model.seen[0][0].text)


async def test_one_loop_reuses_its_async_client(opened: Opened) -> None:
    tool = MnemoverseSearchTool(domain="d")
    await tool.ainvoke(SEARCH)
    await MnemoverseRetriever(domain="d").ainvoke("q")
    assert len(opened.aio) == 1
    assert len(opened.aio[0].calls) == 2


def test_clients_passed_in_are_used_as_given(mem: FakeMnemo, amem: AsyncFakeMnemo, opened: Opened) -> None:
    MnemoverseRetriever(domain="d", client=mem, async_client=amem, api_key="mk_live_x").invoke("q")
    assert len(mem.named("read")) == 1
    assert opened.sync == [] and opened.aio == []
