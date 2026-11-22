"""Security properties: the key never shows, the model never picks the domain, users on shared
instances never see each other's memory, capture keeps only the user's words, and logs hold no
key, query or memory text. (The memory block's fence is in test_middleware.py.)"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from conftest import AsyncFakeMnemo, FakeMnemo, ScriptedChatModel, call_tool, item, say, scripted, user
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd, dumps
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.utils.function_calling import convert_to_openai_tool
from mnemoverse import MnemoAuthError, MnemoError, ReadResponse

from langchain_mnemoverse import (
    MnemoverseMiddleware,
    MnemoverseRateTool,
    MnemoverseRetriever,
    MnemoverseSaveTool,
    MnemoverseSearchTool,
    MnemoverseToolkit,
)

SECRET = "mk_live_SECRETKEY_0123456789abcdef"
ARGS = {
    "mnemoverse_search": {"query": "q"},
    "mnemoverse_save": {"content": "c"},
    "mnemoverse_rate": {"memory_ids": ["x"], "helpful": True},
}


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call", "id": "1", "name": name, "args": args}


class Everything(BaseCallbackHandler):
    """Keeps every argument LangChain hands a callback handler: what a tracer could keep."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("on_"):
            events = object.__getattribute__(self, "events")
            return lambda *a, **k: events.append(f"{name} {a!r} {k!r}")
        return object.__getattribute__(self, name)


class DomainEcho(FakeMnemo):
    """Answers each read with one memory naming the domain it was asked for."""

    def read(self, query: str, **kwargs: Any) -> ReadResponse:  # type: ignore[override]
        self._enter("read", query=query, **kwargs)
        domain = kwargs["domain"]
        return ReadResponse(
            items=[item(0, domain=domain, content=f"SECRET-OF[{domain}]")],
            episodic_hit=False,
            query_concepts=[],
            expanded_concepts=[],
            search_time_ms=1.0,
        )


class AsyncDomainEcho(AsyncFakeMnemo):
    async def read(self, query: str, **kwargs: Any) -> ReadResponse:  # type: ignore[override]
        await self._enter("read", query=query, **kwargs)
        await asyncio.sleep(0.001 * (len(query) % 5))  # interleave the runs
        domain = kwargs["domain"]
        return ReadResponse(
            items=[item(0, domain=domain, content=f"SECRET-OF[{domain}]")],
            episodic_hit=False,
            query_concepts=[],
            expanded_concepts=[],
            search_time_ms=1.0,
        )


# -- the key -----------------------------------------------------------------------------------


def _texts(component: Any) -> list[str]:
    texts = [repr(component), str(component), json.dumps(dumpd(component), default=str), dumps(component)]
    if hasattr(component, "model_dump"):
        texts.append(json.dumps(component.model_dump(exclude={"args_schema"}), default=str))
    return texts


def test_the_key_never_shows_in_repr_dumps_or_serialisation(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    passed = {"api_key": SECRET, "client": mem, "async_client": amem}
    components = [
        MnemoverseRetriever(domain="user:ana", **passed),
        MnemoverseSearchTool(domain="user:ana", **passed),
        MnemoverseSaveTool(domain="user:ana", **passed),
        MnemoverseRateTool(domain="user:ana", **passed),
        MnemoverseToolkit(domain="user:ana", **passed),
        *MnemoverseToolkit(domain="user:ana", **passed).get_tools(),
        MnemoverseMiddleware(domain="user:ana", **passed),
    ]
    for component in components:
        for text in _texts(component):
            assert SECRET not in text, type(component).__name__


def test_the_key_never_shows_when_the_package_opens_the_clients(no_key: None) -> None:
    components = [
        MnemoverseRetriever(domain="user:ana", api_key=SECRET),
        MnemoverseSearchTool(domain="user:ana", api_key=SECRET),
        MnemoverseToolkit(domain="user:ana", api_key=SECRET),
        *MnemoverseToolkit(domain="user:ana", api_key=SECRET).get_tools(),
        MnemoverseMiddleware(domain="user:ana", api_key=SECRET),
    ]
    for component in components:
        for text in _texts(component):
            assert SECRET not in text, type(component).__name__


def test_the_key_never_reaches_callbacks_or_traced_runs(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    everything, collector = Everything(), RunCollectorCallbackHandler()
    config: Any = {"callbacks": [everything, collector], "configurable": {"mnemoverse_domain": "user:ana"}}
    retriever = MnemoverseRetriever(domain="user:ana", api_key=SECRET, client=mem, async_client=amem)
    retriever.invoke("q", config=config)
    retriever.invoke("q", config=config, domain="user:bob", k=2)
    tools = MnemoverseToolkit(api_key=SECRET, client=mem, async_client=amem).get_tools()
    for t in tools:
        t.invoke(tool_call(t.name, ARGS[t.name]), config=config)
    mw = MnemoverseMiddleware(api_key=SECRET, client=mem, async_client=amem, capture=True)
    model = scripted(call_tool("mnemoverse_search", {"query": "q"}), say("done"))
    create_agent(model=model, tools=tools, middleware=[mw]).invoke(user("hello"), config=config)
    runs = []
    stack = list(collector.traced_runs)
    while stack:
        run = stack.pop()
        runs.append(run.model_dump_json())
        stack.extend(run.child_runs)
    assert everything.events and runs
    assert SECRET not in "\n".join(everything.events)
    assert SECRET not in "\n".join(runs)


def test_the_key_never_reaches_the_model_in_a_tool_error(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    for error in (MnemoAuthError("Invalid API key"), MnemoError("boom", status=500), RuntimeError(SECRET)):
        mem.fail["read"] = error
        t = MnemoverseSearchTool(domain="user:ana", api_key=SECRET, client=mem, async_client=amem)
        msg = t.invoke(tool_call(t.name, {"query": "q"}))
        assert msg.status == "error"
        assert SECRET not in str(msg.content)


def test_tool_schemas_show_no_key_domain_or_config(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    for t in MnemoverseToolkit(domain="user:ana", api_key=SECRET, client=mem, async_client=amem).get_tools():
        for schema in (
            json.dumps(convert_to_openai_tool(t)),
            json.dumps(t.tool_call_schema.model_json_schema()),
        ):
            assert SECRET not in schema
            for word in ("domain", "api_key", "config", "run_manager", "base_url"):
                assert f'"{word}"' not in schema, (t.name, word)


# -- the model cannot pick the domain ----------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"domain": "user:victim"},
        {"config": {"configurable": {"mnemoverse_domain": "user:victim"}}},
        {"mnemoverse_domain": "user:victim"},
        {"run_manager": "x", "callbacks": "y"},
    ],
)
def test_arguments_the_model_adds_never_change_the_domain(
    mem: FakeMnemo, amem: AsyncFakeMnemo, extra: dict[str, Any]
) -> None:
    config: Any = {"configurable": {"mnemoverse_domain": "user:ana"}}
    for t in MnemoverseToolkit(client=mem, async_client=amem).get_tools():
        assert isinstance(t.invoke(tool_call(t.name, {**ARGS[t.name], **extra}), config=config), ToolMessage)
    assert mem.calls
    assert {kw["domain"] for _, kw in mem.calls} == {"user:ana"}


def test_a_config_argument_from_the_model_is_ignored_in_create_agent(
    mem: FakeMnemo, amem: AsyncFakeMnemo
) -> None:
    evil = {"query": "q", "config": {"configurable": {"mnemoverse_domain": "user:victim"}}}
    model = scripted(call_tool("mnemoverse_search", evil), say("done"))
    tools = MnemoverseToolkit(client=mem, async_client=amem).get_tools()
    create_agent(model=model, tools=tools).invoke(
        user("hi"), config={"configurable": {"mnemoverse_domain": "user:ana"}}
    )
    assert [kw["domain"] for kw in mem.named("read")] == ["user:ana"]


# -- users on shared instances -----------------------------------------------------------------


def _check_isolation(results: list[tuple[str, ScriptedChatModel, Any]]) -> None:
    for domain, model, result in results:
        system = str(model.seen[0][0].text)
        assert f"SECRET-OF[{domain}]" in system and system.count("SECRET-OF[") == 1
        answers = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(answers) == 1
        assert f"SECRET-OF[{domain}]" in str(answers[0].content)
        assert str(answers[0].content).count("SECRET-OF[") == 1


async def test_shared_middleware_and_tools_never_mix_users_async() -> None:
    mem, amem = DomainEcho(), AsyncDomainEcho()
    mw = MnemoverseMiddleware(client=mem, async_client=amem, capture=True)
    tools = MnemoverseToolkit(client=mem, async_client=amem).get_tools()

    async def run(i: int) -> tuple[str, ScriptedChatModel, Any]:
        domain = f"user:{i}"
        model = scripted(call_tool("mnemoverse_search", {"query": f"tool-q-{i}"}), say("done"))
        named = {"mnemoverse_domain": domain}
        # Half the runs name the domain in the config, half in the context.
        kwargs: dict[str, Any] = {"config": {"configurable": named}} if i % 2 else {"context": named}
        result = await create_agent(model=model, tools=tools, middleware=[mw]).ainvoke(
            user(f"question-of-{domain}"), **kwargs
        )
        return domain, model, result

    _check_isolation(list(await asyncio.gather(*(run(i) for i in range(30)))))
    writes = amem.named("write")
    assert len(writes) == 30
    assert all(kw["content"] == f"question-of-{kw['domain']}" for kw in writes)


def test_shared_middleware_and_tools_never_mix_users_threads() -> None:
    mem, amem = DomainEcho(), AsyncDomainEcho()
    mw = MnemoverseMiddleware(client=mem, async_client=amem, capture=True)
    tools = MnemoverseToolkit(client=mem, async_client=amem).get_tools()

    def run(i: int) -> tuple[str, ScriptedChatModel, Any]:
        domain = f"user:{i}"
        model = scripted(call_tool("mnemoverse_search", {"query": f"tool-q-{i}"}), say("done"))
        result = create_agent(model=model, tools=tools, middleware=[mw]).invoke(
            user(f"question-of-{domain}"), config={"configurable": {"mnemoverse_domain": domain}}
        )
        return domain, model, result

    with ThreadPoolExecutor(max_workers=8) as pool:
        _check_isolation(list(pool.map(run, range(24))))
    writes = mem.named("write")
    assert len(writes) == 24
    assert all(kw["content"] == f"question-of-{kw['domain']}" for kw in writes)


# -- what capture keeps ------------------------------------------------------------------------


@tool
def leak(x: str) -> str:
    """Return tool output."""
    return "TOOL-OUTPUT-MARKER"


def test_capture_keeps_only_the_users_messages_of_the_run(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted(call_tool("leak", {"x": "MODEL-ARG-MARKER"}), say("MODEL-REPLY-MARKER"))
    mw = MnemoverseMiddleware(domain="user:ana", client=mem, async_client=amem, capture=True)
    create_agent(model=model, tools=[leak], system_prompt="SYSTEM-MARKER", middleware=[mw]).invoke(
        {
            "messages": [
                HumanMessage("old human"),
                AIMessage("OLD-AI-MARKER"),
                SystemMessage("INPUT-SYSTEM-MARKER"),
                HumanMessage("first"),
                HumanMessage("second"),
            ]
        }
    )
    writes = mem.named("write")
    assert [w["content"] for w in writes] == ["first\n\nsecond"]
    assert writes[0]["metadata"] == {"platform": "langchain", "role": "user"}


# -- logs --------------------------------------------------------------------------------------


def test_logs_hold_no_key_query_or_memory_text(
    amem: AsyncFakeMnemo, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    query = "MY-PRIVATE-QUESTION " * 3
    cases: list[tuple[BaseException | None, dict[str, Any] | None]] = [
        (MnemoAuthError("Invalid API key"), None),
        (MnemoError("server error", status=500), None),
        (RuntimeError("boom MY-PRIVATE-QUESTION"), None),
        (None, {"configurable": {"mnemoverse_domain": ""}}),
        (None, {"configurable": {"mnemoverse_domain": "user:a"}}),
    ]
    for error, config in cases:
        mem = FakeMnemo()
        mem.items = [item(0, content="PRIVATE-MEMORY-CONTENT")]
        if error is not None:
            mem.fail["read"] = mem.fail["write"] = error
        mw = MnemoverseMiddleware(
            domain=None if config else "user:ana", api_key=SECRET, client=mem, async_client=amem, capture=True
        )
        create_agent(model=scripted(), middleware=[mw]).invoke(user(query + "x" * 10_001), config=config)
    text = "\n".join(f"{r.name} {r.levelname} {r.getMessage()}" for r in caplog.records)
    assert SECRET not in text
    assert "MY-PRIVATE-QUESTION" not in text
    assert "PRIVATE-MEMORY-CONTENT" not in text
    ours = [r for r in caplog.records if r.name.startswith("langchain_mnemoverse")]
    assert ours and all(r.levelno >= logging.WARNING for r in ours)
