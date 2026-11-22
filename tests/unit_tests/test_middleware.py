"""MnemoverseMiddleware driven by the real ``create_agent`` with a scripted chat model."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from conftest import ERRORS, AsyncFakeMnemo, FakeMnemo, ScriptedChatModel, atom_id, call_tool, say, scripted
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_mnemoverse import MnemoverseMiddleware, MnemoverseToolkit
from langchain_mnemoverse.middleware import DEFAULT_CONTEXT_PROMPT

LOGGER = "langchain_mnemoverse"


@tool
def echo(x: str) -> str:
    """Echo the input."""
    return x


def user(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def agent(model: ScriptedChatModel, mw: MnemoverseMiddleware, **kwargs: Any) -> Any:
    kwargs.setdefault("system_prompt", "You are helpful.")
    return create_agent(model=model, middleware=[mw], **kwargs)


def system_text(model: ScriptedChatModel, i: int = 0) -> str:
    first = model.seen[i][0]
    return str(first.text) if first.type == "system" else ""


def middleware(mem: FakeMnemo, amem: AsyncFakeMnemo, **kwargs: Any) -> MnemoverseMiddleware:
    kwargs.setdefault("domain", "user:ana")
    return MnemoverseMiddleware(client=mem, async_client=amem, **kwargs)


# -- recall ------------------------------------------------------------------------------------


def test_one_read_per_run_even_across_a_tool_loop(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted(call_tool("echo", {"x": "hi"}), say("done"))
    result = agent(model, middleware(mem, amem, top_k=3), tools=[echo]).invoke(user("which seat?"))
    assert mem.named("read") == [
        {"query": "which seat?", "top_k": 3, "domain": "user:ana", "min_relevance": 0.3}
    ]
    assert len(model.seen) == 2  # the model ran twice; memory was read once
    for i in (0, 1):
        assert "<mnemoverse_memory>" in system_text(model, i)
    assert any(isinstance(m, ToolMessage) for m in result["messages"])
    assert "mnemoverse_context" not in result  # private state stays out of the output


def test_system_message_keeps_the_prompt_and_gains_at_most_top_k_lines(
    mem: FakeMnemo, amem: AsyncFakeMnemo
) -> None:
    model = scripted()
    agent(model, middleware(mem, amem, top_k=3)).invoke(user("which seat?"))
    text = system_text(model)
    assert text.startswith("You are helpful.\n\n" + DEFAULT_CONTEXT_PROMPT)
    assert text.count("\n- [") == 3  # top_k, although the server answered with eight
    assert f"- [{atom_id(0)}] memory 0" in text
    assert text.endswith("</mnemoverse_memory>")
    assert [m.type for m in model.seen[0]][1:] == ["human"]  # nothing added to the conversation


def test_no_system_prompt_gets_the_block_alone(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted()
    create_agent(model=model, middleware=[middleware(mem, amem)]).invoke(user("q"))
    assert system_text(model).startswith(DEFAULT_CONTEXT_PROMPT)


def test_system_message_with_content_blocks(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted()
    prompt = SystemMessage(content=[{"type": "text", "text": "You are helpful."}])
    agent(model, middleware(mem, amem), system_prompt=prompt).invoke(user("q"))
    content = model.seen[0][0].content
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "You are helpful."}
    assert content[-1]["text"].startswith(DEFAULT_CONTEXT_PROMPT)


def test_custom_context_prompt(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted()
    agent(model, middleware(mem, amem, context_prompt="Notes:")).invoke(user("q"))
    assert "You are helpful.\n\nNotes:\n\n<mnemoverse_memory>" in system_text(model)


def test_nothing_recalled_leaves_the_prompt_alone(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    mem.items = []
    model = scripted()
    agent(model, middleware(mem, amem)).invoke(user("q"))
    assert system_text(model) == "You are helpful."


def test_query_is_the_latest_user_message_cut_to_5000(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted()
    messages = [{"role": "user", "content": "first"}, {"role": "user", "content": "a" * 10 + "b" * 5000}]
    agent(model, middleware(mem, amem)).invoke({"messages": messages})
    assert mem.named("read")[0]["query"] == "b" * 5000


@pytest.mark.parametrize("text", ["", "   "])
def test_no_user_text_no_read(mem: FakeMnemo, amem: AsyncFakeMnemo, text: str) -> None:
    model = scripted()
    agent(model, middleware(mem, amem)).invoke(user(text))
    assert mem.calls == []
    assert system_text(model) == "You are helpful."


def test_no_human_message_no_read(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted()
    agent(model, middleware(mem, amem)).invoke({"messages": [{"role": "assistant", "content": "Hello."}]})
    assert mem.calls == []
    assert system_text(model) == "You are helpful."


async def test_async_run_uses_the_async_client(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted(call_tool("echo", {"x": "hi"}), say("done"))
    await agent(model, middleware(mem, amem, top_k=2), tools=[echo]).ainvoke(user("which seat?"))
    assert len(amem.named("read")) == 1
    assert mem.calls == []
    assert system_text(model, 1).count("\n- [") == 2


def test_each_run_reads_again_with_a_checkpointer(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted(say("one"), say("two"))
    graph = agent(model, middleware(mem, amem), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(user("first question"), config)
    mem.items = []
    graph.invoke(user("second question"), config)
    assert [c["query"] for c in mem.named("read")] == ["first question", "second question"]
    assert "<mnemoverse_memory>" not in system_text(model, 1)  # the first run's block is gone


# -- which domain ------------------------------------------------------------------------------


def test_domain_from_the_run_config(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    graph = agent(scripted(), middleware(mem, amem, domain=None))
    graph.invoke(user("q"), {"configurable": {"mnemoverse_domain": "user:bob"}})
    assert mem.named("read")[0]["domain"] == "user:bob"


def test_domain_from_the_run_context(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    graph = agent(scripted(), middleware(mem, amem, domain=None))
    graph.invoke(user("q"), context={"mnemoverse_domain": "user:carol"})
    assert mem.named("read")[0]["domain"] == "user:carol"


def test_no_domain_warns_once_and_reads_nothing(
    mem: FakeMnemo, amem: AsyncFakeMnemo, caplog: pytest.LogCaptureFixture
) -> None:
    mw = middleware(mem, amem, domain=None)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(2):
            model = scripted()
            agent(model, mw).invoke(user("q"))
            assert system_text(model) == "You are helpful."
    assert mem.calls == []
    warnings = [r for r in caplog.records if "no Mnemoverse domain" in r.getMessage()]
    assert len(warnings) == 1


def test_toolkit_tools_get_the_run_domain_through_create_agent(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    model = scripted(call_tool("mnemoverse_save", {"content": "I take the window seat"}), say("saved"))
    kit = MnemoverseToolkit(client=mem, async_client=amem)
    graph = agent(model, middleware(mem, amem, domain=None), tools=kit.get_tools())
    graph.invoke(user("remember my seat"), {"configurable": {"mnemoverse_domain": "user:dan"}})
    assert [(n, kw["domain"]) for n, kw in mem.calls] == [("read", "user:dan"), ("write", "user:dan")]


# -- failing open ------------------------------------------------------------------------------


def test_timeout_runs_without_memory(
    mem: FakeMnemo, amem: AsyncFakeMnemo, caplog: pytest.LogCaptureFixture
) -> None:
    mem.delay = 0.5
    model = scripted()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = agent(model, middleware(mem, amem, recall_timeout=0.05)).invoke(user("q"))
    assert result["messages"][-1].content == "ok"
    assert system_text(model) == "You are helpful."
    assert "recall took longer than 0.05s" in caplog.text
    assert mem.completed == []  # not finished yet, and not cancelled either


async def test_timeout_runs_without_memory_async(
    mem: FakeMnemo, amem: AsyncFakeMnemo, caplog: pytest.LogCaptureFixture
) -> None:
    amem.delay = 0.3
    model = scripted()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await agent(model, middleware(mem, amem, recall_timeout=0.05)).ainvoke(user("q"))
    assert result["messages"][-1].content == "ok"
    assert system_text(model) == "You are helpful."
    assert "recall took longer" in caplog.text
    await asyncio.sleep(0.4)
    assert amem.completed == ["read"]  # the call was left to finish, not cancelled


@pytest.mark.parametrize("name", ["rate", "unavailable", "server"])
def test_failed_recall_runs_without_memory(
    mem: FakeMnemo, amem: AsyncFakeMnemo, caplog: pytest.LogCaptureFixture, name: str
) -> None:
    error, phrase = ERRORS[name]
    mem.fail["read"] = error
    model = scripted()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        agent(model, middleware(mem, amem)).invoke(user("q"))
    assert system_text(model) == "You are helpful."
    assert phrase in caplog.text


def test_refusal_stops_recall_for_that_domain_only(
    mem: FakeMnemo, amem: AsyncFakeMnemo, caplog: pytest.LogCaptureFixture
) -> None:
    mem.fail_in[("read", "user:ana")] = ERRORS["auth"][0]
    mw = middleware(mem, amem, domain=None)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        for domain in ("user:ana", "user:ana", "user:bob"):
            agent(scripted(), mw).invoke(user("q"), {"configurable": {"mnemoverse_domain": domain}})
    assert [c["domain"] for c in mem.named("read")] == ["user:ana", "user:bob"]
    assert "recall stopped for domain 'user:ana'" in caplog.text
    assert "refused the call (401): Invalid API key" in caplog.text


async def test_refusal_stops_recall_for_that_domain_only_async(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    amem.fail_in[("read", "user:ana")] = ERRORS["forbidden"][0]
    mw = middleware(mem, amem, domain=None)
    for domain in ("user:ana", "user:ana", "user:bob"):
        await agent(scripted(), mw).ainvoke(user("q"), {"configurable": {"mnemoverse_domain": domain}})
    assert [c["domain"] for c in amem.named("read")] == ["user:ana", "user:bob"]


def test_only_an_async_client_and_no_key_runs_without_memory(no_key: None, amem: AsyncFakeMnemo) -> None:
    model = scripted()
    mw = MnemoverseMiddleware(domain="user:ana", async_client=amem)
    agent(model, mw).invoke(user("q"))
    assert system_text(model) == "You are helpful."


# -- capture -----------------------------------------------------------------------------------


def test_capture_off_by_default(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    agent(scripted(), middleware(mem, amem)).invoke(user("I take the window seat"))
    assert mem.named("write") == []


def test_capture_saves_the_user_messages_of_the_run(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    graph = agent(
        scripted(say("one"), say("two")), middleware(mem, amem, capture=True), checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(
        {"messages": [{"role": "user", "content": "I take"}, {"role": "user", "content": "the window"}]},
        config,
    )
    graph.invoke(user("and tea, not coffee"), config)
    assert mem.named("write") == [
        {
            "content": "I take\n\nthe window",
            "domain": "user:ana",
            "metadata": {"platform": "langchain", "role": "user"},
        },
        {
            "content": "and tea, not coffee",
            "domain": "user:ana",
            "metadata": {"platform": "langchain", "role": "user"},
        },
    ]


def test_capture_keeps_the_first_10000_characters(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    agent(scripted(), middleware(mem, amem, capture=True)).invoke(user("a" * 10_000 + "b" * 50))
    assert mem.named("write")[0]["content"] == "a" * 10_000


async def test_capture_async(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    await agent(scripted(), middleware(mem, amem, capture=True)).ainvoke(user("tea, not coffee"))
    assert [n for n, _ in amem.calls] == ["read", "write"]
    assert amem.named("write")[0]["content"] == "tea, not coffee"
    assert mem.calls == []


def test_failed_capture_does_not_break_the_run(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    mem.fail["write"] = ERRORS["server"][0]
    result = agent(scripted(), middleware(mem, amem, capture=True)).invoke(user("x"))
    assert result["messages"][-1].content == "ok"


def test_refused_capture_stops_for_that_domain(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    mem.fail_in[("write", "user:ana")] = ERRORS["auth"][0]
    mw = middleware(mem, amem, capture=True)
    for _ in range(2):
        agent(scripted(), mw).invoke(user("x"))
    assert len(mem.named("write")) == 1
    assert len(mem.named("read")) == 2  # recall goes on


# -- settings ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"domain": ""},
        {"domain": "x" * 101},
        {"top_k": 0},
        {"top_k": 501},
        {"min_relevance": 1.5},
        {"recall_timeout": 0},
    ],
)
def test_invalid_settings_are_refused(mem: FakeMnemo, amem: AsyncFakeMnemo, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        middleware(mem, amem, **kwargs)


def test_defaults(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    mw = MnemoverseMiddleware(client=mem, async_client=amem)
    assert (mw.domain, mw.top_k, mw.min_relevance, mw.recall_timeout, mw.capture) == (
        None,
        5,
        0.3,
        8.0,
        False,
    )
    assert mw.context_prompt == DEFAULT_CONTEXT_PROMPT
