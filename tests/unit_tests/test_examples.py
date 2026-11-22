"""The examples, run offline: SDK clients and the chat model replaced by fakes."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import langchain.agents
import mnemoverse
import pytest
from conftest import Opened, OpenedSync, call_tool, say, scripted

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


class _ContextFake(OpenedSync):
    def __enter__(self) -> _ContextFake:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_agent_example_keeps_the_middleware_off_the_tools_client(
    opened: Opened, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mnemoverse, "MnemoClient", _ContextFake)  # a client the example builds itself
    real_create_agent = langchain.agents.create_agent
    model = scripted(
        call_tool("mnemoverse_save", {"content": "I always take the window seat."}),
        say("Saved."),
        say("The window seat."),
    )

    def offline_create_agent(*, model_name: Any = None, **kwargs: Any) -> Any:
        kwargs.pop("model")
        return real_create_agent(model=model, **kwargs)

    monkeypatch.setattr(langchain.agents, "create_agent", offline_create_agent)
    monkeypatch.setattr(sys, "argv", ["agent_with_memory.py", "ana"])
    runpy.run_path(str(EXAMPLES / "agent_with_memory.py"), run_name="__main__")

    assert "The window seat." in capsys.readouterr().out
    by_client = [[(name, kw["domain"]) for name, kw in client.calls] for client in opened.sync]
    reads = [c for c in by_client if ("read", "user:ana") in c]
    writes = [c for c in by_client if ("write", "user:ana") in c]
    assert len(reads) == 1 and len(writes) == 1
    # The middleware's recalls and the tools' save went through different clients.
    assert reads[0] is not writes[0]
    assert reads[0] == [("read", "user:ana"), ("read", "user:ana")]
    assert writes[0] == [("write", "user:ana")]
