"""MnemoverseRetriever against the fake clients."""

from __future__ import annotations

import pytest
from conftest import CREATED, ERRORS, ITEM_COUNT, REPLACEMENT_ID, AsyncFakeMnemo, FakeMnemo, atom_id
from langchain_core.runnables import ConfigurableField
from mnemoverse import MnemoError

from langchain_mnemoverse import MnemoverseRetriever


def _retriever(mem: FakeMnemo, amem: AsyncFakeMnemo, **kwargs: object) -> MnemoverseRetriever:
    return MnemoverseRetriever(domain="user:ana", client=mem, async_client=amem, **kwargs)  # type: ignore[arg-type]


def test_reads_the_domain_with_k_and_min_relevance(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    docs = _retriever(mem, amem).invoke("Which seat?")
    assert mem.named("read") == [
        {"query": "Which seat?", "top_k": 4, "domain": "user:ana", "min_relevance": 0.3}
    ]
    assert amem.calls == []
    assert len(docs) == 4  # k, although the server answered with 8


def test_never_more_than_k_documents(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    assert ITEM_COUNT > 2
    docs = _retriever(mem, amem, k=2).invoke("q")
    assert [d.page_content for d in docs] == ["memory 0", "memory 1"]


async def test_async_uses_the_async_client(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    docs = await _retriever(mem, amem, min_relevance=0.1).ainvoke("Which seat?", k=3)
    assert amem.named("read") == [
        {"query": "Which seat?", "top_k": 3, "domain": "user:ana", "min_relevance": 0.1}
    ]
    assert mem.calls == []
    assert len(docs) == 3


def test_query_is_cut_to_its_last_5000_characters(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    query = "a" * 1000 + "b" * 5000
    _retriever(mem, amem).invoke(query)
    assert mem.named("read")[0]["query"] == "b" * 5000


def test_per_call_k_and_domain(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    retriever = _retriever(mem, amem)
    docs = retriever.invoke("q", k=1, domain=" user:bob ")
    assert len(docs) == 1
    assert mem.named("read")[0]["top_k"] == 1
    assert mem.named("read")[0]["domain"] == "user:bob"
    retriever.invoke("q")  # the per-call values do not stick
    assert mem.named("read")[1]["domain"] == "user:ana"
    assert mem.named("read")[1]["top_k"] == 4


def test_configurable_domain(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    retriever = _retriever(mem, amem).configurable_fields(domain=ConfigurableField(id="mnemoverse_domain"))
    retriever.invoke("q", config={"configurable": {"mnemoverse_domain": "user:42"}})
    retriever.invoke("q")
    assert [c["domain"] for c in mem.named("read")] == ["user:42", "user:ana"]


def test_document_mapping(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    doc = _retriever(mem, amem).invoke("q")[0]
    assert doc.id == str(atom_id(0))
    assert doc.page_content == "memory 0"
    assert doc.metadata == {
        "atom_id": str(atom_id(0)),
        "relevance": 0.9,
        "similarity": 0.8,
        "valence": 0.1,
        "importance": 0.5,
        "domain": "user:ana",
        "concepts": ["seat"],
        "created_at": CREATED.isoformat(),
        "source": "semantic",
        "provenance": {"agent": "claude-code", "is_external": False},
        "metadata": {"platform": "test"},
        "superseded_by": str(REPLACEMENT_ID),
    }
    plain = _retriever(mem, amem).invoke("q")[1].metadata
    assert plain["created_at"] is None
    assert plain["provenance"] is None
    assert plain["superseded_by"] is None
    assert plain["metadata"] == {}


@pytest.mark.parametrize("name", list(ERRORS))
def test_sdk_errors_propagate(mem: FakeMnemo, amem: AsyncFakeMnemo, name: str) -> None:
    error, _ = ERRORS[name]
    mem.fail["read"] = error
    with pytest.raises(type(error)) as info:
        _retriever(mem, amem).invoke("q")
    assert info.value is error
    assert isinstance(info.value, MnemoError)


@pytest.mark.parametrize("name", list(ERRORS))
async def test_sdk_errors_propagate_async(mem: FakeMnemo, amem: AsyncFakeMnemo, name: str) -> None:
    error, _ = ERRORS[name]
    amem.fail["read"] = error
    with pytest.raises(type(error)) as info:
        await _retriever(mem, amem).ainvoke("q")
    assert info.value is error


@pytest.mark.parametrize("domain", ["", "   ", "x" * 101])
def test_invalid_domain_is_refused(mem: FakeMnemo, amem: AsyncFakeMnemo, domain: str) -> None:
    with pytest.raises(ValueError, match="domain"):
        MnemoverseRetriever(domain=domain, client=mem, async_client=amem)
    with pytest.raises(ValueError, match="domain"):
        _retriever(mem, amem).invoke("q", domain=domain)
    assert mem.calls == []


def test_domain_of_100_characters_is_accepted(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    MnemoverseRetriever(domain="x" * 100, client=mem, async_client=amem).invoke("q")
    assert mem.named("read")[0]["domain"] == "x" * 100


def test_domain_is_required(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    with pytest.raises(ValueError, match="domain"):
        MnemoverseRetriever(client=mem, async_client=amem)  # type: ignore[call-arg]


@pytest.mark.parametrize("k", [0, 501])
def test_k_out_of_range_is_refused(mem: FakeMnemo, amem: AsyncFakeMnemo, k: int) -> None:
    with pytest.raises(ValueError):
        _retriever(mem, amem, k=k)
    with pytest.raises(ValueError, match="k must be"):
        _retriever(mem, amem).invoke("q", k=k)


async def test_only_a_sync_client_serves_async_calls(mem: FakeMnemo) -> None:
    docs = await MnemoverseRetriever(domain="user:ana", client=mem).ainvoke("q")
    assert len(docs) == 4
    assert len(mem.named("read")) == 1


def test_only_an_async_client_and_no_key_names_client(no_key: None, amem: AsyncFakeMnemo) -> None:
    retriever = MnemoverseRetriever(domain="user:ana", async_client=amem)
    with pytest.raises(ValueError, match="client="):
        retriever.invoke("q")


def test_clients_passed_in_stay_open(mem: FakeMnemo, amem: AsyncFakeMnemo) -> None:
    retriever = _retriever(mem, amem)
    retriever.invoke("q")
    del retriever
    assert not mem.closed
    assert not amem.closed
