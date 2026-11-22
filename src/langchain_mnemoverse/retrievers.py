"""``MnemoverseRetriever``: memories from one Mnemoverse domain as LangChain documents."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from mnemoverse import AsyncMnemoClient, MnemoClient
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, field_validator

from ._common import DEFAULT_BASE_URL, MAX_TOP_K, Clients, check_domain, check_top_k, cut_query, to_document


class MnemoverseRetriever(BaseRetriever):
    """Retrieve memories from a Mnemoverse domain as ``Document`` objects.

    Each call reads the memories most relevant to the query from one domain, at most ``k`` of
    them, also when the server returns more. The document id is the memory's atom id; relevance,
    domain, concepts, creation time and the rest are in ``metadata``.

    Setup:
        Install ``langchain-mnemoverse`` and set ``MNEMOVERSE_API_KEY``, or pass ``api_key=``.

        .. code-block:: bash

            pip install -U langchain-mnemoverse
            export MNEMOVERSE_API_KEY="mk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_mnemoverse import MnemoverseRetriever

            retriever = MnemoverseRetriever(domain="project:docs", k=4)

    Invoke:
        .. code-block:: python

            docs = retriever.invoke("Which database does the ledger use?")
            docs = retriever.invoke("Which seat?", k=2, domain="user:42")  # per-call k and domain

    The query is stripped and cut to its last 5,000 characters; a blank query returns no
    documents and makes no request.

    SDK errors (``MnemoAuthError``, ``MnemoRateLimitError``, ``MnemoUnavailableError``,
    ``MnemoError``) propagate unchanged; an invalid domain raises ``ValueError``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: str
    """The domain every call reads, for example ``project:docs`` or ``user:42``. Required: there
    is no search across domains by default. Override per call with ``invoke(query, domain=...)``."""

    k: int = Field(default=4, ge=1, le=MAX_TOP_K)
    """Documents per call, 1 to 500. Override per call with ``invoke(query, k=...)``."""

    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0)
    """Lowest relevance, 0 to 1, of a returned memory."""

    api_key: SecretStr | None = None
    """A Mnemoverse API key. Falls back to the ``MNEMOVERSE_API_KEY`` environment variable."""

    base_url: str = DEFAULT_BASE_URL

    client: MnemoClient | None = Field(default=None, exclude=True)
    """A ready sync client, used by ``invoke``. Never closed by the retriever."""

    async_client: AsyncMnemoClient | None = Field(default=None, exclude=True)
    """A ready async client, used by ``ainvoke``. Never closed by the retriever."""

    _clients: Clients = PrivateAttr()

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, value: str) -> str:
        return check_domain(value)

    def model_post_init(self, context: Any, /) -> None:
        key = self.api_key.get_secret_value() if self.api_key else None
        self._clients = Clients(self.client, self.async_client, key, self.base_url)

    def _request(self, query: str, k: int | None, domain: str | None) -> tuple[str, dict[str, Any], int]:
        top_k = self.k if k is None else check_top_k(k)
        name = self.domain if domain is None else check_domain(domain)
        return (
            cut_query(query),
            {"top_k": top_k, "domain": name, "min_relevance": self.min_relevance},
            top_k,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int | None = None,
        domain: str | None = None,
    ) -> list[Document]:
        text, kwargs, top_k = self._request(query, k, domain)
        if not text:  # a blank query: nothing to look for, no request
            return []
        result = self._clients.sync.read(text, **kwargs)
        return [to_document(item) for item in result.items[:top_k]]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        k: int | None = None,
        domain: str | None = None,
    ) -> list[Document]:
        text, kwargs, top_k = self._request(query, k, domain)
        if not text:  # a blank query: nothing to look for, no request
            return []
        result = await self._clients.aio.read(text, **kwargs)
        return [to_document(item) for item in result.items[:top_k]]


__all__ = ["MnemoverseRetriever"]
