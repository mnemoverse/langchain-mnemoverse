"""Mnemoverse long-term memory for LangChain.

* :class:`MnemoverseRetriever` returns memories from a domain as ``Document`` objects.
* :class:`MnemoverseToolkit` gives an agent the ``mnemoverse_search``, ``mnemoverse_save`` and
  ``mnemoverse_rate`` tools (also importable one by one).
* :class:`MnemoverseMiddleware` recalls memory once per ``create_agent`` run. It needs the
  ``langchain`` package: ``pip install "langchain-mnemoverse[agents]"``.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from ._version import __version__
from .retrievers import MnemoverseRetriever
from .tools import MnemoverseRateTool, MnemoverseSaveTool, MnemoverseSearchTool, MnemoverseToolkit

if TYPE_CHECKING:
    from .middleware import MnemoverseMiddleware

MIDDLEWARE_IMPORT_ERROR = (
    'MnemoverseMiddleware needs the langchain package: pip install "langchain-mnemoverse[agents]"'
)


def __getattr__(name: str) -> Any:
    # MnemoverseMiddleware needs the `langchain` package, which the rest of this package does not,
    # so it is imported on first access instead of with the package.
    if name == "MnemoverseMiddleware":
        try:
            from .middleware import MnemoverseMiddleware
        except ModuleNotFoundError as exc:
            if exc.name == "langchain" or (exc.name or "").startswith("langchain."):
                raise ImportError(MIDDLEWARE_IMPORT_ERROR) from exc
            raise
        return MnemoverseMiddleware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MnemoverseMiddleware",
    "MnemoverseRateTool",
    "MnemoverseRetriever",
    "MnemoverseSaveTool",
    "MnemoverseSearchTool",
    "MnemoverseToolkit",
    "__version__",
]
if importlib.util.find_spec("langchain") is None:
    # Without the `agents` extra, `from langchain_mnemoverse import *` must still work: it would
    # fetch every name in __all__ and stop at the middleware's ImportError.
    __all__.remove("MnemoverseMiddleware")
