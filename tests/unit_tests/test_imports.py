"""The import surface: the exported names, the lazy middleware import, and the failure without a
key."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

import langchain_mnemoverse
from langchain_mnemoverse import (
    MIDDLEWARE_IMPORT_ERROR,
    MnemoverseRateTool,
    MnemoverseRetriever,
    MnemoverseSaveTool,
    MnemoverseSearchTool,
    MnemoverseToolkit,
)

EXPECTED_ALL = [
    "MnemoverseMiddleware",
    "MnemoverseRateTool",
    "MnemoverseRetriever",
    "MnemoverseSaveTool",
    "MnemoverseSearchTool",
    "MnemoverseToolkit",
    "__version__",
]


def test_all_imports() -> None:
    assert sorted(langchain_mnemoverse.__all__) == sorted(EXPECTED_ALL)
    for name in EXPECTED_ALL:
        assert getattr(langchain_mnemoverse, name) is not None


def test_version_matches_the_version_file() -> None:
    from langchain_mnemoverse._version import __version__

    assert langchain_mnemoverse.__version__ == __version__ == "0.1.0"


def test_middleware_is_not_imported_with_the_package() -> None:
    code = (
        "import sys, langchain_mnemoverse; "
        "print('langchain' in sys.modules, 'langchain_mnemoverse.middleware' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_middleware_import_resolves() -> None:
    from langchain_mnemoverse.middleware import MnemoverseMiddleware

    assert langchain_mnemoverse.MnemoverseMiddleware is MnemoverseMiddleware


def test_middleware_without_langchain_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "langchain" or name.startswith("langchain.") or name == "langchain_mnemoverse.middleware":
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "langchain", None)  # makes `import langchain` fail
    with pytest.raises(ImportError) as info:
        _ = langchain_mnemoverse.MnemoverseMiddleware
    assert str(info.value) == (
        'MnemoverseMiddleware needs the langchain package: pip install "langchain-mnemoverse[agents]"'
    )
    # The rest of the package works without langchain.
    assert langchain_mnemoverse.MnemoverseRetriever is MnemoverseRetriever


def test_without_langchain_star_import_works_and_the_middleware_says_how_to_install() -> None:
    # A fresh interpreter where `import langchain` fails, as on an install without the extra.
    code = """
import sys
sys.modules["langchain"] = None
import langchain_mnemoverse as pkg
namespace = {}
exec("from langchain_mnemoverse import *", namespace)
print(sorted(n for n in namespace if not n.startswith("__")) + ["__version__" in namespace])
print("MnemoverseMiddleware" in pkg.__all__)
try:
    pkg.MnemoverseMiddleware
except ImportError as exc:
    print("ImportError:", exc)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    names, listed, error = out.stdout.strip().split("\n")
    without_middleware = [n for n in sorted(EXPECTED_ALL) if n.startswith("Mnemoverse")]
    without_middleware.remove("MnemoverseMiddleware")
    assert names == str([*without_middleware, True])
    assert listed == "False"
    assert error == "ImportError: " + MIDDLEWARE_IMPORT_ERROR


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        _ = langchain_mnemoverse.Nope


def _middleware(**kwargs: Any) -> Any:
    from langchain_mnemoverse import MnemoverseMiddleware

    return MnemoverseMiddleware(**kwargs)


@pytest.mark.parametrize(
    "build",
    [
        lambda: MnemoverseRetriever(domain="user:ana"),
        lambda: MnemoverseSearchTool(domain="user:ana"),
        lambda: MnemoverseSaveTool(domain="user:ana"),
        lambda: MnemoverseRateTool(domain="user:ana"),
        lambda: MnemoverseToolkit(domain="user:ana"),
        lambda: _middleware(domain="user:ana"),
    ],
    ids=["retriever", "search", "save", "rate", "toolkit", "middleware"],
)
def test_no_key_and_no_client_fails_at_construction(no_key: None, build: Any) -> None:
    with pytest.raises(ValueError, match="No API key"):
        build()


def test_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MNEMOVERSE_API_KEY", "mk_live_test")
    assert MnemoverseRetriever(domain="user:ana").domain == "user:ana"


def test_api_key_is_never_shown(no_key: None) -> None:
    retriever = MnemoverseRetriever(domain="user:ana", api_key="mk_live_secret")
    tool = MnemoverseSearchTool(domain="user:ana", api_key="mk_live_secret")
    for component in (retriever, tool):
        assert "mk_live_secret" not in repr(component)
        assert "mk_live_secret" not in str(component.model_dump())
