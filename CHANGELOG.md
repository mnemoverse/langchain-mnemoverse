# Changelog

All notable changes to `langchain-mnemoverse`.

## [Unreleased]

### Changed

- Building the package (from the repository or the sdist) uses `hatchling==1.33.0` exactly instead of any `hatchling>=1.24`.
- The sdist also ships `scripts/`, which `mypy` checks, so `mypy` passes in an unpacked sdist.

## [0.1.0] - 2026-11-22

### Added

- `MnemoverseRetriever`: memories from one Mnemoverse domain as LangChain documents, at most `k` per call (default 4) also when the server returns more; `k` and `domain` can be overridden per call.
- `MnemoverseSearchTool`, `MnemoverseSaveTool` and `MnemoverseRateTool` (`mnemoverse_search`, `mnemoverse_save`, `mnemoverse_rate`), and `MnemoverseToolkit`, which builds them around one set of SDK clients. Failures, including arguments that fail the schema, reach the model as error tool messages, also when a tool is invoked directly.
- `MnemoverseMiddleware` for `create_agent` (extra `agents`): one recall per run within `recall_timeout`, added to the system message of every model call in that run; optional capture of the user's messages. Memory never breaks a run. A 401 pauses recall (or capture) for every domain, a 403 for that domain only, each for 15 minutes.
- The domain is fixed, or the run names it in `config["configurable"]["mnemoverse_domain"]` or the agent's `context`; never the model. The middleware and the tools follow the same rule, and a run with no domain, or with a context and a config naming different domains, reads and writes nothing.
- Recalled memory is shown to the model one line per memory, with `<` and `>` replaced, so its text cannot close the memory block or pose as another memory.
- SDK clients the package opens are shared by components with the same key and address, one async client per event loop; with only `async_client=` passed, sync calls never fall back to `MNEMOVERSE_API_KEY`.
- Queries and memories are stripped, queries cut to their last 5,000 characters, saves over 10,000 characters refused, and blank input makes no request; a domain over 100 characters is refused. These are the API's limits.
- Examples in `examples/agent_with_memory.py` and `examples/rag_retriever.py`.
- Requires `langchain-core>=1.4.7,<2`, `mnemoverse>=0.5.2,<0.6.0` and `pydantic>=2.9.1,<3`; the middleware also `langchain>=1.3.0,<2`. Python 3.11 to 3.14.
