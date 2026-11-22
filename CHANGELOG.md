# Changelog

All notable changes to `langchain-mnemoverse`.

## [0.1.0] - unreleased

### Added

- `MnemoverseRetriever`: memories from one Mnemoverse domain as LangChain documents, at most `k` per call (default 4) also when the server returns more; `k` and `domain` can be overridden per call.
- `MnemoverseSearchTool`, `MnemoverseSaveTool` and `MnemoverseRateTool` (`mnemoverse_search`, `mnemoverse_save`, `mnemoverse_rate`), and `MnemoverseToolkit`, which builds them around one pair of SDK clients. The domain is fixed or comes from `config["configurable"]["mnemoverse_domain"]`, never from the model; a call with no domain fails closed. Failures reach the model as error tool messages with the server's reason.
- `MnemoverseMiddleware` for `create_agent` (extra `agents`): one recall per run within `recall_timeout`, added to the system message of every model call in that run; optional capture of the user's messages. Memory never breaks a run.
- Queries are cut to their last 5,000 characters, saves over 10,000 characters are refused and a domain over 100 characters is refused, the API's limits.
- Examples in `examples/agent_with_memory.py` and `examples/rag_retriever.py`.
- Requires `langchain-core>=1.4.7,<2` and `mnemoverse>=0.5.2,<0.6.0`; the middleware also `langchain>=1.3.0,<2`. Python 3.11 to 3.14.
