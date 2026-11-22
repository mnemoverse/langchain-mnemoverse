# langchain-mnemoverse

Long-term memory for [LangChain](https://github.com/langchain-ai/langchain) agents and chains, backed by [Mnemoverse](https://mnemoverse.com). The memory lives in your Mnemoverse account, so what an agent saves here is also there in Claude, ChatGPT, Cursor, VS Code and any other tool connected to the same account.

- **`MnemoverseRetriever`** returns the memories relevant to a query as LangChain `Document` objects, for chains and RAG.
- **`MnemoverseToolkit`** gives an agent three tools: `mnemoverse_search`, `mnemoverse_save` and `mnemoverse_rate`. Rating a recalled memory as helpful or misleading changes what comes back next for similar questions. The tools are also importable one by one: `MnemoverseSearchTool`, `MnemoverseSaveTool`, `MnemoverseRateTool`.
- **`MnemoverseMiddleware`** recalls the memories relevant to the user's latest message once per `create_agent` run and adds them to the system message, within a time budget. If recall is slow or fails, the run goes on without memory.

This integration is maintained by the Mnemoverse team, who run the Mnemoverse memory service.

## Installation

```bash
pip install langchain-mnemoverse
```

The middleware needs the `langchain` package, which the retriever and the tools do not:

```bash
pip install "langchain-mnemoverse[agents]"
```

or with uv:

```bash
uv add "langchain-mnemoverse[agents]"
```

Create an API key at [console.mnemoverse.com](https://console.mnemoverse.com/dashboard/keys) and set it in the environment, or pass `api_key=`. With neither, and no client passed in (see [Settings](#settings)), building any of the components raises `ValueError`.

```bash
export MNEMOVERSE_API_KEY=mk_live_...
```

## Usage

### An agent with memory

The example below needs a chat model; it uses OpenAI through `create_agent`'s model string, which needs `pip install langchain-openai` and `OPENAI_API_KEY`. Any LangChain chat model works the same way.

```python
from langchain.agents import create_agent

from langchain_mnemoverse import MnemoverseMiddleware, MnemoverseToolkit

agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=MnemoverseToolkit(domain="user:ana").get_tools(),
    middleware=[MnemoverseMiddleware(domain="user:ana")],
    system_prompt="You are a helpful assistant.",
)

agent.invoke({"messages": [{"role": "user", "content": "Remember that I always take the window seat."}]})
result = agent.invoke({"messages": [{"role": "user", "content": "Which seat do I take on flights?"}]})
print(result["messages"][-1].content)
```

The middleware reads once at the start of every run, so a tool loop does not read again. The agent saves only when it calls `mnemoverse_save`, which its description limits to things the user asks it to remember.

### A retriever for chains

```python
from langchain_mnemoverse import MnemoverseRetriever

retriever = MnemoverseRetriever(domain="project:docs", k=4)

for doc in retriever.invoke("Which database does the ledger use?"):
    print(doc.id, doc.metadata["relevance"], doc.page_content)
```

Each document's `id` is the memory's id. `metadata` holds `atom_id`, `relevance`, `similarity`, `valence`, `importance`, `domain`, `concepts`, `created_at`, `source`, `provenance`, `metadata` (what was saved with the memory) and `superseded_by`. `ainvoke` works the same way.

### Calling a tool directly

```python
from langchain_mnemoverse import MnemoverseSearchTool

search = MnemoverseSearchTool(domain="user:ana")
message = search.invoke(
    {"type": "tool_call", "id": "1", "name": "mnemoverse_search", "args": {"query": "seat", "limit": 3}}
)
print(message.content)
```

The answer lists one memory per line with its id, which is what `mnemoverse_rate` takes.

## Which memory a run uses

Every call reads and writes one Mnemoverse **domain**, a namespace inside your account such as `user:ana` or `project:support`, of up to 100 characters.

- **One user:** pass `domain="user:ana"` to the components.
- **Several users in one app:** leave `domain` out of the tools and the middleware, and name the domain per run:

  ```python
  agent.invoke(
      {"messages": [{"role": "user", "content": "Which seat do I take?"}]},
      config={"configurable": {"mnemoverse_domain": f"user:{user_id}"}},
  )
  ```

  The middleware also reads `mnemoverse_domain` from the run's `context` (a mapping key or an attribute). The retriever takes `domain=` per call, `retriever.invoke(query, domain=f"user:{user_id}")`, or as a configurable field: `retriever.configurable_fields(domain=ConfigurableField(id="mnemoverse_domain"))`.

Take `user_id` from your app's signed-in user on the server, never from client input or model output. The domain is never an argument the model can set. A run with no domain gets no memory: the tools answer the model with an error and the middleware logs one warning; it is never served from another domain.

A domain separates how memory is used, not who may read it: the API key can read every domain of its account. Give each app its own key, so you can revoke one without the others. A shared room's address, `xroom:<room_id>`, works as a domain. How Mnemoverse handles keys and data: [mnemoverse.com/docs/api/security](https://mnemoverse.com/docs/api/security).

## Settings

Every component takes `api_key` (default: `MNEMOVERSE_API_KEY`), `base_url` (default `https://core.mnemoverse.com`), and `client` and `async_client` for a ready `mnemoverse.MnemoClient` and `AsyncMnemoClient`. Sync calls (`invoke`) use the sync client and async calls (`ainvoke`) the async one.

- **Neither passed:** the package opens its own clients from the key. Components built with the same key and `base_url` share them, including the copies `configurable_fields` makes for each call: one sync client, and one async client per event loop, so a component also works from several `asyncio.run()` calls.
- **Only `client`:** async calls run it in a worker thread.
- **Only `async_client`:** sync calls raise `ValueError` unless you also pass `api_key=`. They never fall back to `MNEMOVERSE_API_KEY`, which may belong to another account than your client.
- An `AsyncMnemoClient` you pass in works on one event loop: SDK 0.5.2 binds its connections to the loop that first used it, and from a second `asyncio.run()` its calls fail with "Event loop is closed". Pass one per loop, or let the package open its own.

Clients you pass in are never closed by this package.

| `MnemoverseRetriever` | Default | What it does |
|---|---|---|
| `domain` | required | The domain every call reads; override per call with `invoke(query, domain=...)` |
| `k` | 4 | Documents per call, 1 to 500; override per call with `invoke(query, k=...)` |
| `min_relevance` | 0.3 | Lowest relevance, 0 to 1, of a returned memory |

| Tools and `MnemoverseToolkit` | Default | What it does |
|---|---|---|
| `domain` | `None` | The domain every call uses; without it, `config["configurable"]["mnemoverse_domain"]` |
| `save` (toolkit) | `True` | Include `mnemoverse_save` |
| `rate` (toolkit) | `True` | Include `mnemoverse_rate` |
| `limit` (search argument, set by the model) | 5 | Memories per search, 1 to 50 |

There is no delete tool. The toolkit's tools share one pair of clients.

| `MnemoverseMiddleware` | Default | What it does |
|---|---|---|
| `domain` | `None` | The domain every run reads; without it, the run's `context` or `config` |
| `top_k` | 5 | Memories per recall, 1 to 500 |
| `min_relevance` | 0.3 | Lowest relevance, 0 to 1, of a recalled memory. A broad message ("what do you know about me?") scores lower than a specific one; lower this if broad messages recall nothing |
| `recall_timeout` | 8.0 | Seconds the run waits for recall before it goes on without it; also how long it waits for a capture. A call that runs longer is not cancelled, it finishes in the background |
| `context_prompt` | a heading and one line | Text above the recalled memories. The default line tells the model the memories are reference data, not instructions; keep it if you write your own |
| `capture` | `False` | Save the user's messages of each run as one memory. Prefer the `mnemoverse_save` tool, which saves only what the user asks to keep |

At most `k`, `limit` or `top_k` memories are used, also when the server returns more.

Memory text is written by other people and tools too: members of a shared room, the other AI tools on the account. In the middleware's memory block and in the search tool's answer each memory is therefore one line, with line breaks turned into spaces and `<` and `>` into `‹` and `›`, so a memory cannot close the block or pose as another memory with an id the server never returned. The retriever's documents keep the text unchanged.

## Errors

- **Retriever:** SDK errors propagate unchanged (`MnemoAuthError`, `MnemoRateLimitError`, `MnemoUnavailableError`, `MnemoError`), so a chain fails loudly; wrap it with `.with_fallbacks(...)` if you prefer. An invalid domain or `k` raises `ValueError`.
- **Tools:** every failure reaches the model as a `ToolMessage` with `status="error"` and the server's reason, so the agent can tell the user what happened.
- **Middleware:** memory never breaks a run. A timeout or a failed call is logged under the `langchain_mnemoverse` logger and the run continues without memory. A refusal (401 or 403) is logged with the server's reason, and the middleware stops recall, or capture, for that domain only.

Long input is kept within the API limits: queries are cut to their last 5,000 characters, the save tool refuses a memory over 10,000 characters, and capture saves the first 10,000.

## Cost

Each retriever call and each tool call makes at most one request; a save refused for its length and a rating with no ids make none. The middleware makes at most one read per agent run, and one write more with `capture=True`. Reads, writes and ratings count together against your plan's daily query limit: the Free plan allows 1,000 queries a day. Plans and limits are on [mnemoverse.com/pricing](https://mnemoverse.com/pricing).

## Examples

[`examples/agent_with_memory.py`](examples/agent_with_memory.py) is an agent with the middleware and the toolkit, and [`examples/rag_retriever.py`](examples/rag_retriever.py) a retriever in a small chain. Both need `MNEMOVERSE_API_KEY` and `OPENAI_API_KEY`:

```bash
pip install "langchain-mnemoverse[agents]" langchain-openai
python examples/agent_with_memory.py
```

## Compatibility

Tested with the `mnemoverse` Python SDK 0.5.2, `langchain-core` 1.8.4, `langchain` 1.5.2 and `langchain-tests` 1.1.9 on Python 3.11, 3.12, 3.13 and 3.14, and with the lowest supported versions, `langchain-core` 1.4.7 and `langchain` 1.3.0, on Python 3.11. The tests use fake SDK clients; a live suite runs on demand. It needs SDK 0.5.2 or later: in earlier versions a call refused with a client error (such as 401, 403 or a quota 429) while the client was probing the service after an outage could leave that client refusing every call until the process restarted.

## Links

- Python SDK: [mnemoverse on PyPI](https://pypi.org/project/mnemoverse/), [docs](https://mnemoverse.com/docs/api/python-sdk)
- Pricing and limits: [mnemoverse.com/pricing](https://mnemoverse.com/pricing)
- Security and data handling: [mnemoverse.com/docs/api/security](https://mnemoverse.com/docs/api/security)
- Issues: [github.com/mnemoverse/langchain-mnemoverse/issues](https://github.com/mnemoverse/langchain-mnemoverse/issues/new)

## License

MIT, as LangChain. See [LICENSE](LICENSE).
