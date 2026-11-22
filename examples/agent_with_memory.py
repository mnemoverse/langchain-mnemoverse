"""An agent that remembers between runs: MnemoverseMiddleware recalls before each run, the
toolkit lets the model search, save and rate.

Needs:
    pip install "langchain-mnemoverse[agents]" langchain-openai
    export MNEMOVERSE_API_KEY=mk_live_...
    export OPENAI_API_KEY=...

Run:
    python examples/agent_with_memory.py [user_id]

Each run of the agent costs one Mnemoverse read, plus one request per tool call the model makes.
"""

from __future__ import annotations

import sys

from langchain.agents import create_agent

from langchain_mnemoverse import MnemoverseMiddleware, MnemoverseToolkit


def main() -> None:
    user_id = sys.argv[1] if len(sys.argv) > 1 else "demo"
    # Both open their clients from MNEMOVERSE_API_KEY. The middleware keeps its own, so a recall
    # that outlives its time budget never holds up the agent's tool calls. If you pass clients
    # in, give the middleware one the tools do not use.
    agent = create_agent(
        model="openai:gpt-4.1-mini",  # any LangChain chat model works
        tools=MnemoverseToolkit().get_tools(),
        middleware=[MnemoverseMiddleware()],
        system_prompt="You are a helpful assistant. Keep answers short.",
    )
    # The domain comes from the signed-in user on the server, never from the model.
    config = {"configurable": {"mnemoverse_domain": f"user:{user_id}"}}

    for text in ("Remember that I always take the window seat.", "Which seat do I take on flights?"):
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
        print(f"> {text}\n{result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
