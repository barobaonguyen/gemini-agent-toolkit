"""Run a Gemini agent with tools loaded from a local MCP stdio server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from gat import Agent, GeminiClient, load_mcp_tools


def main() -> None:
    server = Path(__file__).with_name("server.py")
    tools = load_mcp_tools(sys.executable, [str(server)], name_prefix="fixture")
    client = GeminiClient(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    agent = Agent(
        client=client,
        tools=tools,
        planner=os.getenv("GAT_PLANNER", "react"),  # type: ignore[arg-type]
        max_iterations=5,
    )
    task = os.getenv("GAT_TASK", "Use the fixture echo tool to say hello, then summarize.")
    print(agent.run(task))
    print(client.cost_tracker.summary())


if __name__ == "__main__":
    main()
