"""Tiny local MCP stdio server fixture for the MCP agent example."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gat-local-fixture")


@mcp.tool()
def echo(text: str) -> str:
    """Echo a string back to the caller."""

    return f"echo: {text}"


@mcp.tool()
def add(left: int, right: int) -> int:
    """Add two integers."""

    return left + right


if __name__ == "__main__":
    mcp.run()
