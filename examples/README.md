# Examples

- [`x_scout`](x_scout/): rank candidate X posts with Gemini structured output.
- [`onchain_alerter`](onchain_alerter/): classify wallet activity and send Telegram-ready alerts.
- [`news_digest`](news_digest/): dedupe RSS entries and write a Markdown digest.
- [`research_agent`](research_agent/): decompose a question, search with Gemini grounding, stream the synthesis, and persist memory.
- [`mcp_agent`](mcp_agent/): load tools from a tiny local MCP stdio server fixture.

Most examples have their own `.env.example` and `requirements.txt`. MCP support
uses the package extra: `pip install -e ".[mcp]"`.
