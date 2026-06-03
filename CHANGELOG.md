# Changelog

## v0.2.0 - 2026-06-03

- Added `GeminiClient.stream()` and `gat.stream.stream_generate()` for text chunk streaming with retry-aware stream startup and final usage accounting.
- Implemented `Agent.stream()` events for streamed chunks, tool calls, tool results, and final outputs.
- Added `google_search_grounding_tool(client)` for Gemini Google Search grounding with returned queries, source metadata, and token cost tracking.
- Added the `examples/research_agent/` walkthrough showing decomposition, grounded search, streamed synthesis, citations, and JSONL memory.
- Refreshed pricing source date against the Gemini Developer API pricing page.

## v0.1.0 - 2026-05-25

Initial public release.

- Added `GeminiClient` with text generation, Pydantic structured output, batch calls, and token accounting.
- Added `Agent` loop with tool execution, memory writes, and max-iteration enforcement.
- Added `@tool`, `ToolRegistry`, Pydantic schema conversion, JSONL memory, retry/circuit breaker helpers, and prompt cache utilities.
- Added X scout, on-chain alerter, and news digest reference examples.
- Added CI for Ruff, mypy, and pytest with coverage.

Roadmap:

- v0.3: SQLite and Redis memory backends.
- v0.3: parallel tool execution.
- v0.3: trajectory replay UI.
