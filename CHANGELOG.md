# Changelog

## v0.1.0 - 2026-05-25

Initial public release.

- Added `GeminiClient` with text generation, Pydantic structured output, batch calls, and token accounting.
- Added `Agent` loop with tool execution, memory writes, and max-iteration enforcement.
- Added `@tool`, `ToolRegistry`, Pydantic schema conversion, JSONL memory, retry/circuit breaker helpers, and prompt cache utilities.
- Added X scout, on-chain alerter, and news digest reference examples.
- Added CI for Ruff, mypy, and pytest with coverage.

Roadmap:

- v0.2: streaming agent events.
- v0.2: parallel tool execution.
- v0.3: SQLite and Redis memory backends.
- v0.3: trajectory replay UI.

