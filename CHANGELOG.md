# Changelog

## v0.3.0 - 2026-06-10

- Added `Agent.arun()`, an async agent loop that executes multiple tool calls from a single turn concurrently via `asyncio.gather`, while preserving cost accounting and per-tool retry. Backed by `GeminiClient.agenerate()` / `agenerate_structured()` and `ToolRegistry.aexecute()` (async tools awaited directly, sync tools run in worker threads).
- Added `gat.eval`: a golden-set eval harness with `exact` / `contains` / `regex` / optional Gemini `llm_judge` matchers, JSON or YAML case loading, and a pass-rate + total-cost report. Shipped a fixture suite at `examples/eval_suite/`.
- Added a `gat` console command (`gat run`, `gat eval`, `gat cost`) via `console_scripts`.
- Added a SQLite memory backend (`SqliteStore`) alongside JSONL, plus `build_memory()` so the backend is selectable from config or the CLI (`--memory {memory,jsonl,sqlite}`). All backends share the same `add` / `replay` / `clear` contract.
- Refreshed the pricing source date; Gemini 2.5 Pro / Flash / Flash-Lite standard-tier rates verified unchanged.

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

- v0.3 (shipped): SQLite memory backend and parallel tool execution.
- Later: Redis memory backend; trajectory replay UI.
