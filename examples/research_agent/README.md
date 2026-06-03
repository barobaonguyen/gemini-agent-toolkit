# Research Agent

Multi-step research agent that decomposes a question, uses Gemini Google Search grounding as an agent tool, writes run memory, and streams agent events.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `GEMINI_API_KEY` in `.env`, then run:

```bash
python main.py
```

PowerShell one-off question:

```powershell
$env:RESEARCH_QUESTION="What are the latest Gemini API grounding rules?"
python main.py
```

## What It Demonstrates

- `Agent.stream(...)` event streaming for model turns, tool calls, tool results, and final output.
- `google_search_grounding_tool(client)` as an agent-callable search helper.
- `JsonlStore` memory for replaying task, assistant, tool, and final records.
- Final synthesis with source links returned by Gemini grounding metadata.

Google Search grounding is a Gemini API feature and may have separate billing rules from token usage. Review the Gemini pricing page before running high-volume research jobs.
