from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from gat import Agent, GeminiClient, JsonlStore, google_search_grounding_tool


SYSTEM_PROMPT = """
You are a precise research agent. Break broad questions into specific searches,
use grounded sources, and synthesize a concise answer with Markdown citations.
Never invent URLs. Prefer source links returned by tools.
""".strip()


def research_task(question: str) -> str:
    return f"""
Question: {question}

Workflow:
1. Decompose the question into 2-4 focused subquestions.
2. Call google_search for each subquestion that needs fresh evidence.
3. Synthesize the final answer in Markdown with inline citations and a Sources section.

When calling a tool, respond only as tool_call JSON. When done, return the final answer as plain text.
""".strip()


def print_event(event_type: str, payload: dict[str, Any]) -> None:
    if event_type == "chunk":
        print(str(payload["text"]), end="", flush=True)
        return
    if event_type == "tool_call":
        print(f"\n\n[tool] {payload['name']} {json.dumps(payload['args'], ensure_ascii=False)}")
        return
    if event_type == "tool_result":
        result = payload.get("result", {})
        sources = result.get("sources", []) if isinstance(result, dict) else []
        print(f"[tool] returned {len(sources)} source(s)\n")
        return
    if event_type == "final":
        print("\n\n[cost]")


def main() -> None:
    load_dotenv()
    client = GeminiClient(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    memory = JsonlStore(os.getenv("RESEARCH_MEMORY_PATH", "research_memory.jsonl"))
    search = google_search_grounding_tool(client)
    agent = Agent(
        client=client,
        tools=[search],
        memory=memory,
        max_iterations=8,
        system=SYSTEM_PROMPT,
    )
    question = os.getenv(
        "RESEARCH_QUESTION",
        "What changed in Gemini API search grounding recently?",
    )

    for event in agent.stream(research_task(question)):
        print_event(event.type, event.payload)

    print(json.dumps(client.cost_tracker.summary(), indent=2))
    print(f"\nMemory: {os.getenv('RESEARCH_MEMORY_PATH', 'research_memory.jsonl')}")


if __name__ == "__main__":
    main()
