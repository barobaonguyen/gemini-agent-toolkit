from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import feedparser
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from gat import GeminiClient, JsonlStore
from gat.cache import prompt_cache_key


SYSTEM_PROMPT = (
    "You write concise daily AI and developer news digests. Prefer actionable "
    "details over hype. Merge duplicates and preserve source links."
)


class ArticleSummary(BaseModel):
    title: str
    summary: str
    why_it_matters: str
    link: str


def load_feeds(path: Path) -> list[dict[str, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    feeds = data.get("feeds", []) if isinstance(data, dict) else []
    return [feed for feed in feeds if isinstance(feed, dict) and "url" in feed]


def fetch_entries(feeds: list[dict[str, str]], limit: int = 10) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for feed in feeds:
        parsed = feedparser.parse(feed["url"])
        for entry in parsed.entries[:limit]:
            entries.append(
                {
                    "feed": feed.get("name", feed["url"]),
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                }
            )
    return entries


def main() -> None:
    load_dotenv()
    client = GeminiClient(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    memory = JsonlStore("seen_articles.jsonl")
    seen = {record.get("link") for record in memory.replay()}
    new_entries = [entry for entry in fetch_entries(load_feeds(Path("feeds.yaml"))) if entry["link"] not in seen]

    summaries: list[ArticleSummary] = []
    for entry in new_entries[:15]:
        prompt = (
            f"Cache key: {prompt_cache_key(SYSTEM_PROMPT)}\n"
            "Summarize this article for a technical founder.\n"
            f"Article: {entry}"
        )
        summary = client.generate_structured(prompt, ArticleSummary, system=SYSTEM_PROMPT)
        assert isinstance(summary, ArticleSummary)
        summaries.append(summary)
        memory.add({"link": entry["link"], "title": entry["title"]})

    output = ["# Daily Digest", ""]
    for item in summaries:
        output.append(f"## {item.title}")
        output.append("")
        output.append(item.summary)
        output.append("")
        output.append(f"Why it matters: {item.why_it_matters}")
        output.append("")
        output.append(f"Source: {item.link}")
        output.append("")
    Path("digest.md").write_text("\n".join(output), encoding="utf-8")
    print(f"wrote digest.md with {len(summaries)} items")
    print(client.cost_tracker.summary())


if __name__ == "__main__":
    main()

