from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from gat import GeminiClient


class RankedPost(BaseModel):
    url: str
    author: str
    score: int = Field(ge=1, le=10)
    reason: str


def sample_posts() -> list[dict[str, Any]]:
    return [
        {
            "url": "https://x.com/example/status/1",
            "author": "builder_a",
            "text": "Looking for someone to automate daily Gemini summaries for our research desk.",
        },
        {
            "url": "https://x.com/example/status/2",
            "author": "builder_b",
            "text": "Shipping another generic AI wrapper today.",
        },
    ]


def fetch_apify_posts() -> list[dict[str, Any]]:
    token = os.getenv("APIFY_TOKEN")
    dataset_id = os.getenv("APIFY_DATASET_ID")
    if not token or not dataset_id:
        return sample_posts()
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    response = httpx.get(url, params={"token": token, "clean": "true"}, timeout=30)
    response.raise_for_status()
    items = response.json()
    return items if isinstance(items, list) else sample_posts()


def prompt_for(post: dict[str, Any]) -> str:
    return f"""
Rank this X post as a potential freelance AI-agent lead.

Score 1 means noise. Score 10 means strong buying intent for Gemini agent work.

Return JSON with url, author, score, and reason.

Post:
{json.dumps(post, ensure_ascii=False)}
""".strip()


def main() -> None:
    load_dotenv()
    client = GeminiClient(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    posts = fetch_apify_posts()
    ranked = client.batch([prompt_for(post) for post in posts], schema=RankedPost, concurrency=4)
    ranked_posts = sorted(
        (item.model_dump() for item in ranked if isinstance(item, RankedPost)),
        key=lambda item: item["score"],
        reverse=True,
    )
    print(json.dumps(ranked_posts, indent=2, ensure_ascii=False))
    print(json.dumps(client.cost_tracker.summary(), indent=2))


if __name__ == "__main__":
    main()

