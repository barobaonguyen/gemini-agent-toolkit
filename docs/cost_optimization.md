# Cost Optimization

`gemini-agent-toolkit` keeps a frozen pricing table in `gat.pricing` so local cost estimates stay reproducible. The v0.2 table was checked on 2026-06-02 against the official Gemini Developer API pricing page:

https://ai.google.dev/gemini-api/docs/pricing

## Default To The Cheapest Model That Works

| Workload | Suggested model | Notes |
|---|---|---|
| Classification, extraction, routing | `gemini-2.5-flash-lite` | Lowest text input/output cost. |
| General agent steps and ranking | `gemini-2.5-flash` | Good default for speed and quality. |
| Hard reasoning and code review | `gemini-2.5-pro` | Use selectively, especially if output is long. |

## Disable Thinking When You Do Not Need It

For Flash-family models, use `thinking_budget=0` when the task is extraction, tagging, ranking, or summarization that does not need long reasoning.

```python
client.generate(
    "Classify this post into one of: lead, noise, competitor.",
    thinking_budget=0,
    temperature=0.1,
)
```

## Cache Stable Context

If every call repeats the same long policy, rubric, examples, or docs, split stable context from the changing input. Gemini has implicit caching for repeated large prompts, and the SDK also supports explicit cached content.

```python
from gat.cache import ExplicitPromptCache, should_explicitly_cache

rubric = open("ranking_rubric.md", encoding="utf-8").read()
if should_explicitly_cache(rubric):
    cache = ExplicitPromptCache(client._ensure_client())
    cached = cache.create(model="gemini-2.5-flash", contents=rubric)
```

## Batch Similar Jobs

`GeminiClient.batch()` keeps your code simple while capping concurrency.

```python
ranked = client.batch(prompts, schema=RankedPost, concurrency=4)
```

## Example Cost Shape

For a 100k-token shared context and 10k-token output:

| Strategy | Approx cost |
|---|---:|
| Pro, no cache | $0.2250 |
| Flash, no cache | $0.0550 |
| Flash-Lite, no cache | $0.0140 |
| Flash with 80k cached input | $0.0334 |
| Flash-Lite with 80k cached input | $0.0068 |

The exact savings depend on prompt length, output length, model selection, and whether the cached portion is actually reused.
