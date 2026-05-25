# gemini-agent-toolkit

**LangGraph for Gemini - production AI agents with cost tracking, prompt caching, and structured output in 30 lines.**

[![PyPI](https://img.shields.io/pypi/v/gemini-agent-toolkit.svg)](https://pypi.org/project/gemini-agent-toolkit/)
[![CI](https://github.com/barobaonguyen/gemini-agent-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/barobaonguyen/gemini-agent-toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

```bash
pip install gemini-agent-toolkit
```

[See examples](examples/) | [Quickstart](docs/quickstart.md)

![Terminal session showing an agent loop and cost summary](screenshots/hero.png)

## Why This Exists

LangChain is powerful, but it carries multi-provider abstractions you do not always need. The raw Gemini SDK is clean, but it leaves orchestration, tool schemas, retries, memory, structured output parsing, and cost accounting to every project. `gemini-agent-toolkit` is the middle ground: Gemini-only, Python-only, small enough to understand, and shaped around production chores that repeat in real client work.

## 30-Second Example

```python
from pydantic import BaseModel
from gat import Agent, GeminiClient, tool


class TokenInfo(BaseModel):
    symbol: str
    market_cap_usd: float
    risk_score: int  # 1-10


@tool
def fetch_token(symbol: str) -> dict:
    """Fetch token info from DexScreener.

    Args:
        symbol: Token symbol like PEPE.
    """
    return {"symbol": symbol, "market_cap_usd": 50_000_000, "liquidity_usd": 2_000_000}


client = GeminiClient(model="gemini-2.5-flash")
agent = Agent(client=client, tools=[fetch_token])

result = agent.run(
    task="Get info for PEPE token and assess risk 1-10",
    output_schema=TokenInfo,
)

print(result)
print(client.cost_tracker.summary())
```

Output:

```python
TokenInfo(symbol='PEPE', market_cap_usd=50000000.0, risk_score=6)
{'total_usd': 0.0023, 'calls': 3, 'cached_tokens': 1200, ...}
```

## What's In The Box

- **Agent loop**: a compact orchestration loop with max-iteration enforcement, tool execution, and memory writes.
- **Tool decorator**: `@tool` extracts Python signatures, type hints, docstring descriptions, and OpenAPI-style parameter schemas.
- **Structured output**: `generate_structured()` returns a validated Pydantic model, not a loose dict.
- **Cost tracking**: per-call token accounting with frozen Gemini 2.5 Pro, Flash, and Flash-Lite pricing.
- **Prompt caching helpers**: cache-key utilities plus a thin explicit-cache wrapper for the Google GenAI SDK.
- **Retry primitives**: exponential backoff and a circuit breaker that distinguishes transient 429/503 errors from hard quota/auth failures.

## Reference Examples

| Example | What it does | Demonstrates |
|---|---|---|
| [X Scout](examples/x_scout/) | Pull candidate posts, rank them with Gemini structured output, emit ranked JSON. | Structured output, batch API, cost tracking |
| [On-chain Alerter](examples/onchain_alerter/) | Receive wallet activity, enrich it, classify with Gemini, and optionally send Telegram alerts. | Tool use, memory, retry |
| [News Digest](examples/news_digest/) | Read RSS feeds, dedupe through JSONL memory, summarize into Markdown. | Prompt caching, JSONL replay, structured summaries |

## Cost Optimization

The pricing table in `gat.pricing` is frozen on 2026-05-25 from the Gemini Developer API pricing page. It intentionally tracks a small set of commonly used text models so local estimates are deterministic.

| Model | Input / 1M | Cached input / 1M | Output / 1M | Same 100k in + 10k out |
|---|---:|---:|---:|---:|
| Gemini 2.5 Pro | $1.25 | $0.125 | $10.00 | $0.2250 |
| Gemini 2.5 Flash | $0.30 | $0.03 | $2.50 | $0.0550 |
| Gemini 2.5 Flash-Lite | $0.10 | $0.01 | $0.40 | $0.0140 |

For high-volume classification and ranking, the cheap path is usually Flash-Lite or Flash with `thinking_budget=0`, plus cached system prompts for repeated jobs. A naive Pro pipeline that repeatedly sends the same long context can cost an order of magnitude more than a cached Flash pipeline.

More detail: [Cost Optimization](docs/cost_optimization.md).

## When To Use This

Use this when you are building Gemini-first Python agents and want the repeated production pieces solved without adopting a broad framework. Use raw `google-genai` when you only need one or two calls. Use LangChain or LangGraph when you need a large ecosystem of integrations, graph orchestration, or multi-provider portability.

See [Comparison](docs/comparison.md).

## Install And Quickstart

```bash
pip install gemini-agent-toolkit
export GEMINI_API_KEY="..."
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="..."
```

Then follow [docs/quickstart.md](docs/quickstart.md).

## Contributing

Issues and small PRs are welcome. Keep the project Gemini-native, Python-only, and focused on orchestration, structured output, caching, retries, memory, and cost visibility. RAG integrations, browser automation, and multi-provider abstractions are intentionally outside v0.1.

## License

MIT

