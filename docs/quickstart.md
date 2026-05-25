# Quickstart

## Install

```bash
pip install gemini-agent-toolkit
```

For local development:

```bash
git clone https://github.com/barobaonguyen/gemini-agent-toolkit
cd gemini-agent-toolkit
python -m pip install -e ".[dev,examples]"
```

Set a Gemini key:

```bash
export GEMINI_API_KEY="..."
```

PowerShell:

```powershell
$env:GEMINI_API_KEY="..."
```

## Build A Small Agent

```python
from pydantic import BaseModel
from gat import Agent, GeminiClient, tool


class LeadScore(BaseModel):
    company: str
    score: int
    reason: str


@tool
def lookup_company(company: str) -> dict:
    """Look up company metadata.

    Args:
        company: Company name to inspect.
    """
    return {"company": company, "industry": "fintech", "team_size": 12}


client = GeminiClient(model="gemini-2.5-flash")
agent = Agent(client=client, tools=[lookup_company], max_iterations=5)

result = agent.run("Score AcmePay as a freelance AI lead.", output_schema=LeadScore)
print(result)
print(client.cost_tracker.summary())
```

## Use The Client Directly

```python
from pydantic import BaseModel
from gat import GeminiClient


class Summary(BaseModel):
    title: str
    bullets: list[str]


client = GeminiClient(model="gemini-2.5-flash-lite")
summary = client.generate_structured(
    "Summarize why prompt caching matters for repeated daily jobs.",
    Summary,
    thinking_budget=0,
)
```

## Add Memory

```python
from gat import JsonlStore

memory = JsonlStore("runs/latest.jsonl")
agent = Agent(client=client, tools=[lookup_company], memory=memory)
agent.run("Score AcmePay.")

print(memory.replay())
```

## Run Tests

```bash
ruff check src/ tests/
mypy src/gat
pytest -q --cov=gat --cov-fail-under=80
```

