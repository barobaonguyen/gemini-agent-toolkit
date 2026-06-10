# Eval Suite Example

A small golden-set suite for the `gat eval` harness (`gat.eval`).

## Run it

```bash
export GEMINI_API_KEY="..."        # PowerShell: $env:GEMINI_API_KEY="..."
gat eval examples/eval_suite/cases.yaml
```

Or from Python:

```python
from gat import Agent, GeminiClient, evaluate, load_cases

agent = Agent(client=GeminiClient(model="gemini-2.5-flash"))
report = evaluate(agent, load_cases("examples/eval_suite/cases.yaml"))
print(report.to_dict()["pass_rate"], report.total_usd)
```

## Case format

Each case is `prompt` + `expected` with one `matcher`:

| matcher     | passes when                                                        |
|-------------|-------------------------------------------------------------------|
| `exact`     | output equals `expected` (trimmed; case-insensitive by default)   |
| `contains`  | `expected` is a substring of the output (the default)             |
| `regex`     | `expected`, treated as a regex, matches the output                |
| `llm_judge` | a Gemini judge rules the answer substantively correct             |

Set `case_sensitive: true` to make `exact`/`contains`/`regex` case-sensitive.

Both `cases.yaml` and `cases.json` are loadable — `load_cases` picks the parser
from the file extension. The suite reads no secrets and contains only generic,
public questions.
