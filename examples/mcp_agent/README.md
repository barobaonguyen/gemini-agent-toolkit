# MCP Agent Example

This example runs a tiny local MCP stdio server and exposes its tools to a GAT
agent.

```bash
pip install -e ".[mcp]"
export GEMINI_API_KEY="..."
python examples/mcp_agent/main.py
```

PowerShell:

```powershell
$env:GEMINI_API_KEY="..."
python examples/mcp_agent/main.py
```

Optional environment variables:

- `GEMINI_MODEL`: model id, defaults to `gemini-2.5-flash`
- `GAT_TASK`: task prompt sent to the agent
- `GAT_PLANNER`: `default` or `react`, defaults to `react`
