# Comparison

## Use `gemini-agent-toolkit` When

- You are committed to Gemini for the project.
- You want Python-first agent code with minimal framework weight.
- You need structured output, tool schemas, retries, memory, and cost accounting.
- You want examples that are close to production jobs: scouting, alerts, and digests.

## Use Raw `google-genai` When

- You only need one or two calls.
- You do not need an agent loop.
- You prefer to own every retry, schema, and cost-accounting detail.

Raw SDK code is the right baseline. This package is for the point where those repeated details start spreading through your app.

## Use LangChain When

- You need a large integration ecosystem.
- You need retrievers, loaders, vector stores, and many model providers.
- Team familiarity matters more than a small dependency surface.

LangChain is broader. `gemini-agent-toolkit` is deliberately narrower.

## Use LangGraph When

- You need explicit graph state, branching, replay, persistence, and human-in-the-loop checkpoints.
- Your agent has complex multi-step workflow control.
- You expect to visualize or audit graph transitions.

This package is closer to a compact agent loop than a full graph runtime.

## Non-Goals

- Multi-provider abstraction.
- Vector store integrations.
- Browser or computer-use automation.
- TypeScript port.

