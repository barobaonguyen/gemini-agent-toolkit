# News Digest

Read RSS feeds, dedupe links through JSONL memory, summarize with Gemini, and write `digest.md`.

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `GEMINI_API_KEY`, edit `feeds.yaml`, then run:

```bash
python main.py
```

