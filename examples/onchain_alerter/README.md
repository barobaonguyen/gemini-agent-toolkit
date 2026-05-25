# On-chain Alerter

Classify wallet activity with Gemini and optionally send Telegram alerts.

This example keeps chain logic intentionally generic. It demonstrates the agent loop, a wallet lookup tool, retry-safe HTTP calls, and memory-backed dedupe without shipping tuned cluster thresholds or private wallet lists.

## Setup

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `GEMINI_API_KEY`. `HELIUS_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID` are optional.

## Run

```bash
python main.py
```

Webhook demo:

```bash
python webhook_server.py
```

