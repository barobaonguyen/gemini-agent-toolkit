from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from gat import Agent, GeminiClient, JsonlStore, retry, tool


class WalletAlert(BaseModel):
    wallet: str
    action: str
    token_symbol: str
    usd_value: float
    confidence: int = Field(ge=1, le=10)
    message: str


@retry(attempts=3, base_delay_s=1)
def helius_lookup(wallet: str) -> dict[str, Any]:
    api_key = os.getenv("HELIUS_API_KEY")
    if not api_key:
        return {
            "wallet": wallet,
            "recent_transfers": [
                {"token_symbol": "PEPE", "usd_value": 50_000, "side": "buy"},
            ],
        }
    url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
    response = httpx.get(url, params={"api-key": api_key, "limit": 5}, timeout=20)
    response.raise_for_status()
    return {"wallet": wallet, "recent_transactions": response.json()}


@tool
def lookup_wallet(wallet: str) -> dict[str, Any]:
    """Look up recent wallet activity.

    Args:
        wallet: Public wallet address to inspect.
    """
    return helius_lookup(wallet)


def send_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(text)
        return
    response = httpx.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=15,
    )
    response.raise_for_status()


def main() -> None:
    load_dotenv()
    client = GeminiClient(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    memory = JsonlStore("alerts_memory.jsonl")
    agent = Agent(client=client, tools=[lookup_wallet], memory=memory, max_iterations=5)
    wallet = os.getenv("WATCH_WALLETS", "DemoWallet111111111111111111111111111111111").split(",")[0]
    alert = agent.run(
        task=(
            f"Inspect wallet {wallet}. If recent activity looks like a smart-money buy, "
            "produce a concise Telegram alert."
        ),
        output_schema=WalletAlert,
    )
    assert isinstance(alert, WalletAlert)
    send_telegram(alert.message)
    print(json.dumps(client.cost_tracker.summary(), indent=2))


if __name__ == "__main__":
    main()

