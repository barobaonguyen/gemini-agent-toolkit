"""Frozen Gemini pricing used by :mod:`gat.cost`.

Prices are USD per 1M tokens for the Gemini Developer API paid standard tier.
They are intentionally frozen for reproducible local accounting and should be
updated by hand when releasing a new package version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

PRICING_SOURCE_URL: Final[str] = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_SOURCE_DATE: Final[str] = "2026-06-10"


@dataclass(frozen=True)
class ModelPricing:
    """Token prices in USD per 1M tokens."""

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float = 0.0
    notes: str = ""


PRICING: Final[dict[str, ModelPricing]] = {
    "gemini-2.5-pro": ModelPricing(
        input_per_1m=1.25,
        output_per_1m=10.00,
        cached_input_per_1m=0.125,
        notes="Standard tier, prompts <= 200k tokens.",
    ),
    "gemini-2.5-flash": ModelPricing(
        input_per_1m=0.30,
        output_per_1m=2.50,
        cached_input_per_1m=0.03,
        notes="Standard tier, text/image/video input.",
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        input_per_1m=0.10,
        output_per_1m=0.40,
        cached_input_per_1m=0.01,
        notes="Standard tier, text/image/video input.",
    ),
}


def normalize_model(model: str) -> str:
    """Normalize SDK model names such as ``models/gemini-2.5-flash``."""

    model = model.strip()
    if model.startswith("models/"):
        return model.split("/", 1)[1]
    return model


def get_pricing(model: str) -> ModelPricing | None:
    """Return frozen pricing for a model, if known."""

    return PRICING.get(normalize_model(model))


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Estimate USD cost for a single call.

    Unknown models return ``0.0`` instead of raising so callers can still track
    token totals when Google launches a model before this table is updated.
    """

    pricing = get_pricing(model)
    if pricing is None:
        return 0.0
    cached = min(max(cached_tokens, 0), max(input_tokens, 0))
    billable_input = max(input_tokens, 0) - cached
    output = max(output_tokens, 0)
    return (
        billable_input * pricing.input_per_1m
        + cached * pricing.cached_input_per_1m
        + output * pricing.output_per_1m
    ) / 1_000_000
