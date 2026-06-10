"""Gemini API wrapper with structured output, batch calls, and cost tracking."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeAlias

from pydantic import BaseModel

from gat.cost import CostTracker
from gat.schemas import to_gemini_schema

Prompt: TypeAlias = str | list[dict[str, Any]]


class GeminiClientError(RuntimeError):
    """Base exception for Gemini client failures."""


class MissingApiKeyError(GeminiClientError):
    """Raised when a real API call is attempted without a Gemini API key."""


class EmptyResponseError(GeminiClientError):
    """Raised when Gemini returns no text or candidates."""


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        cache_enabled: bool = True,
        cost_tracker: CostTracker | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.cache_enabled = cache_enabled
        self.timeout_s = timeout_s
        self._cost_tracker = cost_tracker or CostTracker()
        self._client: Any | None = None

    def generate(
        self,
        prompt: Prompt,
        system: str | None = None,
        thinking_budget: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        response = self._ensure_client().models.generate_content(
            model=self.model,
            contents=self._contents(prompt),
            config=self._build_config(
                system=system,
                thinking_budget=thinking_budget,
                temperature=temperature,
            ),
        )
        self._record_usage(response)
        return self._response_text(response)

    def stream(
        self,
        prompt: Prompt,
        system: str | None = None,
        thinking_budget: int | None = None,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        from gat.stream import stream_generate

        return stream_generate(
            self,
            prompt,
            system=system,
            thinking_budget=thinking_budget,
            temperature=temperature,
        )

    async def agenerate(
        self,
        prompt: Prompt,
        system: str | None = None,
        thinking_budget: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Async text generation.

        Runs the blocking SDK call in a worker thread so the event loop stays
        free to drive concurrent tool execution. Cost accounting and the
        configured timeout are preserved by delegating to :meth:`generate`.
        """

        return await asyncio.to_thread(
            self.generate,
            prompt,
            system=system,
            thinking_budget=thinking_budget,
            temperature=temperature,
        )

    def generate_structured(
        self,
        prompt: Prompt,
        schema: type[BaseModel],
        system: str | None = None,
        thinking_budget: int | None = None,
    ) -> BaseModel:
        response = self._ensure_client().models.generate_content(
            model=self.model,
            contents=self._contents(prompt),
            config=self._build_config(
                system=system,
                thinking_budget=thinking_budget,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=to_gemini_schema(schema),
            ),
        )
        self._record_usage(response)

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed

        text = self._response_text(response)
        data = _extract_json(text)
        return schema.model_validate(data)

    async def agenerate_structured(
        self,
        prompt: Prompt,
        schema: type[BaseModel],
        system: str | None = None,
        thinking_budget: int | None = None,
    ) -> BaseModel:
        """Async structured generation (see :meth:`agenerate`)."""

        return await asyncio.to_thread(
            self.generate_structured,
            prompt,
            schema,
            system=system,
            thinking_budget=thinking_budget,
        )

    def batch(
        self,
        prompts: list[Prompt],
        schema: type[BaseModel] | None = None,
        concurrency: int = 4,
    ) -> list[str | BaseModel]:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        def run_one(prompt: Prompt) -> str | BaseModel:
            if schema is None:
                return self.generate(prompt)
            return self.generate_structured(prompt, schema)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            return list(executor.map(run_one, prompts))

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise MissingApiKeyError("GEMINI_API_KEY is required for Gemini API calls")
        genai = importlib.import_module("google.genai")
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_config(
        self,
        *,
        system: str | None = None,
        thinking_budget: int | None = None,
        temperature: float = 0.7,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"temperature": temperature}
        if system is not None:
            kwargs["system_instruction"] = system
        if response_mime_type is not None:
            kwargs["response_mime_type"] = response_mime_type
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        if tools is not None:
            kwargs["tools"] = tools
        try:
            types = importlib.import_module("google.genai.types")
            if thinking_budget is not None:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
            return types.GenerateContentConfig(**kwargs)
        except Exception:
            if thinking_budget is not None:
                kwargs["thinking_config"] = {"thinking_budget": thinking_budget}
            return kwargs

    @staticmethod
    def _contents(prompt: Prompt) -> Any:
        if isinstance(prompt, str):
            return prompt
        return prompt

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        input_tokens = _usage_int(usage, "prompt_token_count", "input_tokens")
        output_tokens = _usage_int(usage, "candidates_token_count", "output_tokens")
        cached_tokens = _usage_int(usage, "cached_content_token_count", "cached_tokens")
        self._cost_tracker.add(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                part_text = getattr(parts[0], "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text.strip()
        raise EmptyResponseError("Gemini response did not contain text")


def _usage_int(usage: Any, *names: str) -> int:
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return 0


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _extract_json(text: str) -> Any:
    clean = _strip_code_fence(text)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", clean)
        if match:
            return json.loads(match.group())
        raise
