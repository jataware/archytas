"""Convenience constructors for common model configurations."""

from __future__ import annotations

from typing import Any

from .model import Model
from .base_provider import BaseProvider
from .base_family import BaseModelFamily


# --- OpenAI GPT models ---

def GPT4o(*, api_key: str | None = None, provider: BaseProvider | type[BaseProvider] | None = None, **kwargs: Any) -> Model:
    from .families.gpt import GPTFamily
    return Model(
        provider=provider,
        family=GPTFamily,
        model_name="gpt-4o",
        api_key=api_key,
        **kwargs,
    )


def GPT41(*, api_key: str | None = None, provider: BaseProvider | type[BaseProvider] | None = None, **kwargs: Any) -> Model:
    from .families.gpt import GPTFamily
    return Model(
        provider=provider,
        family=GPTFamily,
        model_name="gpt-4.1-2025-04-14",
        api_key=api_key,
        **kwargs,
    )


def GPT5(*, api_key: str | None = None, provider: BaseProvider | type[BaseProvider] | None = None, **kwargs: Any) -> Model:
    from .families.gpt import GPTFamily
    return Model(
        provider=provider,
        family=GPTFamily,
        model_name="gpt-5",
        api_key=api_key,
        **kwargs,
    )


# --- Anthropic Claude models ---

def Sonnet(*, api_key: str | None = None, provider: BaseProvider | type[BaseProvider] | None = None, **kwargs: Any) -> Model:
    from .families.claude import ClaudeFamily
    return Model(
        provider=provider,
        family=ClaudeFamily,
        model_name="claude-sonnet-4-6-20250514",
        api_key=api_key,
        **kwargs,
    )


def Opus(*, api_key: str | None = None, provider: BaseProvider | type[BaseProvider] | None = None, **kwargs: Any) -> Model:
    from .families.claude import ClaudeFamily
    return Model(
        provider=provider,
        family=ClaudeFamily,
        model_name="claude-opus-4-6",
        api_key=api_key,
        **kwargs,
    )


def Haiku(*, api_key: str | None = None, provider: BaseProvider | type[BaseProvider] | None = None, **kwargs: Any) -> Model:
    from .families.claude import ClaudeFamily
    return Model(
        provider=provider,
        family=ClaudeFamily,
        model_name="claude-3-5-haiku-latest",
        api_key=api_key,
        **kwargs,
    )


# --- Google Gemini models ---

def GeminiFlash(*, api_key: str | None = None, provider: BaseProvider | type[BaseProvider] | None = None, **kwargs: Any) -> Model:
    from .families.gemini import GeminiFamily
    return Model(
        provider=provider,
        family=GeminiFamily,
        model_name="gemini-2.5-flash",
        api_key=api_key,
        **kwargs,
    )


def GeminiPro(*, api_key: str | None = None, provider: BaseProvider | type[BaseProvider] | None = None, **kwargs: Any) -> Model:
    from .families.gemini import GeminiFamily
    return Model(
        provider=provider,
        family=GeminiFamily,
        model_name="gemini-2.5-pro",
        api_key=api_key,
        **kwargs,
    )
