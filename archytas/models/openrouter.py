"""Compatibility wrapper for models served by OpenRouter."""
from .openai import OpenAIModel
from .providers.openrouter import (
    ATTRIBUTION_HEADERS, DEFAULT_CONTEXT_SIZE, OPENROUTER_BASE_URL,
    OpenRouterProvider, _openrouter_context_lengths,
)


class OpenRouterModel(OpenAIModel):
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    DEFAULT_PROVIDER = OpenRouterProvider
