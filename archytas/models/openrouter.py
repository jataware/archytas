"""OpenRouter provider, served over OpenRouter's OpenAI-compatible API.

OpenRouter (https://openrouter.ai) exposes many providers' models behind the
OpenAI chat-completions API (including tool calling), so this provider is a
thin subclass of `OpenAIModel` pointed at the OpenRouter base URL rather than
a bespoke client. Model ids are namespaced by upstream provider, e.g.
`openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`, `qwen/qwen3-coder:free`.
"""
import logging
import os
from functools import lru_cache
from typing import Optional

import httpx
from langchain_openai.chat_models import ChatOpenAI

from .openai import OpenAIModel
from ..exceptions import AuthenticationError

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Used when the model isn't found in the live model listing. Conservative on
# purpose: overestimating the window delays summarization until requests fail.
DEFAULT_CONTEXT_SIZE = 128_000

# Attribution headers, per OpenRouter's app-discoverability convention.
ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/jataware/archytas",
    "X-Title": "Archytas",
}


@lru_cache(maxsize=1)
def _openrouter_context_lengths() -> dict[str, int]:
    """Fetch context sizes for all models from OpenRouter's public model list.

    Cached for the process lifetime; returns an empty mapping on any failure so
    callers fall back to DEFAULT_CONTEXT_SIZE rather than erroring.
    """
    try:
        response = httpx.get(f"{OPENROUTER_BASE_URL}/models", timeout=10)
        response.raise_for_status()
        models = response.json().get("data", [])
        return {
            model["id"]: int(model["context_length"])
            for model in models
            if model.get("id") and model.get("context_length")
        }
    except Exception as err:
        logger.warning("Unable to fetch OpenRouter model list (%s); context sizes will use the default.", err)
        return {}


class OpenRouterModel(OpenAIModel):
    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def auth(self, **kwargs) -> None:
        # Unlike OpenAIModel.auth, no environment round-trip: the key is used
        # explicitly and must never be written to (or read back from)
        # OPENAI_API_KEY, which may belong to a different provider entry.
        api_key = (
            kwargs.get("api_key")
            or self.config.api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        if not api_key:
            raise AuthenticationError(
                "No OpenRouter API key found. Set one in the provider configuration or via OPENROUTER_API_KEY."
            )
        self.config.api_key = api_key

    def initialize_model(self, **kwargs):
        return ChatOpenAI(
            model=self.config.model_name or self.DEFAULT_MODEL,
            api_key=self.config.api_key,
            base_url=OPENROUTER_BASE_URL,
            # Namespaced ids like "anthropic/claude-3.5-sonnet" have no tiktoken
            # encoding; pin a stable one so token estimates are quiet and
            # consistent (they are budgeting approximations either way).
            tiktoken_model_name="gpt-4o",
            default_headers=ATTRIBUTION_HEADERS,
        )

    @lru_cache()
    def contextsize(self, model_name: Optional[str] = None) -> int | None:
        name = model_name or self.model_name
        context_size = _openrouter_context_lengths().get(name)
        if context_size:
            return context_size
        logger.warning(
            "OpenRouter context size unknown for model %r; using default of %d.",
            name, DEFAULT_CONTEXT_SIZE,
        )
        return DEFAULT_CONTEXT_SIZE
