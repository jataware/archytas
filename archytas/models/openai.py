"""Compatibility model wrapper and legacy OpenAI reasoning settings."""
import os
from typing import Optional

from .base import BaseArchytasModel
from .families.openai import OpenAIModelFamily
from .providers.openai import DEFERRED_TOKEN_VALUE, OpenAIProvider

# Reasoning effort levels supported by each model family, keyed by model prefix pattern.
# Order within each list reflects what the model accepts.
REASONING_EFFORT_LEVELS: dict[str, list[str]] = {
    "o3":      ["low", "medium", "high"],
    "o4":      ["low", "medium", "high"],
    "gpt-5.1": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.2": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.3": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.4": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5":   ["minimal", "low", "medium", "high"],
}
PREFERRED_EFFORT_ORDER: list[str] = ["medium", "high", "low", "xhigh", "none"]
REASONING_EFFORT_REQUEST: Optional[str] = os.environ.get("LLM_REASONING_EFFORT", None)
REASONING_SUMMARY_REQUEST: Optional[str] = os.environ.get("LLM_REASONING_SUMMARY_TYPE", "concise")


class OpenAIModel(BaseArchytasModel):
    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_PROVIDER = OpenAIProvider
    DEFAULT_FAMILY = OpenAIModelFamily

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
