"""Compatibility model wrapper and legacy Gemini reasoning settings."""
import os
from typing import Optional

from .base import BaseArchytasModel
from .families.gemini import GeminiModelFamily
from .providers.gemini import GeminiProvider

# Reasoning effort levels supported by each model family, keyed by model prefix pattern.
# Order within each list reflects what the model accepts.
REASONING_EFFORT_LEVELS: dict[str, list[str]] = {
    "gemini-2.5":       ["budget"],
    "gemini-3-flash": ["minimal", "low", "medium", "high"],
    "gemini-3.1-pro": ["low", "medium", "high"],
    "gemini-3.1-flash-lite": ["minimal", "low", "medium", "high"],
}
PREFERRED_EFFORT_ORDER: list[str] = ["medium", "high", "low", "minimal", "budget"]
REASONING_EFFORT_REQUEST: Optional[str] = os.environ.get("LLM_REASONING_EFFORT", None)
REASONING_BUDGET_REQUEST: Optional[str] = os.environ.get("LLM_REASONING_BUDGET", None)
DEFAULT_REASONING_BUDGET: int = -1


class GeminiModel(BaseArchytasModel):
    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_PROVIDER = GeminiProvider
    DEFAULT_FAMILY = GeminiModelFamily
    api_key: str
