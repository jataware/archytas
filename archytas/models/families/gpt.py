from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

from langchain_openai.llms.base import OpenAI

from ..base_family import BaseModelFamily, ModelVersion


class GPTFamily(BaseModelFamily):
    """Model family for OpenAI GPT models."""

    VERSIONS: ClassVar[dict[str, ModelVersion]] = {
        "latest": ModelVersion(model_name="gpt-4o"),
        "gpt-4.1": ModelVersion(model_name="gpt-4.1-2025-04-14", context_size=1_000_000),
        "gpt-4o": ModelVersion(model_name="gpt-4o"),
        "gpt-5": ModelVersion(model_name="gpt-5", context_size=400_000),
        "o3": ModelVersion(model_name="o3", context_size=200_000),
        "o4-mini": ModelVersion(model_name="o4-mini", context_size=200_000),
    }
    DEFAULT_VERSION: ClassVar[str] = "latest"
    PROMPT_INSTRUCTIONS: ClassVar[str] = ""

    @lru_cache()
    def context_size(self, model_name: str) -> int | None:
        try:
            return OpenAI.modelname_to_contextsize(model_name)
        except ValueError:
            if "gpt-4.1" in model_name:
                return 1_000_000
            elif model_name.startswith(("o3", "o4")):
                return 200_000
            elif "gpt-5" in model_name.lower():
                return 400_000
            raise

    def supports_temperature(self, model_name: str) -> bool:
        lower = model_name.lower()
        return "o3" not in lower and "gpt-5" not in lower
