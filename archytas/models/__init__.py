"""Archytas model system.

Public API for creating and configuring LLM models.
"""

from .model import Model
from .base_provider import BaseProvider
from .base_family import BaseModelFamily, ModelVersion
from .config import ModelConfig

# Convenience constructors
from .shortcuts import (
    GPT4o,
    GPT41,
    GPT5,
    Sonnet,
    Opus,
    Haiku,
    GeminiFlash,
    GeminiPro,
)

# Provider classes
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    BedrockProvider,
    AzureOpenAIProvider,
    GroqProvider,
    OllamaProvider,
    OpenRouterProvider,
)

# Family classes
from .families import (
    GPTFamily,
    ClaudeFamily,
    GeminiFamily,
    LlamaFamily,
    GenericFamily,
)

# Backward-compat re-exports from old base.py
from .tool_convert import final_answer, fail_task, convert_tools

__all__ = [
    # Core
    "Model",
    "BaseProvider",
    "BaseModelFamily",
    "ModelVersion",
    "ModelConfig",
    # Shortcuts
    "GPT4o",
    "GPT41",
    "GPT5",
    "Sonnet",
    "Opus",
    "Haiku",
    "GeminiFlash",
    "GeminiPro",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "BedrockProvider",
    "AzureOpenAIProvider",
    "GroqProvider",
    "OllamaProvider",
    "OpenRouterProvider",
    # Families
    "GPTFamily",
    "ClaudeFamily",
    "GeminiFamily",
    "LlamaFamily",
    "GenericFamily",
    # Tools
    "final_answer",
    "fail_task",
    "convert_tools",
]
