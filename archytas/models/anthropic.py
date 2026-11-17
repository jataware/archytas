"""Compatibility wrapper for models served by Anthropic."""
from .base import BaseArchytasModel, ModelConfig
from .families.anthropic import AnthropicModelFamily
from .providers.anthropic import (
    AnthropicProvider, DummyTool, _attach_cache_control, _prompt_cache_disabled,
)


class AnthropicModel(BaseArchytasModel):
    DEFAULT_MODEL: str = "claude-3-5-sonnet-latest"
    DEFAULT_PROVIDER = AnthropicProvider
    DEFAULT_FAMILY = AnthropicModelFamily
    api_key: str = ""
    tool_name_map: dict
    rev_tool_name_map: dict

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.tool_name_map = {}
        self.rev_tool_name_map = {}
        self._temperature_unsupported = False
