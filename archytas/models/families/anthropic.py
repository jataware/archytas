"""Anthropic model-family behavior."""
from typing import TYPE_CHECKING

from .base import ModelFamily, _combine_system_messages, register_model_family

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from ..base import BaseArchytasModel


@register_model_family
class AnthropicModelFamily(ModelFamily):
    slug = "anthropic"

    def preprocess_messages(
        self, model: "BaseArchytasModel", messages: "list[BaseMessage]",
    ) -> "list[BaseMessage]":
        return _combine_system_messages(messages)

    def contextsize(self, model: "BaseArchytasModel", model_name: str | None = None) -> int | None:
        name = model_name or model.model_name
        return 200_000 if name and ("haiku" in name or "sonnet" in name) else None
