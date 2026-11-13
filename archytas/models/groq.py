"""Compatibility wrapper for models served by Groq."""
from typing import ClassVar

import groq
from langchain_groq import ChatGroq

from .base import BaseArchytasModel, ModelConfig
from .families.llama import LegacyGroqModelFamily
from .providers.groq import GroqProvider, Structure


class GroqModel(BaseArchytasModel):
    model: ChatGroq
    _client: groq.Groq
    api_key: str = ""
    DEFAULT_MODEL: ClassVar[str] = "llama3-8b-8192"
    DEFAULT_PROVIDER = GroqProvider
    DEFAULT_FAMILY = LegacyGroqModelFamily

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

    @property
    def model_name(self) -> str | None:
        model_name = getattr(self.model, "model_name", None)
        if model_name is not None:
            return model_name
        return getattr(self.config, "model_name", self.DEFAULT_MODEL)
