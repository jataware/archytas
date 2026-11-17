"""Compatibility wrapper for models served by Ollama."""
from .base import BaseArchytasModel
from .families.llama import LlamaModelFamily
from .providers.ollama import OllamaProvider


class OllamaModel(BaseArchytasModel):
    DEFAULT_MODEL = "llama3"
    DEFAULT_PROVIDER = OllamaProvider
    DEFAULT_FAMILY = LlamaModelFamily
