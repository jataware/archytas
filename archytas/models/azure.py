"""Compatibility wrapper for Azure OpenAI models."""
from .base import ModelConfig
from .openai import OpenAIModel
from .providers.azure import DEFERRED_TOKEN_VALUE, AzureOpenAIProvider


class AzureOpenAIModel(OpenAIModel):
    DEFAULT_PROVIDER = AzureOpenAIProvider

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config, **kwargs)


class AzureFoundryModel():
    pass
