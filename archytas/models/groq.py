from typing import cast, ClassVar

import groq
import groq.resources
from langchain_groq import ChatGroq
from pydantic import BaseModel as PydanticModel, Field

from .base import BaseArchytasModel, ModelConfig
from .families import LegacyGroqModelFamily
from .provider import ModelProvider


class Structure(PydanticModel):
    thought: str = Field(..., description="Thought process of why you are calling the tool")
    tool: str
    tool_input: str | dict | list
    helpful_thoguht: bool


class GroqProvider(ModelProvider):
    def auth(self, model, **kwargs) -> None:
        model.api_key = kwargs['api_key'] if 'api_key' in kwargs else model.config.api_key
        if not model.api_key:
            raise ValueError("No auth credentials found.")

    def initialize_model(self, model, **kwargs):
        client = ChatGroq(
            model=model.config.model_name or model.DEFAULT_MODEL,
            api_key=model.api_key,
            base_url="https://api.groq.com/"
        )
        model._client = cast(groq.resources.chat.Completions, client.client)._client
        return client

    def contextsize(self, model, model_name: str | None = None) -> int | None:
        client = getattr(model, '_client', None)
        if client is None:
            client = cast(groq.resources.chat.Completions, model.model.client)._client
        model_list = client.models.list()
        model_index = {item.id: item for item in model_list.data}
        return getattr(model_index.get(model_name or model.model_name), "context_window", None)


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
