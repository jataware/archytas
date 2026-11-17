import os
from typing import cast

import groq
import groq.resources
from langchain_groq import ChatGroq
from pydantic import BaseModel as PydanticModel, Field

from .base import ModelProvider


class Structure(PydanticModel):
    thought: str = Field(..., description="Thought process of why you are calling the tool")
    tool: str
    tool_input: str | dict | list
    helpful_thoguht: bool


class GroqProvider(ModelProvider):
    def auth(self, model, **kwargs) -> None:
        model.api_key = (
            kwargs.get('api_key')
            or model.config.api_key
            or os.environ.get('GROQ_API_KEY', '')
        )
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

