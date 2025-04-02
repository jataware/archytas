import json
import re
import groq.resources
import requests
from functools import lru_cache
from typing import Optional, cast, ClassVar

import groq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel as PydanticModel, Field

from .base import BaseArchytasModel, set_env_auth, ModelConfig

class Structure(PydanticModel):
    thought: str = Field(..., description="Thought process of why you are calling the tool")
    tool: str
    tool_input: str | dict | list
    helpful_thoguht: bool

class GroqModel(BaseArchytasModel):
    model: ChatGroq
    _client: groq.Groq
    api_key: str = ""

    DEFAULT_MODEL: ClassVar[str] = "llama3-8b-8192"
    MODEL_PROMPT_INSTRUCTIONS: str = """\
When generating JSON, remember to not wrap strings in triple quotes such as \'\'\' or \"\"\". If you want to add newlines \
to the JSON text, use `\\n` to add newlines.
Ensure all generated JSON is valid and would pass a JSON validator.
"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = cast(groq.resources.chat.Completions, self.model.client)._client

    def auth(self, **kwargs) -> None:
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        else:
            self.api_key = self.config.api_key
        if not self.api_key:
            raise ValueError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        model = ChatGroq(
            model=self.config.model_name or self.DEFAULT_MODEL,
            api_key=self.api_key,
            base_url="https://api.groq.com/"
        )
        return model

    @property
    def model_name(self) -> str | None:
        model_name = getattr(self.model, "model_name", None)
        if model_name is not None:
            return model_name
        else:
            return getattr(self.config, "model_name", self.DEFAULT_MODEL)

    def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        return super().ainvoke(input, config=config, stop=stop, **kwargs)

    def _preprocess_messages(self, messages: list[BaseMessage]):
        from ..agent import AutoContextMessage, ContextMessage
        output = []
        system_messages = []
        for message in messages:
            match message:
                case SystemMessage() | ContextMessage() | AutoContextMessage():
                    system_messages.append(message.content)
                case AIMessage():
                    # Duplicate mesage so we don't change raw storage
                    msg = message.model_copy()
                    if msg.tool_calls:
                        msg.tool_calls.clear()
                    output.append(msg)
                case _:
                    output.append(message)
        # Condense all context/system messages into a single first message as required by Anthropic
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output

    @lru_cache()
    def contextsize(self, model_name: Optional[str]=None) -> int | None:
        if model_name is None:
            model_name = self.model_name
        model_list = self._client.models.list()
        model_index = {model.id: model for model in model_list.data}
        model_info = model_index.get(model_name, None)
        context_window = getattr(model_info, "context_window", None)
        return context_window
