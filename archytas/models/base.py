import json
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, ClassVar
from pydantic import BaseModel as PydanticModel
from contextlib import contextmanager
from copy import copy

from langchain_core.messages import ToolMessage, AIMessage, ToolCall, ChatMessage, HumanMessage, FunctionMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel


class EnvironmentAuth:
    env_settings: dict[str, str]

    def __init__(self, **env_settings: dict[str, str]) -> None:
        for key, value in env_settings.items():
            if not (isinstance(key, str) and isinstance(value, str)):
                raise ValueError("EnvironmentAuth variables names and values must be strings.")
        self.env_settings = env_settings

    def apply(self):
        import os
        os.environ.update(self.env_settings)

def set_env_auth(**env_settings: dict[str, str]) -> None:
    import os
    for key, value in env_settings.items():
        if not (isinstance(key, str) and isinstance(value, str)):
            raise ValueError("EnvironmentAuth variables names and values must be strings.")
    os.environ.update(env_settings)


class ModelConfig(PydanticModel):
    pass


class BaseArchytasModel(ABC):

    MODEL_PROMPT_INSTRUCTIONS: str = """"""

    model: BaseChatModel
    config: ModelConfig

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        self.config = config
        self.auth(**kwargs)
        self.model = self.initialize_model(**kwargs)

    def auth(self, **kwargs) -> None:
        pass

    @property
    def additional_prompt_info(self) -> str | None:
        return None

    @abstractmethod
    def initialize_model(self, **kwargs):
        ...

    def invoke(self, input, *, config=None, stop=None, **kwargs):
        return self.model.invoke(
            self._preprocess_messages(input),
            config,
            stop=stop,
            **kwargs
        )

    async def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        try:
            return await self.model.ainvoke(
                self._preprocess_messages(input),
                config,
                stop=stop,
                **kwargs
            )
        except Exception as e:
            raise

    def _preprocess_messages(self, messages: list[BaseMessage]):
        return messages

    def process_result(self, response_message: AIMessage):
        content = response_message.content
        if isinstance(content, list):
            return "\n".join(item['text'] for item in content if item.get('type', None) == "text")
        return content

    def handle_invoke_error(error: BaseException):
        pass
