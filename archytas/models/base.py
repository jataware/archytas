from abc import ABC, abstractmethod
from typing import Any, ClassVar
from pydantic import BaseModel as PydanticModel


from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)

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


class ModelConfig(PydanticModel):
    pass


class BaseArchytasModel(ABC):
    langchain_cls: ClassVar[type]
    auth_cls: ClassVar[type]
    model_name: str

    model: BaseLanguageModel
    config: ModelConfig

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model = self.initialize_model()

    @abstractmethod
    def auth(self, **kwargs) -> None:
        ...

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

    def _preprocess_messages(self, messages):
        return messages
