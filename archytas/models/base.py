import os
from abc import ABC, abstractmethod
from pydantic import BaseModel as PydanticModel, ConfigDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.language_models.chat_models import BaseChatModel


class EnvironmentAuth:
    env_settings: dict[str, str]

    def __init__(self, **env_settings: dict[str, str]) -> None:
        for key, value in env_settings.items():
            if not (isinstance(key, str) and isinstance(value, str)):
                raise ValueError("EnvironmentAuth variables names and values must be strings.")
        self.env_settings = env_settings

    def apply(self):
        os.environ.update(self.env_settings)

def set_env_auth(**env_settings: dict[str, str]) -> None:
    for key, value in env_settings.items():
        if not (isinstance(key, str) and isinstance(value, str)):
            raise ValueError("EnvironmentAuth variables names and values must be strings.")
    os.environ.update(env_settings)


class ModelConfig(PydanticModel, extra='allow'):
    model_name: str
    api_key: str

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class BaseArchytasModel(ABC):

    MODEL_PROMPT_INSTRUCTIONS: str = ""

    model: "BaseChatModel"
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
            messages = self._preprocess_messages(input)
            return await self.model.ainvoke(
                messages,
                config,
                stop=stop,
                **kwargs
            )
        except Exception as error:
            print(error)
            return self.handle_invoke_error(error)

    def _preprocess_messages(self, messages: "list[BaseMessage]"):
        return messages

    def _rectify_result(self, response_message: "AIMessage"):
        return response_message

    def process_result(self, response_message: "AIMessage"):
        content = response_message.content
        if isinstance(content, list):
            return "\n".join(item['text'] for item in content if item.get('type', None) == "text")
        return content

    def handle_invoke_error(self, error: BaseException):
        raise error
