import json
import re
import logging

from anthropic import AuthenticationError as AnthropicAuthenticError, RateLimitError
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel as PydanticModel, Field

from archytas.agent import AIMessage, BaseMessage

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig
from ..exceptions import AuthenticationError, ExecutionError

class DummyTool(PydanticModel):
    """
    Dummy Tool
    """
    input: str = Field(..., description="input")


class AnthropicModel(BaseArchytasModel):
    DEFAULT_MODEL: str = "claude-3-5-sonnet-latest"
    api_key: str = ""
    tool_name_map: dict
    rev_tool_name_map: dict

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.last_messages: list[BaseMessage] = None
        self.tool_name_map = {}
        self.rev_tool_name_map = {}

    def auth(self, **kwargs) -> None:
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        elif 'api_key' in self.config:
            self.api_key = self.config['api_key']
        if not self.api_key:
            raise AuthenticationError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        return ChatAnthropic(model=self.config.get("model_name", self.DEFAULT_MODEL), api_key=self.api_key)

    def _preprocess_messages(self, messages):
        from ..agent import AutoContextMessage, ContextMessage
        output = []

        system_messages = []
        # Combine all system/context/autocontext messages into a single initial system message
        for message in messages:
            match message:
                case SystemMessage() | ContextMessage() | AutoContextMessage():
                    system_messages.append(message.content)
                case _:
                    output.append(message)
        # Condense all context/system messages into a single first message as required by Anthropic
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        self.last_messages = [msg.model_copy(deep=True) for msg in output]
        return output

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, AnthropicAuthenticError):
            raise AuthenticationError("Anthropic Authentication Error") from error
        # TODO: Retry with delay on rate limit errors?
        # elif isinstance(error, RateLimitError):
        #     raise
        else:
            if self.last_messages:
                message_output = [msg.model_dump() for msg in self.last_messages]
                logging.warning(
                    "An exception has occurred. Below are the messages that were sent to in the most recent request:\n" +
                    json.dumps(message_output, indent=2)
                )
            raise
