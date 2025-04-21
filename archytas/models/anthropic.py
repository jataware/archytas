import json
import re
import logging
from functools import lru_cache

from anthropic import AuthenticationError as AnthropicAuthenticError, RateLimitError, BadRequestError
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic.llms import AnthropicLLM
from pydantic import BaseModel as PydanticModel, Field

from archytas.agent import AIMessage, BaseMessage

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig
from ..exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

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
        self.tool_name_map = {}
        self.rev_tool_name_map = {}

    def auth(self, **kwargs) -> None:
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        else:
            self.api_key = self.config.api_key or ""
        if not self.api_key:
            raise AuthenticationError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        max_tokens = 4096
        if self.config.model_extra:
            max_tokens = self.config.model_extra.get('max_tokens', 4096)
        return ChatAnthropic(
            model=self.config.model_name or self.DEFAULT_MODEL,
            api_key=self.api_key,
            max_tokens=max_tokens
        )

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
        return output

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, AnthropicAuthenticError):
            raise AuthenticationError("Anthropic Authentication Error") from error
        # TODO: Retry with delay on rate limit errors?
        # elif isinstance(error, RateLimitError):
        #     raise
        elif isinstance(error, BadRequestError) and "prompt is too long" in error.message:
            sent = None
            maximum = None
            try:
                body = error.response.json().get("error", {})
                if "message" in body:
                    match: re.Match = re.search(r'prompt is too long: (\d+) .* (\d+) maximum', body["message"])
                    if match:
                        sent, maximum = match.groups()
            finally:
                raise ContextWindowExceededError(*error.args, sent=sent, maximum=maximum) from error
        else:
            # if self.last_messages:
            #     message_output = [msg.model_dump() for msg in self.last_messages]
            #     logging.warning(
            #         "An exception has occurred. Below are the messages that were sent to in the most recent request:\n" +
            #         json.dumps(message_output, indent=2)
            #     )
            raise

    @lru_cache()
    def contextsize(self, model_name = None):
        if model_name is None:
            model_name = self.model_name
        # TODO: This is accurate for all current models (as of 2025-02-27), for all Haiku and Sonnet models. Seems like
        # new models would have a new name if the context window changes. Hopefully in the future, this value can be
        # retrieved programatically.
        if model_name and ("haiku" in model_name or "sonnet" in model_name):
            return 200_000
        else:
            return None
