import os
import re
from functools import lru_cache
from typing import Any, Optional, Annotated
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import FunctionMessage, AIMessage
from langchain.tools import StructuredTool

from openai import AuthenticationError as OpenAIAuthenticationError, APIError, APIConnectionError, RateLimitError, OpenAIError
from .base import BaseArchytasModel, ModelConfig, set_env_auth
from ..exceptions import AuthenticationError, ExecutionError

DEFERRED_TOKEN_VALUE = "***deferred***"

class OpenAIModel(BaseArchytasModel):
    tool_descriptions: dict[str, str]

    @property
    def MODEL_PROMPT_INSTRUCTIONS(self):

        tool_desc = ["The following tools are available:"]
        for tool_name, tool_description in self.tool_descriptions.items():
            tool_desc.append(
                f"""\
{tool_name}:
    {tool_description}
"""
            )
        return "\n------\n".join(tool_desc)

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.tool_descriptions = {}

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        elif 'api_key' in self.config:
            auth_token = self.config['api_key']
        if not auth_token:
            auth_token = DEFERRED_TOKEN_VALUE
        set_env_auth(OPENAI_API_KEY=auth_token)

        # Replace local auth token from value from environment variables to allow fetching preset auth variables in the
        # environment.
        auth_token = os.environ.get('OPENAI_API_KEY', DEFERRED_TOKEN_VALUE)

        if auth_token != DEFERRED_TOKEN_VALUE:
            self.config['api_key'] = auth_token
        # Reset the openai client with the new value, if needed.
        if getattr(self, "model", None):
            self.model.openai_api_key._secret_value = auth_token
            self.model.client = None
            self.model.async_client = None

            # This method reinitializes the clients
            self.model.validate_environment()

    def convert_tools(self, archytas_tools: tuple[tuple[str, Any], ...])-> "list[StructuredTool]":
        tools = super().convert_tools(archytas_tools)
        self.tool_descriptions = {}
        for tool in tools:
            if len(tool.description) > 1024:
                self.tool_descriptions[tool.name] = tool.description
                tool.description = f"The description for this tool, `{tool.name}`, can be found in the system message on this call."
        return tools

    def initialize_model(self, **kwargs):
        try:
            return ChatOpenAI(model=self.config.get("model_name", "gpt-4o"))
        except (APIConnectionError, OpenAIError) as err:
            if not self.config.get('api_key', None):
                raise AuthenticationError("OpenAI API Key not set")
            else:
                raise AuthenticationError("OpenAI Authentication Error") from err

    def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        if self.model is None or not getattr(self.model, 'openai_api_key', None):
            raise AuthenticationError("OpenAI API Key missing")
        return super().ainvoke(input, config=config, stop=stop, **kwargs)

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, OpenAIAuthenticationError):
            raise AuthenticationError("OpenAI Authentication Error") from error
        elif isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
        elif isinstance(error, (APIConnectionError, OpenAIError)) and not self.model.openai_api_key:
            raise AuthenticationError("OpenAI Authentication Error") from error
        else:
            raise error
