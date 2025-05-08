import os
import re
from functools import lru_cache
from typing import Any, Optional, Annotated
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms.base import OpenAI
from langchain_core.messages import FunctionMessage, AIMessage
from langchain.tools import StructuredTool

from openai import AuthenticationError as OpenAIAuthenticationError, APIError, APIConnectionError, RateLimitError, OpenAIError, BadRequestError
from .base import BaseArchytasModel, ModelConfig, set_env_auth
from ..exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

import logging
logger = logging.getLogger(__name__)

DEFERRED_TOKEN_VALUE = "***deferred***"

class OpenAIModel(BaseArchytasModel):
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        else:
            auth_token = self.config.api_key
        if not auth_token:
            auth_token = DEFERRED_TOKEN_VALUE
        set_env_auth(OPENAI_API_KEY=auth_token)

        # Replace local auth token from value from environment variables to allow fetching preset auth variables in the
        # environment.
        auth_token = os.environ.get('OPENAI_API_KEY', DEFERRED_TOKEN_VALUE)

        if auth_token != DEFERRED_TOKEN_VALUE:
            self.config.api_key = auth_token
        # Reset the openai client with the new value, if needed.
        if getattr(self, "model", None):
            self.model.openai_api_key._secret_value = auth_token
            self.model.client = None
            self.model.async_client = None

            # This method reinitializes the clients
            self.model.validate_environment()

    def initialize_model(self, **kwargs):
        try:
            model = self.config.model_name or self.DEFAULT_MODEL
            titoken_model_name =  "gpt-4o" if 'gpt-4.1' in model else model
            return ChatOpenAI(model=model, tiktoken_model_name=titoken_model_name)
        except (APIConnectionError, OpenAIError) as err:
            if not self.config.api_key:
                raise AuthenticationError("OpenAI API Key not set")
            else:
                raise AuthenticationError("OpenAI Authentication Error") from err

    def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        if self.model is None or not getattr(self.model, 'openai_api_key', None):
            raise AuthenticationError("OpenAI API Key missing")
        if "o3" in self.model.model_name.lower():
            # o3 doesn't accept a temperature keyword on invoke
            kwargs.pop("temperature")
        return super().ainvoke(input, config=config, stop=stop, **kwargs)

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, OpenAIAuthenticationError):
            raise AuthenticationError("OpenAI Authentication Error") from error
        elif isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
        elif isinstance(error, (APIConnectionError, OpenAIError)) and not self.model.openai_api_key:
            raise AuthenticationError("OpenAI Authentication Error") from error
        elif isinstance(error, (BadRequestError)) and error.code == "context_length_exceeded":
            raise ContextWindowExceededError(error.body.get('message', None)) from error
        else:
            raise error

    @lru_cache()
    def contextsize(self, model_name = None):
        if model_name is None:
            model_name = self.model_name
        try:
            return OpenAI.modelname_to_contextsize(model_name)
        except ValueError as err:
            if 'gpt-4.1' in model_name:
                return 1_000_000
            elif model_name.startswith(('o3', 'o4')):
                return 200_000
            raise
