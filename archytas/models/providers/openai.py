import os
from typing import Any

from langchain_openai.chat_models import ChatOpenAI
from openai import (
    AuthenticationError as OpenAIAuthenticationError,
    APIConnectionError,
    OpenAIError,
    RateLimitError,
    BadRequestError,
)

from ..base_provider import BaseProvider
from ...exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

DEFERRED_TOKEN_VALUE = "***deferred***"


class OpenAIProvider(BaseProvider):
    """Provider for the OpenAI API."""

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, **kwargs)

    def auth(self, **kwargs: Any) -> None:
        auth_token = self.api_key
        if not auth_token:
            auth_token = DEFERRED_TOKEN_VALUE
        os.environ.setdefault("OPENAI_API_KEY", auth_token)
        auth_token = os.environ.get("OPENAI_API_KEY", DEFERRED_TOKEN_VALUE)
        if auth_token != DEFERRED_TOKEN_VALUE:
            self.api_key = auth_token

    def create_chat_model(self, model_name: str, **kwargs: Any) -> ChatOpenAI:
        try:
            tiktoken_model_name = "gpt-4o" if "gpt-4.1" in model_name else model_name
            return ChatOpenAI(model=model_name, tiktoken_model_name=tiktoken_model_name)
        except (APIConnectionError, OpenAIError):
            if not self.api_key:
                raise AuthenticationError("OpenAI API Key not set")
            else:
                raise AuthenticationError("OpenAI Authentication Error")

    def handle_api_error(self, error: Exception) -> None:
        if isinstance(error, OpenAIAuthenticationError):
            raise AuthenticationError("OpenAI Authentication Error") from error
        elif isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
        elif isinstance(error, (APIConnectionError, OpenAIError)) and not self.api_key:
            raise AuthenticationError("OpenAI Authentication Error") from error
        elif isinstance(error, BadRequestError) and error.code == "context_length_exceeded":
            raise ContextWindowExceededError(error.body.get("message", None)) from error
        else:
            raise error
