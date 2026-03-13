import os
from typing import Any

from langchain_openai import AzureChatOpenAI
from openai import APIConnectionError, OpenAIError

from ..base_provider import BaseProvider
from ...exceptions import AuthenticationError

DEFERRED_TOKEN_VALUE = "***deferred***"


class AzureOpenAIProvider(BaseProvider):
    """Provider for Azure OpenAI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str = "2024-10-21",
        **kwargs: Any,
    ) -> None:
        self.endpoint = endpoint
        self.api_version = api_version
        super().__init__(api_key=api_key, **kwargs)

    def auth(self, **kwargs: Any) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        if not self.api_key:
            self.api_key = DEFERRED_TOKEN_VALUE
        os.environ.setdefault("AZURE_OPENAI_API_KEY", self.api_key)
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY", DEFERRED_TOKEN_VALUE)

        if not self.endpoint:
            self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", None)
        if not self.endpoint:
            raise AuthenticationError("Azure OpenAI models must have endpoint set.")

    def create_chat_model(self, model_name: str, **kwargs: Any) -> AzureChatOpenAI:
        try:
            return AzureChatOpenAI(
                model=model_name,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
            )
        except (APIConnectionError, OpenAIError):
            if not self.api_key or self.api_key == DEFERRED_TOKEN_VALUE:
                raise AuthenticationError("Azure OpenAI API Key not set")
            else:
                raise AuthenticationError("Azure OpenAI Authentication Error")

    def handle_api_error(self, error: Exception) -> None:
        # Reuse OpenAI error handling patterns
        from openai import (
            AuthenticationError as OpenAIAuthenticationError,
            RateLimitError,
            BadRequestError,
        )
        from ...exceptions import ExecutionError, ContextWindowExceededError

        if isinstance(error, OpenAIAuthenticationError):
            raise AuthenticationError("Azure OpenAI Authentication Error") from error
        elif isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
        elif isinstance(error, BadRequestError) and error.code == "context_length_exceeded":
            raise ContextWindowExceededError(error.body.get("message", None)) from error
        else:
            raise error
