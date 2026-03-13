from typing import Any

from langchain_anthropic.chat_models import ChatAnthropic
from anthropic import AuthenticationError as AnthropicAuthenticError, BadRequestError

from ..base_provider import BaseProvider
from ...exceptions import AuthenticationError, ContextWindowExceededError

import re


class AnthropicProvider(BaseProvider):
    """Provider for the Anthropic API."""

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, **kwargs)

    def auth(self, **kwargs: Any) -> None:
        if not self.api_key:
            import os
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise AuthenticationError("No auth credentials found.")

    def create_chat_model(self, model_name: str, **kwargs: Any) -> ChatAnthropic:
        max_tokens = kwargs.get("max_tokens", None)
        return ChatAnthropic(
            model=model_name,
            api_key=self.api_key,
            max_tokens=max_tokens,
        )

    def handle_api_error(self, error: Exception) -> None:
        if isinstance(error, AnthropicAuthenticError):
            raise AuthenticationError("Anthropic Authentication Error") from error
        elif isinstance(error, BadRequestError) and "prompt is too long" in error.message:
            sent = None
            maximum = None
            try:
                body = error.response.json().get("error", {})
                if "message" in body:
                    match = re.search(
                        r"prompt is too long: (\d+) .* (\d+) maximum",
                        body["message"],
                    )
                    if match:
                        sent, maximum = match.groups()
            finally:
                raise ContextWindowExceededError(
                    *error.args, sent=sent, maximum=maximum
                ) from error
        else:
            raise error
