from typing import Any

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI, ChatGoogleGenerativeAIError

from ..base_provider import BaseProvider
from ...exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError


class GeminiProvider(BaseProvider):
    """Provider for the Google Gemini API."""

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, **kwargs)

    def auth(self, **kwargs: Any) -> None:
        if not self.api_key:
            import os
            self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise AuthenticationError("Gemini API key not provided.")

    def create_chat_model(self, model_name: str, **kwargs: Any) -> ChatGoogleGenerativeAI:
        model_kwargs: dict[str, Any] = dict(
            model=model_name,
            api_key=self.api_key,
        )
        # Merge in any extra kwargs (e.g. include_thoughts from family)
        model_kwargs.update(kwargs)
        return ChatGoogleGenerativeAI(**model_kwargs)

    def handle_api_error(self, error: Exception) -> None:
        if isinstance(error, ChatGoogleGenerativeAIError):
            if any("400 API key not valid" in arg for arg in error.args):
                raise AuthenticationError("API key invalid.") from error
            elif any(
                "exceeds the maximum number of tokens allowed" in arg
                for arg in error.args
            ):
                raise ContextWindowExceededError(
                    "Context window maximum tokens exceeded."
                ) from error
        raise ExecutionError(*error.args) from error
