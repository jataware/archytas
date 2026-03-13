from typing import Any

import groq
from langchain_groq import ChatGroq

from ..base_provider import BaseProvider
from ...exceptions import AuthenticationError


class GroqProvider(BaseProvider):
    """Provider for the Groq API."""

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        self._groq_client: groq.Groq | None = None
        super().__init__(api_key=api_key, **kwargs)

    def auth(self, **kwargs: Any) -> None:
        if not self.api_key:
            import os
            self.api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise AuthenticationError("No auth credentials found.")

    def create_chat_model(self, model_name: str, **kwargs: Any) -> ChatGroq:
        model = ChatGroq(
            model=model_name,
            api_key=self.api_key,
            base_url="https://api.groq.com/",
        )
        # Store reference to underlying Groq client for context size lookup
        from typing import cast
        import groq.resources
        self._groq_client = cast(groq.resources.chat.Completions, model.client)._client
        return model

    @property
    def groq_client(self) -> groq.Groq | None:
        return self._groq_client

    def context_size(self, model_name: str) -> int | None:
        if self._groq_client is None:
            return None
        try:
            model_list = self._groq_client.models.list()
            model_index = {model.id: model for model in model_list.data}
            model_info = model_index.get(model_name, None)
            return getattr(model_info, "context_window", None)
        except Exception:
            return None
