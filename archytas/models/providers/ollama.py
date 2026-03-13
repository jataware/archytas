from typing import Any

from langchain_ollama import ChatOllama

from ..base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider for local Ollama models."""

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, **kwargs)

    def auth(self, **kwargs: Any) -> None:
        # No auth needed for local Ollama
        pass

    def create_chat_model(self, model_name: str, **kwargs: Any) -> ChatOllama:
        self._chat_ollama = ChatOllama(model=model_name)
        return self._chat_ollama

    def context_size(self, model_name: str) -> int | None:
        if not hasattr(self, "_chat_ollama"):
            return None
        try:
            show_response = self._chat_ollama._client.show(model_name)
            model_info = show_response.modelinfo
            model_arch = model_info["general.architecture"]
            context_length = model_info[f"{model_arch}.context_length"]
            return int(context_length)
        except (KeyError, Exception):
            return None

    def transform_invoke_kwargs(self, kwargs: dict) -> dict:
        """Ollama passes temperature via the 'options' dict rather than as a top-level kwarg."""
        if "temperature" in kwargs:
            temp = kwargs.pop("temperature")
            options = kwargs.get("options", {})
            options["temperature"] = temp
            kwargs["options"] = options
        return kwargs
