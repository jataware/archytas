from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class BaseProvider(ABC):
    """Owns authentication and client creation for a specific API service."""

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self._extra = kwargs
        self.auth(**kwargs)

    @abstractmethod
    def auth(self, **kwargs: Any) -> None:
        """Validate and store credentials."""
        ...

    @abstractmethod
    def create_chat_model(self, model_name: str, **kwargs: Any) -> "BaseChatModel":
        """Create the underlying LangChain chat model (or custom adapter)."""
        ...

    def context_size(self, model_name: str) -> int | None:
        """Provider-level context size lookup (e.g. via API query).

        Called by Model.contextsize() when both version and family return None.
        Default: return None.
        """
        return None

    def transform_invoke_kwargs(self, kwargs: dict) -> dict:
        """Transform kwargs before passing to the chat model's ainvoke.

        Override this to handle provider-specific kwarg conventions.
        Default: return kwargs unchanged.
        """
        return kwargs

    def handle_api_error(self, error: Exception) -> None:
        """Map provider API exceptions to archytas exceptions.

        Last in the error chain (after version and family).
        Default: re-raise the original error.
        """
        raise error
