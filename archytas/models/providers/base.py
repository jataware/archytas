"""Transport adapters used by composed Archytas models."""
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.runnables import RunnableConfig

    from ...agent import AgentResponse
    from ..base import BaseArchytasModel
    from ..families import ModelFamily


class ModelProvider:
    """Own API/authentication behavior, independent of the model family.

    Providers receive the facade explicitly. Invocation state remains on that
    facade, so provider instances can be selected without sharing client state.
    """

    MODEL_SPECIFIC_CONTENT_TYPES: frozenset[str] = frozenset()

    def parse_prefix(self, model_name: str) -> tuple[type["ModelFamily"] | None, str]:
        """Resolve a family and the model identifier to send to the provider."""
        from ..families import resolve_model_family

        name, family = resolve_model_family(model_name, None)
        return (type(family) if family is not None else None), name

    def auth(self, model: "BaseArchytasModel", **kwargs: Any) -> None:
        pass

    def initialize_model(self, model: "BaseArchytasModel", **kwargs: Any) -> "BaseChatModel":
        raise NotImplementedError

    def reasoning_config(
        self, model: "BaseArchytasModel", model_name: str, profile: dict | None = None,
    ) -> dict | None:
        return None

    def prepare_invoke_kwargs(self, model: "BaseArchytasModel", kwargs: dict) -> dict:
        return kwargs

    def preprocess_messages(
        self, model: "BaseArchytasModel", messages: "list[BaseMessage]",
    ) -> "list[BaseMessage]":
        return messages

    def normalize_result(self, model: "BaseArchytasModel", response: "AIMessage") -> "AIMessage":
        return response

    def process_result(self, model: "BaseArchytasModel", response: "AIMessage") -> "AgentResponse":
        return model._default_process_result(response)

    def contextsize(self, model: "BaseArchytasModel", model_name: str | None = None) -> int | None:
        return None

    def fallback_contextsize(self, model: "BaseArchytasModel", model_name: str | None = None) -> int | None:
        return None

    async def ainvoke(
        self,
        model: "BaseArchytasModel",
        invoke: Callable[..., Awaitable["AIMessage"]],
        input: "list[BaseMessage]",
        *,
        config: "RunnableConfig | None" = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> "AIMessage":
        return await invoke(input, config=config, stop=stop, **kwargs)

    def handle_invoke_error(self, model: "BaseArchytasModel", error: BaseException) -> Any:
        raise error
