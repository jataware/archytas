from __future__ import annotations

import copy
import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Sequence

from langchain_core.tools import StructuredTool

from .base_provider import BaseProvider
from .base_family import BaseModelFamily, ModelVersion
from .tool_convert import convert_tools

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from ..agent import AgentResponse

logger = logging.getLogger(__name__)

# Default provider for each family type (lazy-loaded to avoid circular imports)
_FAMILY_DEFAULT_PROVIDERS: dict[type, type] = {}


def _ensure_defaults_loaded() -> None:
    if _FAMILY_DEFAULT_PROVIDERS:
        return
    from .families.gpt import GPTFamily
    from .families.claude import ClaudeFamily
    from .families.gemini import GeminiFamily
    from .families.llama import LlamaFamily
    from .families.generic import GenericFamily
    from .providers.openai import OpenAIProvider
    from .providers.anthropic import AnthropicProvider
    from .providers.gemini import GeminiProvider
    from .providers.groq import GroqProvider
    from .providers.ollama import OllamaProvider

    _FAMILY_DEFAULT_PROVIDERS.update({
        GPTFamily: OpenAIProvider,
        ClaudeFamily: AnthropicProvider,
        GeminiFamily: GeminiProvider,
        LlamaFamily: GroqProvider,
        GenericFamily: OpenAIProvider,
    })


class Model:
    """User-facing composed model object.

    Binds a Provider + ModelFamily + ModelVersion together and presents
    a unified Archytas interface to Agent. Wraps LangChain entities
    internally but does not expose them directly.
    """

    DEFAULT_SUMMARIZATION_RATIO: float = 0.5

    def __init__(
        self,
        *,
        provider: BaseProvider | type[BaseProvider] | None = None,
        family: BaseModelFamily | type[BaseModelFamily] | None = None,
        version: str | ModelVersion | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        summarization_ratio: float | None = None,
        summarization_threshold: int | None = None,
        summarization_threshold_pct: int | None = None,
        **kwargs: Any,
    ) -> None:
        _ensure_defaults_loaded()

        # --- Resolve family ---
        if family is None:
            from .families.gpt import GPTFamily
            family = GPTFamily
        if isinstance(family, type):
            self._family = family()
        else:
            self._family = family

        # --- Resolve version ---
        if version is None:
            version_key = self._family.DEFAULT_VERSION
        elif isinstance(version, str):
            version_key = version
        else:
            version_key = None

        if version_key is not None:
            if version_key in self._family.VERSIONS:
                self._version = self._family.VERSIONS[version_key]
            else:
                # Treat the version string as a model name directly
                self._version = ModelVersion(model_name=version_key)
        else:
            self._version = version

        # --- Resolve model name ---
        if model_name is not None:
            self._model_name = model_name
        else:
            self._model_name = self._version.model_name

        # Set the model name on the family for preprocessing that needs it
        self._family._current_model_name = self._model_name

        # --- Resolve provider ---
        if provider is None:
            provider_cls = _FAMILY_DEFAULT_PROVIDERS.get(type(self._family))
            if provider_cls is None:
                from .providers.openai import OpenAIProvider
                provider_cls = OpenAIProvider
            provider = provider_cls

        if isinstance(provider, type):
            provider_kwargs = {}
            if api_key is not None:
                provider_kwargs["api_key"] = api_key
            provider_kwargs.update(kwargs)
            self._provider = provider(**provider_kwargs)
        else:
            self._provider = provider

        # --- Summarization config ---
        self._summarization_ratio = summarization_ratio
        self._summarization_threshold = summarization_threshold
        if summarization_threshold_pct is not None:
            self._summarization_ratio = float(summarization_threshold_pct) / 100

        # --- Create the underlying chat model ---
        create_kwargs: dict[str, Any] = {}
        resolved_max_tokens = max_tokens or self._version.max_tokens
        if resolved_max_tokens is not None:
            create_kwargs["max_tokens"] = resolved_max_tokens
        # Let family inject extra kwargs (e.g. Gemini's include_thoughts)
        if hasattr(self._family, "get_create_model_kwargs"):
            create_kwargs.update(self._family.get_create_model_kwargs(self._model_name))
        create_kwargs.update(self._version.extra)

        self._chat_model = self._provider.create_chat_model(self._model_name, **create_kwargs)
        self._lc_tools: list[StructuredTool] | None = None

    # --- Properties ---

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    @property
    def family(self) -> BaseModelFamily:
        return self._family

    @property
    def version(self) -> ModelVersion:
        return self._version

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def lc_tools(self) -> list[StructuredTool] | None:
        return self._lc_tools

    @property
    def MODEL_PROMPT_INSTRUCTIONS(self) -> str:
        return self._family.PROMPT_INSTRUCTIONS

    @property
    def additional_prompt_info(self) -> str | None:
        return None

    @property
    def _model(self) -> "BaseChatModel":
        """Internal access to the underlying chat model. Used by chat_history."""
        return self._chat_model

    @property
    def model(self) -> "BaseChatModel":
        """Returns the chat model with tools bound if any."""
        if self._lc_tools is not None:
            return self._chat_model.bind_tools(self._lc_tools)
        return self._chat_model

    # --- Summarization ---

    @property
    def default_summarization_threshold(self) -> int | None:
        context_max = self._family.context_size(self._model_name)
        if self._version.context_size is not None:
            context_max = self._version.context_size
        if context_max is None:
            return None
        return int(context_max * self.DEFAULT_SUMMARIZATION_RATIO)

    @property
    def summarization_threshold(self) -> int | None:
        context_max = self._family.context_size(self._model_name)
        if self._version.context_size is not None:
            context_max = self._version.context_size

        if self._summarization_threshold is not None:
            if context_max is None:
                return self._summarization_threshold
            return min(self._summarization_threshold, context_max)

        ratio = self._summarization_ratio or self.DEFAULT_SUMMARIZATION_RATIO
        if context_max is None:
            return None
        return int(context_max * ratio)

    # --- Tool management ---

    def set_tools(self, agent_tools: dict) -> list[StructuredTool]:
        agent_tools_tuple = tuple(
            sorted(
                [
                    (name, func)
                    for name, func in agent_tools.items()
                    if not getattr(func, "_disabled", False)
                ],
                key=lambda tool: tool[0],
            )
        )
        tools = convert_tools(agent_tools_tuple)
        self._lc_tools = tools
        return tools

    @staticmethod
    def convert_tools(archytas_tools: tuple[tuple[str, Any], ...]) -> list[StructuredTool]:
        return convert_tools(archytas_tools)

    # --- Invocation ---

    def _preprocess_messages(self, messages: list["BaseMessage"]) -> list["BaseMessage"]:
        return self._family.preprocess_messages(messages)

    def invoke(self, input: Any, *, config: Any = None, stop: Any = None, agent_tools: dict | None = None, **kwargs: Any):
        result = self.model.invoke(
            self._preprocess_messages(input),
            config,
            stop=stop,
            agent_tools=agent_tools,
            **kwargs,
        )
        return result

    async def ainvoke(self, input: Any, *, config: Any = None, stop: Any = None, agent_tools: dict | None = None, **kwargs: Any):
        if self._lc_tools is None and agent_tools is not None:
            self.set_tools(agent_tools)

        # Filter temperature if model doesn't support it
        if not self._family.supports_temperature(self._model_name):
            kwargs.pop("temperature", None)

        # Let provider transform kwargs (e.g. Ollama moves temperature to options)
        kwargs = self._provider.transform_invoke_kwargs(kwargs)

        try:
            messages = self._preprocess_messages(input)
            result = await self.model.ainvoke(
                messages,
                config,
                stop=stop,
                **kwargs,
            )
            return result
        except Exception as error:
            print(error)
            self._handle_error(error)

    def _handle_error(self, error: Exception) -> None:
        """Error handling chain: version → family → provider → bubble up."""
        try:
            self._version.handle_version_error(error)
        except type(error):
            pass
        except Exception:
            raise

        try:
            self._family.handle_model_error(error)
        except type(error):
            pass
        except Exception:
            raise

        self._provider.handle_api_error(error)

    # --- Result processing ---

    def _rectify_result(self, response_message: "AIMessage") -> "AIMessage":
        return self._family.rectify_result(response_message)

    def process_result(self, response_message: "AIMessage") -> "AgentResponse":
        return self._family.process_result(response_message)

    # --- Token counting ---

    async def get_num_tokens_from_messages(
        self,
        messages: list["BaseMessage"],
        tools: Optional[Sequence] = None,
    ) -> int:
        return await self._family.get_num_tokens_from_messages(
            self._chat_model, messages, tools
        )

    async def token_estimate(
        self,
        messages: Optional[list["BaseMessage"]] = None,
        tools: Optional[dict] = None,
    ) -> int:
        from langchain_core.messages import HumanMessage

        if tools:
            tools_tuple = tuple(
                sorted(
                    [
                        (name, func)
                        for name, func in tools.items()
                        if not getattr(func, "_disabled", False)
                    ],
                    key=lambda tool: tool[0],
                )
            )
            lc_tools = convert_tools(tools_tuple)
        else:
            lc_tools = None

        messages = self._preprocess_messages(messages)
        messages = copy.deepcopy(messages)
        for message in messages:
            if isinstance(message.content, list):
                content = []
                for content_object in message.content:
                    match content_object:
                        case {"type": "text"}:
                            content.append(content_object)
                        case dict():
                            content.append(
                                {"type": "text", "text": json.dumps(content_object)}
                            )
                        case _:
                            content.append(content_object)
                message.content = content

        return await self.get_num_tokens_from_messages(messages=messages, tools=lc_tools)

    # --- Context size ---

    def contextsize(self, model_name: str | None = None) -> int | None:
        if model_name is None:
            model_name = self._model_name
        if self._version.context_size is not None:
            return self._version.context_size
        family_size = self._family.context_size(model_name)
        if family_size is not None:
            return family_size
        return self._provider.context_size(model_name)
