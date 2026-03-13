import os
import asyncio
import json
import logging
from typing import Any, Optional, Sequence, cast

from toki import Model as TokiModel
from toki.openrouter import OpenRouterMessage, OpenRouterToolCall, OpenRouterToolFunction
from toki.openrouter_models import ModelName, Attr, attributes_map
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolCall as LangChainToolCall, ToolMessage,
)

from ..base_provider import BaseProvider
from ...exceptions import AuthenticationError, ExecutionError

logger = logging.getLogger(__name__)


def to_langchain_tool_call(tool_call: OpenRouterToolCall) -> LangChainToolCall:
    return LangChainToolCall(
        name=tool_call["function"]["name"],
        args=json.loads(tool_call["function"]["arguments"] or "{}"),
        id=tool_call["id"],
        type="tool_call",
    )


def to_openrouter_tool_call(tool_call: LangChainToolCall) -> OpenRouterToolCall:
    return OpenRouterToolCall(
        id=tool_call["id"] or "",
        type="function",
        function=OpenRouterToolFunction(
            name=tool_call["name"],
            arguments=json.dumps(tool_call["args"]),
        ),
    )


class ChatOpenRouter:
    """Minimal adapter that emulates a chat model interface over OpenRouter's REST API."""

    def __init__(
        self, *, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._tools: Optional[Sequence[Any]] = None
        self._client = TokiModel(model=cast(ModelName, model), openrouter_api_key=api_key)

        try:
            self._attrs = attributes_map[cast(ModelName, model)]
        except KeyError:
            self._attrs = Attr(context_size=200_000, supports_tools=True)
            logger.warning(
                f"Unrecognized OpenRouter model: '{model}'. "
                f"Attempting to continue with default attributes: {self._attrs}"
            )

        if not self._attrs.supports_tools:
            raise ValueError(
                f"OpenRouter model '{model}' does not support tools. "
                "Archytas requires models to support tools."
            )

    def bind_tools(self, tools: Sequence[Any]):
        self._tools = tools
        self._schemas = []
        for tool in tools:
            langchain_schema: dict = tool.tool_call_schema.schema()
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": langchain_schema["properties"],
                        "required": langchain_schema.get("required", []),
                    },
                },
            }
            self._schemas.append(schema)
        return self

    def _convert_messages(self, messages: list[BaseMessage]) -> list[OpenRouterMessage]:
        def serialize_content(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                        parts.append(item["text"])
                    else:
                        parts.append(json.dumps(item))
                return "\n".join(parts)
            return json.dumps(content)

        converted: list[OpenRouterMessage] = []
        for msg in messages:
            content = serialize_content(msg.content)
            match msg:
                case HumanMessage():
                    converted.append({"role": "user", "content": content})
                case AIMessage(tool_calls=list() as tool_calls):
                    converted.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": list(map(to_openrouter_tool_call, tool_calls)),
                    })
                case AIMessage():
                    converted.append({"role": "assistant", "content": content})
                case SystemMessage():
                    converted.append({"role": "system", "content": content})
                case ToolMessage():
                    converted.append({
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": content,
                    })
                case _:
                    raise ValueError(f"Unexpected message type: {type(msg)}\n{msg=}")
        return converted

    def get_num_tokens_from_messages(
        self, *, messages: list[BaseMessage], tools: Optional[Sequence[Any]] = None
    ) -> int:
        raise NotImplementedError("Token count estimation for OpenRouter is not implemented")

    def invoke(self, input: list[BaseMessage], *args, **kwargs) -> AIMessage:
        converted_messages = self._convert_messages(input)
        response = self._client.complete(
            converted_messages, stream=False, tools=self._schemas, **kwargs
        )
        assert self._client._usage_metadata is not None
        usage_metadata = {
            "input_tokens": self._client._usage_metadata["prompt_tokens"],
            "output_tokens": self._client._usage_metadata["completion_tokens"],
            "total_tokens": self._client._usage_metadata["total_tokens"],
        }
        if isinstance(response, dict):
            return AIMessage(
                content=response["thought"],
                tool_calls=list(map(to_langchain_tool_call, response["tool_calls"])),
                usage_metadata=usage_metadata,
            )
        return AIMessage(content=response, usage_metadata=usage_metadata)

    async def ainvoke(self, input: list[BaseMessage], *args, **kwargs) -> AIMessage:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(input, *args, **kwargs))


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API."""

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, **kwargs)

    def auth(self, **kwargs: Any) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise AuthenticationError(
                "No OpenRouter API Key found. Set OPENROUTER_API_KEY or pass api_key."
            )

    def create_chat_model(self, model_name: str, **kwargs: Any) -> ChatOpenRouter:
        return ChatOpenRouter(model=model_name, api_key=self.api_key)

    def context_size(self, model_name: str) -> int | None:
        default_value = 200_000
        try:
            return attributes_map[cast(ModelName, model_name)].context_size
        except KeyError:
            logger.warning(
                f"OpenRouter context size unknown for model '{model_name}'. "
                f"Using default: {default_value}."
            )
            return default_value
