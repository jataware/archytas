import os
import asyncio
import json
from typing import Any, Optional, Sequence, cast
from functools import lru_cache
import logging
from toki import Model
from toki.openrouter import OpenRouterMessage, OpenRouterToolCall, OpenRouterToolFunction
from toki.openrouter_models import ModelName, Attr, attributes_map

logger = logging.getLogger(__name__)

from .base import BaseArchytasModel, ModelConfig
from ..exceptions import AuthenticationError, ExecutionError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolCall as LangChainToolCall, ToolMessage, FunctionMessage


def to_langchain_tool_call(tool_call: OpenRouterToolCall) -> LangChainToolCall:
    return LangChainToolCall(name=tool_call['function']['name'], args=json.loads(tool_call['function']['arguments'] or '{}'), id=tool_call['id'], type="tool_call")

def to_openrouter_tool_call(tool_call: LangChainToolCall) -> OpenRouterToolCall:
    return OpenRouterToolCall(id=tool_call['id'] or '', type="function", function=OpenRouterToolFunction(name=tool_call["name"], arguments=json.dumps(tool_call["args"])))


class ChatOpenRouter:
    """Minimal adapter that emulates a chat model interface over OpenRouter's REST API."""
    def __init__(self, *, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1/chat/completions") -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._tools: Optional[Sequence[Any]] = None

        # Delegate to the minimal client
        # If model isn't recognized in our Literal list, still construct with raw string
        # this allows using newer models that haven't been updated in the openrouter_models.py file yet
        self._client = Model(model=cast(ModelName, model), openrouter_api_key=api_key)  # type: ignore[arg-type]

        # get the model attributes
        try:
            self._attrs = attributes_map[cast(ModelName, model)]
        except KeyError:
            self._attrs = Attr(context_size=200_000, supports_tools=True)
            logger.warning(f"Unrecognized OpenRouter model: '{model}' (this implies model is not officially listed in toki.openrouter_models). To get most up-to-date models list, consider cutting a new toki release (after regenerating models list with `toki-fetch-models` command). Attempting to continue with the following Attributes: {self._attrs}")
        
        if not self._attrs.supports_tools:
            raise ValueError(f"OpenRouter model '{model}' does not support tools. Archytas requires models to support tools. Please use a different model.")

    def bind_tools(self, tools: Sequence[Any]):
        self._tools = tools
        self._schemas = []
        for tool in tools:
            langchain_schema: dict = tool.tool_call_schema.schema()
            schema = {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': {
                        'type': 'object',
                        'properties': langchain_schema['properties'],
                        'required': langchain_schema.get('required', [])
                    },
                }
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
                    converted.append({"role": "assistant", 'content': content, 'tool_calls': list(map(to_openrouter_tool_call, tool_calls))})
                case AIMessage(): # Message without tool calls
                    converted.append({"role": "assistant", "content": content})
                case SystemMessage():
                    converted.append({"role": "system", "content": content})
                case ToolMessage():
                    converted.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": content})
                case _:
                    raise ValueError(f"Unexpected message type: {type(msg)}\n{msg=}")
        return converted

    def get_num_tokens_from_messages(self, *, messages: list[BaseMessage], tools: Optional[Sequence[Any]] = None) -> int:
        """Call OpenRouter to estimate prompt tokens by sending a completion request
        that is configured to produce no output tokens.

        Returns the `prompt_tokens` reported in the response `usage`.
        """
        raise NotImplementedError("Token count estimation for OpenRouter is not implemented")
        
        # TODO: this is very slow... basically it doubles the time to start generating a response
        if True: #self.skip_token_count:
            logger.warning("Skipping token count estimation for OpenRouter model")
            return 0

        # Convert messages to OpenRouter format
        converted_messages = self._convert_messages(messages)

        # Build tool schemas if tools were provided; otherwise, reuse any bound schemas
        schemas: list[dict] | None = None
        if tools is not None:
            try:
                tmp_schemas: list[dict] = []
                for tool in tools:
                    langchain_schema: dict = tool.tool_call_schema.schema()
                    tmp_schemas.append({
                        'type': 'function',
                        'function': {
                            'name': tool.name,
                            'description': tool.description,
                            'parameters': langchain_schema['properties'],
                            'required': langchain_schema['required'],
                        }
                    })
                schemas = tmp_schemas
            except Exception:
                # If tool introspection fails, ignore tools for token counting
                schemas = None
        else:
            schemas = getattr(self, "_schemas", None)

        # Try with zero max tokens to avoid any generation. If the API rejects 0,
        # fall back to 1 token with an immediate stop to minimize output tokens.
        try:
            self._client.complete(converted_messages, stream=False, tools=schemas, max_tokens=0)
        except Exception as first_error:
            try:
                self._client.complete(converted_messages, stream=False, tools=schemas, max_tokens=1, stop=[""])
            except Exception as fallback_error:
                raise ExecutionError(
                    f"Failed to estimate prompt tokens via OpenRouter. First attempt with max_tokens=0 error: {first_error}. "
                    f"Fallback with max_tokens=1 and immediate stop also failed: {fallback_error}"
                ) from fallback_error

        usage = self._client._usage_metadata
        if usage is None:
            raise ExecutionError("OpenRouter did not return usage metadata for the token count request.")
        return usage['prompt_tokens']

    def invoke(self, input: list[BaseMessage], *args, **kwargs) -> AIMessage:
        # Convert LangChain messages to OpenRouter format
        converted_messages = self._convert_messages(input)

        response = self._client.complete(converted_messages, stream=False, tools=self._schemas, **kwargs)

        assert self._client._usage_metadata is not None, "INTERNAL ERROR: Usage metadata was not set for previous completion call"
        usage_metadata = {
            'input_tokens': self._client._usage_metadata['prompt_tokens'],
            'output_tokens': self._client._usage_metadata['completion_tokens'],
            'total_tokens': self._client._usage_metadata['total_tokens']
        }

        if isinstance(response, dict):
            return AIMessage(content=response['thought'], tool_calls=list(map(to_langchain_tool_call, response['tool_calls'])), usage_metadata=usage_metadata)

        return AIMessage(content=response, usage_metadata=usage_metadata)

    async def ainvoke(self, input: list[BaseMessage], *args, **kwargs) -> AIMessage:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(input, *args, **kwargs))


class OpenRouterModel(BaseArchytasModel):
    """Archytas backend model for OpenRouter using direct REST calls."""
    DEFAULT_MODEL = "openrouter/auto"
    api_key: str = ""

    def auth(self, **kwargs) -> None:
        self.api_key = (
            kwargs.get("api_key")
            or getattr(self.config, "api_key", None)
            or os.getenv("OPENROUTER_API_KEY", "")
        )
        if not self.api_key:
            raise AuthenticationError("No OpenRouter API Key found. Set OPENROUTER_API_KEY or pass api_key.")

    def initialize_model(self, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        model_name = getattr(self.config, "model_name", None) or self.DEFAULT_MODEL
        return ChatOpenRouter(model=str(model_name), api_key=self.api_key)

    async def get_num_tokens_from_messages(
        self,
        messages: "list[BaseMessage]",
        tools: Optional[Sequence] = None,
    ) -> int:
        try:
            return self._model.get_num_tokens_from_messages(messages=messages, tools=tools)
        except Exception:
            return 0

    @lru_cache()
    def contextsize(self, model_name: Optional[str] = None) -> int | None:
        name = model_name or self.model_name
        default_value = 200_000
        if model_name is not None:
            try:
                return attributes_map[cast(ModelName, model_name)].context_size
            except KeyError:
                pass
        # Fallback default for safety so summarization threshold is usable
        logger.warning(f"OpenRouter context size unknown for model '{name}' (this implies model is not officially listed in archytas/models/openrouter_models.py, i.e. consider regenerating the file with `create_models_types_file()`). Using default context size: {default_value}.")
        return default_value