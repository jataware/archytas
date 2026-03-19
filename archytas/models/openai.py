import os
import re
from functools import lru_cache
from typing import Any, Optional, Annotated, TYPE_CHECKING
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.llms.base import OpenAI
from langchain_core.messages import FunctionMessage, AIMessage
from langchain_core.tools import StructuredTool

from openai import AuthenticationError as OpenAIAuthenticationError, APIError, APIConnectionError, RateLimitError, OpenAIError, BadRequestError
from .base import BaseArchytasModel, ModelConfig, set_env_auth
from ..exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

if TYPE_CHECKING:
    from ..agent import AgentResponse

import logging
logger = logging.getLogger(__name__)

DEFERRED_TOKEN_VALUE = "***deferred***"

# Reasoning effort levels supported by each model family, keyed by model prefix pattern.
# Order within each list reflects what the model accepts.
REASONING_EFFORT_LEVELS: dict[str, list[str]] = {
    "o3":      ["low", "medium", "high"],
    "o4":      ["low", "medium", "high"],
    "gpt-5.1": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.2": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.3": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5.4": ["none", "low", "medium", "high", "xhigh"],
    "gpt-5":   ["minimal", "low", "medium", "high"],
}

# Preferred effort levels in priority order — the first level that the model
# supports will be selected.
PREFERRED_EFFORT_ORDER: list[str] = ["medium", "high", "low", "xhigh", "none"]

REASONING_EFFORT_REQUEST: Optional[str] = os.environ.get("LLM_REASONING_EFFORT", None)
REASONING_SUMMARY_REQUEST: Optional[str] = os.environ.get("LLM_REASONING_SUMMARY_TYPE", "concise")

class OpenAIModel(BaseArchytasModel):
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        else:
            auth_token = self.config.api_key
        if not auth_token:
            auth_token = DEFERRED_TOKEN_VALUE
        set_env_auth(OPENAI_API_KEY=auth_token)

        # Replace local auth token from value from environment variables to allow fetching preset auth variables in the
        # environment.
        auth_token = os.environ.get('OPENAI_API_KEY', DEFERRED_TOKEN_VALUE)

        if auth_token != DEFERRED_TOKEN_VALUE:
            self.config.api_key = auth_token
        # Reset the openai client with the new value, if needed.
        if getattr(self, "model", None):
            self.model.openai_api_key._secret_value = auth_token
            self.model.client = None
            self.model.async_client = None

            # This method reinitializes the clients
            self.model.validate_environment()

    def _get_supported_reasoning_efforts(self, model_name: str) -> list[str] | None:
        """Return the list of supported reasoning effort levels for the given model, or None if unsupported."""
        model_lower = model_name.lower()
        # Check prefixes longest-first so "gpt-5.4" matches before "gpt-5".
        for prefix in sorted(REASONING_EFFORT_LEVELS, key=len, reverse=True):
            if model_lower.startswith(prefix):
                return REASONING_EFFORT_LEVELS[prefix]
        return None

    def _get_reasoning_config(self, model_name: str) -> dict[str, str] | None:
        """Build the reasoning config dict for a model, or None if reasoning is not supported."""
        supported = self._get_supported_reasoning_efforts(model_name)
        if supported is None:
            return None
        if REASONING_EFFORT_REQUEST and REASONING_EFFORT_REQUEST in supported:
                return {"effort": REASONING_EFFORT_REQUEST, "summary": REASONING_SUMMARY_REQUEST}
        for effort in PREFERRED_EFFORT_ORDER:
            if effort in supported:
                return {"effort": effort, "summary": REASONING_SUMMARY_REQUEST}
        return None

    def initialize_model(self, **kwargs):
        try:
            model = self.config.model_name or self.DEFAULT_MODEL
            titoken_model_name = "gpt-4o" if 'gpt-4.1' in model else model
            model_kwargs: dict[str, Any] = dict(
                model=model,
                tiktoken_model_name=titoken_model_name,
            )
            reasoning = self._get_reasoning_config(model)
            if reasoning is not None:
                model_kwargs["reasoning"] = reasoning
            return ChatOpenAI(**model_kwargs)
        except (APIConnectionError, OpenAIError) as err:
            if not self.config.api_key:
                raise AuthenticationError("OpenAI API Key not set")
            else:
                raise AuthenticationError("OpenAI Authentication Error") from err

    def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        if self.model is None or not getattr(self.model, 'openai_api_key', None):
            raise AuthenticationError("OpenAI API Key missing")
        # Reasoning models don't accept a temperature keyword on invoke.
        model_lower = self.model.model_name.lower()
        if model_lower.startswith(("o3", "o4")) or "gpt-5" in model_lower:
            kwargs.pop("temperature", None)
        return super().ainvoke(input, config=config, stop=stop, **kwargs)

    def process_result(self, response_message: "AIMessage") -> "AgentResponse":
        from ..agent import AgentResponse
        content = response_message.content
        tool_calls = response_message.tool_calls

        text = ""
        metadata: dict[str, Any] = {}
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                block_type = item.get("type")
                if block_type == "text" and len(item.get("text") or "") > 0:
                    text_parts.append(str(item["text"]))
                elif block_type == "reasoning":
                    # Reasoning summary blocks contain a "summary" list of text items.
                    summary_parts = []
                    for summary_item in item.get("summary") or []:
                        if isinstance(summary_item, dict) and summary_item.get("text"):
                            summary_parts.append(summary_item["text"])
                    if summary_parts:
                        metadata["reasoning"] = "\n".join(summary_parts)
            if text_parts:
                text = "\n".join(text_parts)

        # if not text and tool_calls:
        #     tool_thoughts = [
        #         tool_call["args"].pop("thought", f"Calling tool '{tool_call['name']}'")
        #         for tool_call in tool_calls
        #     ]
        #     text = "\n".join(tool_thoughts)
        # if not text:
        #     text = "Thinking..."

        return AgentResponse(text=text, tool_calls=tool_calls, metadata=metadata)

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, OpenAIAuthenticationError):
            raise AuthenticationError("OpenAI Authentication Error") from error
        elif isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
        elif isinstance(error, (APIConnectionError, OpenAIError)) and not self.model.openai_api_key:
            raise AuthenticationError("OpenAI Authentication Error") from error
        elif isinstance(error, (BadRequestError)) and error.code == "context_length_exceeded":
            raise ContextWindowExceededError(error.body.get('message', None)) from error
        else:
            raise error

    @lru_cache()
    def contextsize(self, model_name = None):
        if model_name is None:
            model_name = self.model_name
        try:
            return OpenAI.modelname_to_contextsize(model_name)
        except ValueError as err:
            if 'gpt-4.1' in model_name:
                return 1_000_000
            elif model_name.startswith(('o3', 'o4')):
                return 200_000
            elif 'gpt-5' in model_name.lower():
                return 400_000
            raise
