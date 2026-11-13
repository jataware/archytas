import logging
import os
from typing import Any

from langchain_openai.chat_models import ChatOpenAI
from openai import (
    AuthenticationError as OpenAIAuthenticationError, APIConnectionError,
    RateLimitError, OpenAIError, BadRequestError,
)

from ..base import set_env_auth
from .base import ModelProvider
from ...exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

logger = logging.getLogger(__name__)
DEFERRED_TOKEN_VALUE = "***deferred***"

class OpenAIProvider(ModelProvider):
    def auth(self, model, **kwargs) -> None:
        auth_token = kwargs['api_key'] if 'api_key' in kwargs else model.config.api_key
        if not auth_token:
            set_env_auth(OPENAI_API_KEY=DEFERRED_TOKEN_VALUE)
            auth_token = os.environ.get('OPENAI_API_KEY', DEFERRED_TOKEN_VALUE)
        else:
            os.environ['OPENAI_API_KEY'] = auth_token
        if auth_token != DEFERRED_TOKEN_VALUE:
            model.config.api_key = auth_token
        if getattr(model, "model", None):
            model.model.openai_api_key._secret_value = auth_token
            model.model.client = None
            model.model.async_client = None
            model.model.validate_environment()

    def reasoning_config(self, model, model_name: str, profile: dict | None = None) -> dict | None:
        settings = model.model_family.reasoning_config(model, model_name, profile)
        if settings is None or settings.effort is None:
            return None
        config = {"effort": settings.effort}
        if settings.summary is not None:
            config["summary"] = settings.summary
        return config

    def initialize_model(self, model, **kwargs):
        try:
            name = model.config.model_name or model.DEFAULT_MODEL
            model_kwargs: dict[str, Any] = dict(
                model=name,
                tiktoken_model_name="gpt-4o" if 'gpt-4.1' in name else name,
            )
            if model.config.model_extra and model.config.model_extra.get("base_url"):
                model_kwargs["base_url"] = model.config.model_extra["base_url"]
            chat_model = ChatOpenAI(**model_kwargs)
            reasoning = model._get_reasoning_config(name, getattr(chat_model, "profile", None))
            if reasoning is not None:
                model_kwargs["reasoning"] = reasoning
                chat_model = ChatOpenAI(**model_kwargs)
            return chat_model
        except (APIConnectionError, OpenAIError) as err:
            if not model.config.api_key:
                raise AuthenticationError("OpenAI API Key not set")
            raise AuthenticationError("OpenAI Authentication Error") from err

    async def ainvoke(self, model, invoke, input, *, config=None, stop=None, **kwargs):
        if model.model is None or not getattr(model.model, 'openai_api_key', None):
            raise AuthenticationError("OpenAI API Key missing")
        return await invoke(input, config=config, stop=stop, **kwargs)

    def process_result(self, model, response_message):
        from ...agent import AgentResponse
        content = response_message.content
        text = ""
        metadata: dict[str, Any] = {}
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
                elif item.get("type") == "reasoning":
                    parts = [
                        entry["text"] for entry in item.get("summary") or []
                        if isinstance(entry, dict) and entry.get("text")
                    ]
                    if parts:
                        metadata["reasoning"] = "\n".join(parts)
            text = "\n".join(text_parts)
        return AgentResponse(text=text, tool_calls=response_message.tool_calls, metadata=metadata)

    def handle_invoke_error(self, model, error: BaseException):
        if isinstance(error, OpenAIAuthenticationError):
            raise AuthenticationError("OpenAI Authentication Error") from error
        if isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
        if isinstance(error, (APIConnectionError, OpenAIError)) and not model.model.openai_api_key:
            raise AuthenticationError("OpenAI Authentication Error") from error
        if isinstance(error, BadRequestError) and error.code == "context_length_exceeded":
            raise ContextWindowExceededError(error.body.get('message', None)) from error
        raise error

