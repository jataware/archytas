import os
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI, ChatGoogleGenerativeAIError

from .base import ModelProvider
from ...exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

class GeminiProvider(ModelProvider):
    def auth(self, model, **kwargs) -> None:
        auth_token = (
            kwargs.get('api_key')
            or model.config.api_key
            or os.environ.get('GEMINI_API_KEY', '')
        )
        if auth_token:
            model.api_key = auth_token
        else:
            raise AuthenticationError("Gemini API key not provided.")

    def reasoning_config(self, model, model_name: str, profile: dict | None = None) -> dict | None:
        settings = model.model_family.reasoning_config(model, model_name, profile)
        if settings is None:
            return None
        config: dict[str, object] = {"include_thoughts": settings.include_thoughts}
        if settings.budget is not None:
            config["thinking_budget_token_limit"] = settings.budget
        elif settings.effort is not None:
            config["thinking_level"] = settings.effort
        return config

    def initialize_model(self, model, **kwargs):
        name = model.config.model_name or model.DEFAULT_MODEL
        model_kwargs = dict(model=name, api_key=model.api_key)
        chat_model = ChatGoogleGenerativeAI(**model_kwargs)
        thinking = model._get_thinking_config(name, getattr(chat_model, "profile", None))
        if thinking is not None:
            model_kwargs.update(thinking)
            chat_model = ChatGoogleGenerativeAI(**model_kwargs)
        return chat_model

    def prepare_invoke_kwargs(self, model, kwargs: dict) -> dict:
        if "temperature" in kwargs:
            generation_config = dict(kwargs.get("generation_config") or {})
            generation_config["temperature"] = kwargs.pop("temperature")
            kwargs["generation_config"] = generation_config
        return kwargs

    def handle_invoke_error(self, model, error: BaseException):
        if isinstance(error, ChatGoogleGenerativeAIError):
            if any(('400 API key not valid' in arg for arg in error.args)):
                raise AuthenticationError("API key invalid.") from error
            if any(('exceeds the maximum number of tokens allowed' in arg for arg in error.args)):
                raise ContextWindowExceededError("Context window maximum tokens exceeded.") from error
        raise ExecutionError(*error.args) from error

    def process_result(self, model, response_message):
        from ...agent import AgentResponse
        content = response_message.content
        text = ""
        metadata = {}
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            labeled_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                block_type = item.get("type")
                if block_type == "text" and len(item.get("text") or "") > 0:
                    labeled_parts.append(str(item['text']))
                elif block_type == "reasoning":
                    metadata["reasoning"] = item.get("reasoning")
                elif block_type == "thinking":
                    metadata["thinking"] = item.get("thinking")
            if labeled_parts:
                text = "\n".join(labeled_parts)
        return AgentResponse(text=text, tool_calls=response_message.tool_calls, metadata=metadata)

