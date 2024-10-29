import re
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import FunctionMessage, AIMessage

from openai import AuthenticationError as OpenAIAuthenticationError, APIError, APIConnectionError, RateLimitError
from .base import BaseArchytasModel, ModelConfig, set_env_auth
from ..exceptions import AuthenticationError, ExecutionError


class OpenAIModel(BaseArchytasModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        elif 'api_key' in self.config:
            auth_token = self.config['api_key']
        if auth_token:
            set_env_auth(OPENAI_API_KEY=auth_token)
        else:
            raise ValueError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        return ChatOpenAI(model=self.config.get("model_name", "gpt-4o"))

    def _preprocess_messages(self, messages):
        output = []
        for message in messages:
            if isinstance(message, FunctionMessage):
                message.name = re.sub(r'[^a-zA-Z0-9_-]', '_', message.name)
            elif isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    tool_call["name"] = re.sub(r'[^a-zA-Z0-9_-]', '_', tool_call["name"])
            output.append(message)
        return output

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, OpenAIAuthenticationError):
            raise AuthenticationError("OpenAI Authentication Error") from error
        elif isinstance(error, RateLimitError):
            raise ExecutionError(error.message) from error
