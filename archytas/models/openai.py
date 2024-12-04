import re
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import FunctionMessage, AIMessage

from openai import AuthenticationError as OpenAIAuthenticationError, APIError, APIConnectionError, RateLimitError, OpenAIError
from .base import BaseArchytasModel, ModelConfig, set_env_auth
from ..exceptions import AuthenticationError, ExecutionError

DEFERRED_TOKEN_VALUE = "***deferred***"

class OpenAIModel(BaseArchytasModel):

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        elif 'api_key' in self.config:
            auth_token = self.config['api_key']
        if auth_token is None:
            auth_token = DEFERRED_TOKEN_VALUE
        set_env_auth(OPENAI_API_KEY=auth_token)

        if auth_token != DEFERRED_TOKEN_VALUE:
            self.config['api_key'] = auth_token
        # Reset the openai client with the new value, if needed.
        if getattr(self, "model", None):
            self.model.openai_api_key._secret_value = auth_token
            self.model.client = None
            self.model.async_client = None

            # This method reinitializes the clients
            self.model.validate_environment()

    def initialize_model(self, **kwargs):
        try:
            return ChatOpenAI(model=self.config.get("model_name", "gpt-4o"))
        except (APIConnectionError, OpenAIError) as err:
            if not self.config.get('api_key', None):
                raise AuthenticationError("OpenAI API Key not set")
            else:
                raise AuthenticationError("OpenAI Authentication Error") from err

    def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        if self.model is None or not getattr(self.model, 'openai_api_key', None):
            raise AuthenticationError("OpenAI API Key missing")
        return super().ainvoke(input, config=config, stop=stop, **kwargs)

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
        elif isinstance(error, (APIConnectionError, OpenAIError)) and not self.model.openai_api_key:
            raise AuthenticationError("OpenAI Authentication Error") from error
        else:

            raise error
