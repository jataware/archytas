from langchain_openai import AzureChatOpenAI
from archytas.exceptions import AuthenticationError
from archytas.models.base import ModelConfig, set_env_auth
from openai import AuthenticationError as OpenAIAuthenticationError, APIError, APIConnectionError, RateLimitError, OpenAIError
from .openai import OpenAIModel
import os

DEFERRED_TOKEN_VALUE = "***deferred***"

class AzureOpenAIModel(OpenAIModel):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config, **kwargs)

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        else:
            auth_token = self.config.api_key
        if not auth_token:
            auth_token = DEFERRED_TOKEN_VALUE
        set_env_auth(AZURE_OPENAI_API_KEY=auth_token)

        # Replace local auth token from value from environment variables to allow fetching preset auth variables in the
        # environment.
        auth_token = os.environ.get('AZURE_OPENAI_API_KEY', DEFERRED_TOKEN_VALUE)

        if auth_token != DEFERRED_TOKEN_VALUE:
            self.config.api_key = auth_token
        # Reset the openai client with the new value, if needed.
        if getattr(self, "model", None):
            self.model.azure_openai_api_key._secret_value = auth_token
            self.model.client = None
            self.model.async_client = None

            # This method reinitializes the clients
            self.model.validate_environment()

    def initialize_model(self, **kwargs):
        if 'AZURE_OPENAI_ENDPOINT' in os.environ:
            self.endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        elif self.config.model_extra is None or 'endpoint' not in self.config.model_extra:
            raise AuthenticationError('Azure OpenAI models must have endpoint set.')
        else:
            self.endpoint = self.config.model_extra['endpoint']
        try:
            return AzureChatOpenAI(
                model=self.config.model_name or "",
                azure_endpoint=self.endpoint,
                api_version='2024-10-21'
            )
        except (APIConnectionError, OpenAIError) as err:
            if not self.config.api_key:
                raise AuthenticationError("OpenAI API Key not set")
            else:
                raise AuthenticationError("OpenAI Authentication Error") from err


class AzureFoundryModel():
    pass
