import os

from langchain_openai import AzureChatOpenAI
from openai import APIConnectionError, OpenAIError

from archytas.exceptions import AuthenticationError
from archytas.models.base import ModelConfig, set_env_auth
from .openai import OpenAIModel, OpenAIProvider

DEFERRED_TOKEN_VALUE = "***deferred***"


class AzureOpenAIProvider(OpenAIProvider):
    def auth(self, model, **kwargs) -> None:
        auth_token = kwargs['api_key'] if 'api_key' in kwargs else model.config.api_key
        if not auth_token:
            # Defer to the environment without clobbering a user-set value.
            set_env_auth(AZURE_OPENAI_API_KEY=DEFERRED_TOKEN_VALUE)
            auth_token = os.environ.get('AZURE_OPENAI_API_KEY', DEFERRED_TOKEN_VALUE)
        else:
            # An explicit key wins over a stale environment value.
            os.environ['AZURE_OPENAI_API_KEY'] = auth_token
        if auth_token != DEFERRED_TOKEN_VALUE:
            model.config.api_key = auth_token
        if getattr(model, "model", None):
            model.model.azure_openai_api_key._secret_value = auth_token
            model.model.client = None
            model.model.async_client = None
            model.model.validate_environment()

    def initialize_model(self, model, **kwargs):
        if 'AZURE_OPENAI_ENDPOINT' in os.environ:
            model.endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        elif model.config.model_extra is None or 'endpoint' not in model.config.model_extra:
            raise AuthenticationError('Azure OpenAI models must have endpoint set.')
        else:
            model.endpoint = model.config.model_extra['endpoint']
        try:
            return AzureChatOpenAI(
                model=model.config.model_name or "",
                azure_endpoint=model.endpoint,
                api_version='2024-10-21'
            )
        except (APIConnectionError, OpenAIError) as err:
            if not model.config.api_key:
                raise AuthenticationError("OpenAI API Key not set")
            raise AuthenticationError("OpenAI Authentication Error") from err


class AzureOpenAIModel(OpenAIModel):
    DEFAULT_PROVIDER = AzureOpenAIProvider

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config, **kwargs)


class AzureFoundryModel():
    pass
