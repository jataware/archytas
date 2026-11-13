import json
import logging
import os

from botocore.exceptions import ClientError, NoCredentialsError
from langchain_aws import ChatBedrockConverse

from ..exceptions import AuthenticationError
from .base import BaseArchytasModel, ModelConfig
from .families import AnthropicModelFamily, OpenAIModelFamily
from .provider import ModelProvider

DEFERRED_TOKEN_VALUE = "***deferred***"


class BedrockProvider(ModelProvider):
    MODEL_SPECIFIC_CONTENT_TYPES = frozenset({"reasoning_content"})
    DEFAULT_REGION = "us-east-1"

    def auth(self, model, **kwargs) -> None:
        for attribute in (
            'credentials_profile_name', 'aws_access_key', 'aws_secret_key', 'aws_session_token',
        ):
            if not hasattr(model, attribute):
                setattr(model, attribute, None)
        if model.config.model_extra:
            access = model.config.model_extra.get('aws_access_key', '')
            secret = model.config.model_extra.get('aws_secret_key', '')
            session = model.config.model_extra.get('aws_session_token', '')
            if access != '' and secret != '':
                model.aws_access_key = access
                model.aws_secret_key = secret
                model.aws_session_token = session if session != '' else None

        # not handled - running on EC2 and expecting to authenticate via instance profile and IMDSv2
        # TODO: handle ec2/instance profile if we need it later. could be as easy as removing the exception below
        if 'credentials_profile_name' in kwargs:
            model.credentials_profile_name = kwargs['credentials_profile_name']
            return

        # required if not using credentials or env vars. manually passing the argument should take
        # precedence over env vars
        aws_keys = ['aws_access_key', 'aws_secret_key']
        if any([(key in kwargs) for key in aws_keys + ['aws_session_token']]):
            for key in aws_keys:
                if key in kwargs:
                    setattr(model, key, kwargs['key'])
                else:
                    raise AuthenticationError(f'one of aws_access_key or aws_secret_key was missing: Missing key: {key}')
            # NOT required, but if present, the above two also *must* exist.
            model.aws_session_token = kwargs.get('aws_session_token', None)
            return

        env_vars = {
            'AWS_ACCESS_KEY_ID': 'aws_access_key',
            'AWS_SECRET_ACCESS_KEY': 'aws_secret_key'
        }
        if any([(var in os.environ) for var in list(env_vars.keys()) + ['AWS_SESSION_TOKEN']]):
            for var in env_vars:
                if var in os.environ:
                    setattr(model, env_vars[var], os.environ[var])
                else:
                    raise AuthenticationError(f'missing one of required env vars: access_key or secret_key: {var}')
            model.aws_session_token = os.environ.get('AWS_SESSION_TOKEN', None)

    def initialize_model(self, model, **kwargs):
        region = os.environ.get('AWS_REGION', getattr(model, 'DEFAULT_REGION', self.DEFAULT_REGION))
        max_tokens = None
        if model.config.model_extra:
            if model.config.model_extra.get('region', '') != '':
                region = model.config.model_extra.get('region')
            max_tokens = model.config.model_extra.get('max_tokens', None)
        name = model.config.model_name or model.DEFAULT_MODEL
        if model.credentials_profile_name:
            return ChatBedrockConverse(
                credentials_profile_name=model.credentials_profile_name,
                region_name=region,
                model=name,
                max_tokens=max_tokens
            )
        return ChatBedrockConverse(
            aws_access_key_id=model.aws_access_key,
            aws_secret_access_key=model.aws_secret_key,
            aws_session_token=model.aws_session_token or None,
            region_name=region,
            model=name,
            max_tokens=max_tokens
        )

    def preprocess_messages(self, model, messages):
        # Only Claude uses these markers. Its message ordering is supplied by
        # the family; the marker encoding is specific to the provider API.
        if not isinstance(model.model_family, AnthropicModelFamily):
            return messages
        from .anthropic import _attach_cache_control, _prompt_cache_disabled
        output = list(messages)
        if not _prompt_cache_disabled():
            tail_count = getattr(model, "tail_injection_count", 0) or 0
            last_persisted_idx = len(output) - tail_count - 1
            if last_persisted_idx > 0:
                output[last_persisted_idx] = _attach_cache_control(output[last_persisted_idx])
        return output

    def handle_invoke_error(self, model, error: BaseException):
        # client error catches a lot of credentials errors like incorrect profile names
        if isinstance(error, NoCredentialsError) or (isinstance(error, ClientError) and error.response["ResponseMetadata"]["HTTPStatusCode"] == 403):
            raise AuthenticationError(f"{error}")
        if model.last_sent_messages:
            try:
                message_output = [msg.model_dump() for msg in model.last_sent_messages]
                logging.warning(
                    "An exception has occurred. Below are the messages that were sent to in the most recent request:\n" +
                    json.dumps(message_output, indent=2)
                )
            except Exception:
                # Diagnostic logging must never mask the original error.
                pass
        raise error

    def fallback_contextsize(self, model, model_name=None):
        return 50_000


class BedrockModel(BaseArchytasModel):
    # foundation model names are the last part of the ARN after the slash
    # but we have to use inference profiles
    DEFAULT_MODEL: str = "us.openai.gpt-6.1-terra"
    DEFAULT_REGION: str = "us-east-1"
    DEFAULT_PROVIDER = BedrockProvider
    DEFAULT_FAMILY = OpenAIModelFamily

    tool_name_map: dict
    rev_tool_name_map: dict
    credentials_profile_name: str | None
    aws_access_key: str | None
    aws_secret_key: str | None
    aws_session_token: str | None
    region: str

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        self.tool_name_map = {}
        self.rev_tool_name_map = {}
        self.credentials_profile_name = None
        self.aws_access_key = None
        self.aws_secret_key = None
        self.aws_session_token = None
        super().__init__(config, **kwargs)
