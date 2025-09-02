import json
import logging
import os
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ValidationError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage

from archytas.agent import BaseMessage

from ..exceptions import AuthenticationError
from .base import BaseArchytasModel, ModelConfig

DEFERRED_TOKEN_VALUE = "***deferred***"

class BedrockModel(BaseArchytasModel):
    # foundation model names are the last part of the ARN after the slash
    # but we have to use inference profiles
    DEFAULT_MODEL: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    DEFAULT_REGION: str = "us-east-1"

    tool_name_map: dict
    rev_tool_name_map: dict

    credentials_profile_name: str | None
    aws_access_key: str | None
    aws_secret_key: str | None
    aws_session_token: str | None
    region: str

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        self.last_messages: list[BaseMessage] | None = None
        self.tool_name_map = {}
        self.rev_tool_name_map = {}
        self.credentials_profile_name = None
        self.aws_access_key = None
        self.aws_secret_key = None
        self.aws_session_token = None
        self.has_quota_permissions = False  # Track if we have ListServiceQuotas permission
        super().__init__(config, **kwargs)

    def auth(self, **kwargs) -> None:
        if self.config.model_extra:
            access = self.config.model_extra.get('aws_access_key', '')
            secret = self.config.model_extra.get('aws_secret_key', '')
            session = self.config.model_extra.get('aws_session_token', '')
            if access != '' and secret != '':
                self.aws_access_key = access
                self.aws_secret_key = secret
                self.aws_session_token = session if session != '' else None


        # not handled - running on EC2 and expecting to authenticate via instance profile and IMDSv2
        # TODO: handle ec2/instance profile if we need it later. could be as easy as removing the exception below
        if 'credentials_profile_name' in kwargs:
            self.credentials_profile_name = kwargs['credentials_profile_name']
            return

        # required if not using credentials or env vars. manually passing the argument should take
        # precedence over env vars
        aws_keys = ['aws_access_key', 'aws_secret_key']
        if any([(key in kwargs) for key in aws_keys + ['aws_session_token']]):
            for key in aws_keys:
                if key in kwargs:
                    setattr(self, key, kwargs['key'])
                else:
                    raise AuthenticationError(f'one of aws_access_key or aws_secret_key was missing: Missing key: {key}')
            # NOT required, but if present, the above two also *must* exist.
            self.aws_session_token = kwargs.get('aws_session_token', None)
            return

        env_vars = {
            'AWS_ACCESS_KEY_ID': 'aws_access_key',
            'AWS_SECRET_ACCESS_KEY': 'aws_secret_key'
        }
        if any([(var in os.environ) for var in list(env_vars.keys()) + ['AWS_SESSION_TOKEN']]):
            for var in env_vars:
                if var in os.environ:
                    setattr(self, env_vars[var], os.environ[var])
                else:
                    raise AuthenticationError(f'missing one of required env vars: access_key or secret_key: {var}')
            self.aws_session_token = os.environ.get('AWS_SESSION_TOKEN', None)


    def initialize_model(self, **kwargs):
        region = os.environ.get('AWS_REGION', self.DEFAULT_REGION)
        max_tokens = None
        if self.config.model_extra:
            if self.config.model_extra.get('region', '') != '':
                region = self.config.model_extra.get('region')
            max_tokens = self.config.model_extra.get('max_tokens', 4096)

        model = self.config.model_name or self.DEFAULT_MODEL

        # Check permissions early - warn if missing but don't fail
        try:
            self._check_service_quotas_permission()
            self.has_quota_permissions = True
        except AuthenticationError as e:
            logging.warning(f"AWS ListServiceQuotas permission missing - using default token limits: {e}")
            self.has_quota_permissions = False

        if self.credentials_profile_name:
            return ChatBedrockConverse(
                credentials_profile_name=self.credentials_profile_name,
                region_name=region,
                model=model,
                max_tokens=max_tokens
            )
        else:
            return ChatBedrockConverse(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                aws_session_token=self.aws_session_token or None,
                region_name=region,
                model=model,
                max_tokens=max_tokens or 4096
            )

    def _preprocess_messages(self, messages):
        from ..agent import AutoContextMessage, ContextMessage
        output = []

        system_messages = []
        # Combine all system/context/autocontext messages into a single initial system message
        for message in messages:
            match message:
                case SystemMessage() | ContextMessage() | AutoContextMessage():
                    system_messages.append(message.content)
                case _:
                    output.append(message)
        # Condense all context/system messages into a single first message as required by Anthropic
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        self.last_messages = [msg.model_copy(deep=True) for msg in output]
        return output

    def handle_invoke_error(self, error: BaseException):
        # client error catches a lot of credentials errors like incorrect profile names
        if isinstance(error, NoCredentialsError) or (isinstance(error, ClientError) and error.response["ResponseMetadata"]["HTTPStatusCode"] == 403):
            raise AuthenticationError(f"{error}")
        # TODO: Retry with delay on rate limit errors?
        # elif isinstance(error, RateLimitError):
        #     raise
        else:
            if self.last_messages:
                message_output = [msg.model_dump() for msg in self.last_messages]
                logging.warning(
                    "An exception has occurred. Below are the messages that were sent to in the most recent request:\n" +
                    json.dumps(message_output, indent=2)
                )
            raise

    def _check_service_quotas_permission(self):
        """Quick check if we have ListServiceQuotas permission - fail fast if not"""
        logging.info("BedrockModel._check_service_quotas_permission: Starting permission check")
        try:
            quota_service = boto3.client('service-quotas', region_name=os.environ.get("AWS_REGION", "us-east-1"))
            logging.info("BedrockModel._check_service_quotas_permission: About to test ListServiceQuotas")
            # Just try to list the first quota to test permission
            quota_service.list_service_quotas(ServiceCode='bedrock', MaxResults=1)
            logging.info("BedrockModel._check_service_quotas_permission: Permission check passed")
        except ClientError as e:
            logging.error(f"BedrockModel._check_service_quotas_permission: ClientError - {e.response['Error']['Code']}")
            if e.response['Error']['Code'] == 'AccessDeniedException':
                raise AuthenticationError(f"AWS credentials lack ListServiceQuotas permission. This is required for Bedrock token limits. Error: {e}")
            raise
        except Exception as e:
            logging.error(f"BedrockModel._check_service_quotas_permission: Unexpected error - {e}")
            raise

    @lru_cache()
    def contextsize(self, model_name = None):
        # Reasonable but small default
        limit = 50_000

        if model_name is None:
            model_name = self.model_name

        # Skip AWS API calls if we know permissions are missing
        if not self.has_quota_permissions:
            logging.info(f"BedrockModel.contextsize: Skipping AWS quota lookup (no permissions), using default limit: {limit}")
            return limit

        try:
            bedrock = boto3.client('bedrock', region_name=os.environ.get("AWS_REGION", "us-east-1"))
            quota_service = boto3.client('service-quotas', region_name=os.environ.get("AWS_REGION", "us-east-1"))
        except Exception as e:
            logging.error(f"BedrockModel.contextsize: Failed to create AWS clients, using default limit: {e}")
            return limit

        # Determine if the model name is an inference profile. If so, extract the model id for the profile.
        try:
            response = bedrock.get_inference_profile(inferenceProfileIdentifier=model_name)
            model_ids = set(model["modelArn"].split("foundation-model/")[1] for model in response["models"])
            if len(model_ids) > 1:
                logging.warning(f"Found multiple separate model ids. This shouldn't happen. ({', '.join(model_ids)})")
            if len(model_ids) == 0:
                logging.warning("No model ids found. Something is broken.")
            model_id = list(model_ids)[0]
        except ClientError as err:
            model_id = model_name
        except Exception as err:
            model_id = model_name
            logging.error("Unexpected error caught. Proceeding anyway.", exc_info=err)

        # Look up human name for model. Used in quota names.
        try:
            response = bedrock.get_foundation_model(
                modelIdentifier=model_id
            )
            model_human_name = response["modelDetails"]["modelName"]
        except ValidationError as err:
            model_human_name = None
            logging.error(err)

        # Iterate over quotas to extra quota limit
        try:
            paginator = quota_service.get_paginator('list_service_quotas')
            response_iterator = paginator.paginate(ServiceCode='bedrock')

            for page in response_iterator:
                for quota in page['Quotas']:
                    if model_human_name in quota['QuotaName'] and 'tokens per minute' in quota['QuotaName']:
                        limit = int(quota["Value"])
                        break
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                logging.error(f"BedrockModel.contextsize: ListServiceQuotas permission denied, using default limit: {limit}")
            else:
                logging.error(f"BedrockModel.contextsize: AWS API error, using default limit: {e}")
        except Exception as e:
            logging.error(f"BedrockModel.contextsize: Unexpected error getting quotas, using default limit: {e}")

        return limit
