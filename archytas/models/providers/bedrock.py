import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from langchain_aws import ChatBedrockConverse

from ..base_provider import BaseProvider
from ...exceptions import AuthenticationError

logger = logging.getLogger(__name__)

DEFERRED_TOKEN_VALUE = "***deferred***"


class BedrockProvider(BaseProvider):
    """Provider for AWS Bedrock."""

    DEFAULT_REGION: str = "us-east-1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | None = None,
        credentials_profile_name: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_session_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.region = region or os.environ.get("AWS_REGION", self.DEFAULT_REGION)
        self.credentials_profile_name = credentials_profile_name
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.aws_session_token = aws_session_token
        self.has_quota_permissions = False
        self.last_messages = None
        super().__init__(api_key=api_key, **kwargs)

    def auth(self, **kwargs: Any) -> None:
        # Check for credentials from extra kwargs
        if self.aws_access_key and self.aws_secret_key:
            return

        # Fall back to environment variables
        env_vars = {
            "AWS_ACCESS_KEY_ID": "aws_access_key",
            "AWS_SECRET_ACCESS_KEY": "aws_secret_key",
        }
        if any(var in os.environ for var in list(env_vars.keys()) + ["AWS_SESSION_TOKEN"]):
            for var, attr in env_vars.items():
                if var in os.environ:
                    setattr(self, attr, os.environ[var])
                else:
                    raise AuthenticationError(
                        f"missing one of required env vars: access_key or secret_key: {var}"
                    )
            self.aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

    def create_chat_model(self, model_name: str, **kwargs: Any) -> ChatBedrockConverse:
        max_tokens = kwargs.get("max_tokens", None)

        # Check permissions early
        try:
            self._check_service_quotas_permission()
            self.has_quota_permissions = True
        except AuthenticationError as e:
            logger.warning(
                f"AWS ListServiceQuotas permission missing - using default token limits: {e}"
            )
            self.has_quota_permissions = False

        if self.credentials_profile_name:
            return ChatBedrockConverse(
                credentials_profile_name=self.credentials_profile_name,
                region_name=self.region,
                model=model_name,
                max_tokens=max_tokens,
            )
        else:
            return ChatBedrockConverse(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                aws_session_token=self.aws_session_token or None,
                region_name=self.region,
                model=model_name,
                max_tokens=max_tokens,
            )

    def _check_service_quotas_permission(self) -> None:
        """Quick check if we have ListServiceQuotas permission."""
        try:
            quota_service = boto3.client("service-quotas", region_name=self.region)
            quota_service.list_service_quotas(ServiceCode="bedrock", MaxResults=1)
        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                raise AuthenticationError(
                    f"AWS credentials lack ListServiceQuotas permission. Error: {e}"
                )
            raise

    def context_size(self, model_name: str) -> int | None:
        limit = 50_000  # Reasonable default

        if not self.has_quota_permissions:
            logger.info(f"BedrockProvider.context_size: No quota permissions, using default: {limit}")
            return limit

        try:
            bedrock = boto3.client("bedrock", region_name=self.region)
            quota_service = boto3.client("service-quotas", region_name=self.region)
        except Exception as e:
            logger.error(f"BedrockProvider.context_size: Failed to create AWS clients: {e}")
            return limit

        # Resolve inference profiles to foundation models
        try:
            response = bedrock.get_inference_profile(inferenceProfileIdentifier=model_name)
            model_ids = set(
                model["modelArn"].split("foundation-model/")[1]
                for model in response["models"]
            )
            model_id = list(model_ids)[0] if model_ids else model_name
        except ClientError:
            model_id = model_name
        except Exception as err:
            model_id = model_name
            logger.error("Unexpected error caught. Proceeding anyway.", exc_info=err)

        # Look up human name for model
        try:
            from botocore.exceptions import ValidationError
            response = bedrock.get_foundation_model(modelIdentifier=model_id)
            model_human_name = response["modelDetails"]["modelName"]
        except (ValidationError, Exception) as err:
            model_human_name = None
            logger.error(err)

        # Iterate over quotas to extract quota limit
        try:
            paginator = quota_service.get_paginator("list_service_quotas")
            response_iterator = paginator.paginate(ServiceCode="bedrock")
            for page in response_iterator:
                for quota in page["Quotas"]:
                    if (
                        model_human_name
                        and model_human_name in quota["QuotaName"]
                        and "tokens per minute" in quota["QuotaName"]
                    ):
                        limit = int(quota["Value"])
                        break
        except ClientError as e:
            logger.error(f"BedrockProvider.context_size: AWS API error: {e}")
        except Exception as e:
            logger.error(f"BedrockProvider.context_size: Unexpected error: {e}")

        return limit

    def handle_api_error(self, error: Exception) -> None:
        if isinstance(error, NoCredentialsError) or (
            isinstance(error, ClientError)
            and error.response["ResponseMetadata"]["HTTPStatusCode"] == 403
        ):
            raise AuthenticationError(f"{error}")
        if self.last_messages:
            message_output = [msg.model_dump() for msg in self.last_messages]
            logger.warning(
                "An exception has occurred. Below are the messages sent in the most recent request:\n"
                + json.dumps(message_output, indent=2)
            )
        raise error
