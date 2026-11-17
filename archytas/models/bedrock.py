"""Compatibility wrapper for models served by AWS Bedrock."""
from .base import BaseArchytasModel, ModelConfig
from .families.openai import OpenAIModelFamily
from .providers.bedrock import DEFERRED_TOKEN_VALUE, BedrockProvider


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
