import json
import logging

from anthropic import AuthenticationError as AnthropicAuthenticError, RateLimitError
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage

from archytas.agent import AIMessage, BaseMessage

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig
from ..exceptions import AuthenticationError, ExecutionError

from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

from botocore.exceptions import ClientError

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
        super().__init__(config, **kwargs)
        self.last_messages: list[BaseMessage] | None = None
        self.tool_name_map = {}
        self.rev_tool_name_map = {}
        self.credentials_profile_name = None 
        self.aws_access_key = None 
        self.aws_secret_key = None
        self.aws_session_token = None

    def auth(self, **kwargs) -> None:
        # not handled - running on EC2 and expecting to authenticate via instance profile and IMDSv2
        # TODO: handle ec2/instance profile if we need it later. could be as easy as removing the exception below
        if 'credentials_profile_name' in kwargs:
            self.credentials_profile_name = kwargs['credentials_profile_name']
        else:
            # required if not using credentials
            aws_keys = ['aws_access_key', 'aws_secret_key']
            for key in aws_keys:
                if key in kwargs:
                    setattr(self, key, kwargs['key'])
                else: 
                    raise AuthenticationError(f'No credentials profile name specified, and one of aws_access_key or aws_secret_key was missing: Missing key: {key}')
            # NOT required, but if present, the above two also *must* exist.
            if 'aws_session_token' in kwargs:
                self.aws_session_token = kwargs['aws_session_token']

    def initialize_model(self, **kwargs):
        self.region: str = self.config.region or self.DEFAULT_REGION

        model = self.config.model_name or self.DEFAULT_MODEL
        if self.credentials_profile_name:
            return ChatBedrock(
                credentials_profile_name=self.credentials_profile_name,
                region_name=self.region,
                model=model,
                max_tokens=self.config.max_tokens or 4096
            )
        else:
            return ChatBedrock(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                aws_session_token=self.aws_session_token or None,
                region_name=self.region,
                model=model,
                max_tokens=self.config.max_tokens or 4096
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
        if isinstance(error, ClientError):
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
