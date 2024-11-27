import json
from anthropic import AuthenticationError as AnthropicAuthenticError, RateLimitError
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel as PydanticModel, Field

from archytas.agent import AIMessage

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig
from ..exceptions import AuthenticationError, ExecutionError
from ..utils import extract_json

class DummyTool(PydanticModel):
    """
    Dummy Tool
    """
    input: str = Field(..., description="input")


class AnthropicModel(BaseArchytasModel):
    api_key: str = ""

    def auth(self, **kwargs) -> None:
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        elif 'api_key' in self.config:
            self.api_key = self.config['api_key']
        if not self.api_key:
            raise AuthenticationError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        return ChatAnthropic(model=self.config.get("model_name", "claude-2.1"), api_key=self.api_key).bind_tools([
            DummyTool
        ])

    def _preprocess_messages(self, messages):
        from ..agent import AutoContextMessage, ContextMessage
        output = []
        system_messages = []
        for message in messages:
            match message:
                case SystemMessage() | ContextMessage() | AutoContextMessage():
                    system_messages.append(message.content)
                case AIMessage():
                    # Duplicate mesage so we don't change raw storage
                    msg = message.model_copy()
                    if isinstance(msg.content, list):
                        msg.content = "\n".join(item.get("text") for item in msg.content if item.get("type") == "text")
                    msg.tool_calls = [{**call, 'name': 'DummyTool'} for call in msg.tool_calls if call.get('name', None) != 'DummyTool']
                    output.append(msg)
                case _:
                    output.append(message)
        # Condense all context/system messages into a single first message as required by Anthropic
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output

    def process_result(self, response_message: AIMessage):
        content = super().process_result(response_message)
        try:
            result = extract_json(content)
            if isinstance(result, list):
                # Only perform the first action
                return result[0]
        except Exception as e:
            result = content
        return result

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, AnthropicAuthenticError):
            raise AuthenticationError("Anthropic Authentication Error") from error
        # TODO: Retry with delay on rate limit errors?
        # elif isinstance(error, RateLimitError):
        #     raise
        else:
            raise
