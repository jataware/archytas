import json
import re
import logging

from anthropic import AuthenticationError as AnthropicAuthenticError, RateLimitError
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel as PydanticModel, Field

from archytas.agent import AIMessage, BaseMessage

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
    tool_name_map: dict
    rev_tool_name_map: dict

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.last_messages: list[BaseMessage] = None
        self.tool_name_map = {}
        self.rev_tool_name_map = {}

    def auth(self, **kwargs) -> None:
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        elif 'api_key' in self.config:
            self.api_key = self.config['api_key']
        if not self.api_key:
            raise AuthenticationError("No auth credentials found.")

    def initialize_model(self, **kwargs):
        return ChatAnthropic(model=self.config.get("model_name", "claude-2.1"), api_key=self.api_key)

    def _preprocess_messages(self, messages):
        from ..agent import AutoContextMessage, ContextMessage
        output = []

        system_messages = []
        for message in messages:
            match message:
                case SystemMessage() | ContextMessage() | AutoContextMessage():
                    system_messages.append(message.content)
    #             case AIMessage():
    #                 # Duplicate mesage so we don't change raw storage
    #                 msg = message.model_copy(deep=True)
    #                 if not isinstance(msg.content, list):
    #                     content = [
    #                         {
    #                             "type": "text",
    #                             "text": msg.content,
    #                         }
    #                     ]
    #                     msg.content = content
    #                 for tool_call in msg.tool_calls:
    #                     raw_tool_name= tool_call.get("name", "DummyTool")
    #                     if raw_tool_name not in self.tool_name_map:
    #                         tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_tool_name)
    #                         self.tool_name_map[raw_tool_name] = tool_name
    #                         self.rev_tool_name_map[tool_name] = raw_tool_name
    #                     else:
    #                         tool_name = self.tool_name_map[raw_tool_name]
    #                     if tool_name not in ("final_answer", "fail_task"):
    #                         msg.content.append({
    #                             "type": "tool_use",
    #                             "id": tool_call.get("id"),
    #                             "name": tool_name,
    #                             "input": tool_call.get("args"),
    #                         })
    #                 msg.tool_calls = []
    #                 output.append(msg)
                case _:
                    output.append(message)
        # Condense all context/system messages into a single first message as required by Anthropic
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        self.last_messages = [msg.model_copy(deep=True) for msg in output]
        return output

    # def _rectify_result(self, response_message: AIMessage):
    #     orig_content = response_message.content
    #     message_texts = []
    #     tool_usages = []
    #     if isinstance(response_message.content, list):
    #         if len(response_message.content) == 1:
    #             item = response_message.content[0]
    #             if item.get("type") == "text":
    #                 message_texts.append(item["text"])
    #             elif item.get("type") == "tool_use":
    #                 tool_name = item["name"]
    #                 if tool_name in self.rev_tool_name_map:
    #                     tool_name = self.rev_tool_name_map[tool_name]
    #                 tool_input = json.loads(item["input"]["arg_string"])
    #                 message_texts.append(json.dumps({
    #                     "thought": f"I need run tool `{tool_name}`",
    #                     "tool": tool_name,
    #                     "tool_input": tool_input,
    #                 }, indent=2))
    #         else:
    #             for item in response_message.content:
    #                 if item.get("type") == "text":
    #                     message_texts.append(item["text"])
    #         message_text = "\n".join(message_texts)
    #         response_message.content = message_text
    #         response_message.tool_calls = []
    #     return super()._rectify_result(response_message)

    # def process_result(self, response_message: AIMessage):
    #     content = super().process_result(response_message)
    #     try:
    #         result = extract_json(content)
    #         if isinstance(result, list):
    #             # Only perform the first action
    #             return result[0]
    #     except Exception as e:
    #         result = content
    #     return result

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, AnthropicAuthenticError):
            raise AuthenticationError("Anthropic Authentication Error") from error
        # TODO: Retry with delay on rate limit errors?
        elif isinstance(error, RateLimitError):
            num_tokens = self.ChatAnthropic.get_num_tokens_from_messages(self.last_messages)
            logging.error(f"Rate limit error: {num_tokens} tokens")
            raise
        else:
            if self.last_messages:
                message_output = [msg.model_dump() for msg in self.last_messages]
                logging.warning(
                    "An exception has occurred. Below are the messages that were sent to in the most recent request:\n" +
                    json.dumps(message_output, indent=2)
                )
            raise
