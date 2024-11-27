import json
from typing import TYPE_CHECKING
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI, ChatGoogleGenerativeAIError

from .base import BaseArchytasModel
from ..message_schemas import ToolUseRequest
from ..exceptions import AuthenticationError, ExecutionError

if TYPE_CHECKING:
    from ..agent import SystemMessage, AutoContextMessage, AIMessage, ToolMessage, FunctionMessage


class GeminiModel(BaseArchytasModel):
    api_key: str

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        elif 'api_key' in self.config:
            auth_token = self.config['api_key']
        if auth_token:
            self.api_key = auth_token
        else:
            raise AuthenticationError("Gemini API key not provided.")

    def initialize_model(self, **kwargs):
        from langchain_google_genai._function_utils import _dict_to_gapic_schema
        schema = ToolUseRequest.model_json_schema()
        for property in schema["properties"].values():
            if "type" not in property:
                property["type"] = "string"
                property["description"] += " (As a JSON object encoded as a string)"
        schema = _dict_to_gapic_schema(schema)
        return ChatGoogleGenerativeAI(model=self.config.get("model_name", "gpt-4o"), api_key=self.api_key).bind(
                tool_config={
                    "function_calling_config": {
                        "mode": "NONE",
                    }
                },
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )

    async def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        # Gemini doesn't accept a temperature keyword on invoke
        kwargs.pop("temperature")
        return await super().ainvoke(input, config=config, stop=stop, **kwargs)

    def _preprocess_messages(self, messages):
        from ..agent import SystemMessage, AutoContextMessage
        output = []
        system_messages = []
        for message in messages:
            if isinstance(message, (SystemMessage, AutoContextMessage)):
                system_messages.append(message.content)
            else:
                output.append(message)
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, ChatGoogleGenerativeAIError):
            if any(('400 API key not valid' in arg for arg in error.args)):
                raise AuthenticationError("API key invalid.") from error
        raise ExecutionError(*error.args) from error

    def process_result(self, response_message: "AIMessage"):
        content = response_message.content
        try:
            if isinstance(content, str):
                content = json.loads(content)
            content["tool_input"] = json.loads(content["tool_input"])
        except json.JSONDecodeError:
            pass
        return content
