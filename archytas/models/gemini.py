import json
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI, ChatGoogleGenerativeAIError

from .base import BaseArchytasModel, EnvironmentAuth, ModelConfig, set_env_auth
from ..exceptions import AuthenticationError, ExecutionError


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
        return ChatGoogleGenerativeAI(model=self.config.get("model_name", "gpt-4o"), api_key=self.api_key)

    async def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        # Gemini doesn't accept a temperature keyword on invoke
        kwargs.pop("temperature")
        return await super().ainvoke(input, config=config, stop=stop, **kwargs)

    def _preprocess_messages(self, messages):
        from ..agent import SystemMessage, AutoContextMessage, AIMessage, ToolMessage, FunctionMessage
        output = []
        system_messages = []
        all_tools = {tool_obj['id']: tool_obj for message in messages for tool_obj in getattr(message, 'tool_calls', [])}
        for message in messages:
            if isinstance(message, (SystemMessage, AutoContextMessage)):
                system_messages.append(message.content)
            elif isinstance(message, AIMessage):
                tool_call = message.tool_calls[0]
                message.additional_kwargs["function_call"] = {
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call["args"]),
                }
                output.append(message)
            elif isinstance(message, ToolMessage):
                # tool_call_id = message.tool_call_id
                # tool_info = all_tools.get(tool_call_id)
                # if tool_info:
                #     message_args = {field_name: getattr(message, field_name) for field_name in message.model_fields.keys()}
                #     message_args["type"] = "function"
                #     message_args["name"] = tool_info.get("name")
                #     message = FunctionMessage(**message_args)
                output.append(message)
            else:
                output.append(message)
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, ChatGoogleGenerativeAIError):
            if any(('400 API key not valid' in arg for arg in error.args)):
                raise AuthenticationError("API key invalid.") from error
        raise ExecutionError(*error.args) from error
