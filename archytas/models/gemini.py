import json
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI, ChatGoogleGenerativeAIError

from .base import BaseArchytasModel
from ..message_schemas import ToolUseRequest
from ..exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

if TYPE_CHECKING:
    from ..agent import SystemMessage, AutoContextMessage, AIMessage, ToolMessage, FunctionMessage


class GeminiModel(BaseArchytasModel):
    DEFAULT_MODEL = "gemini-2.0-flash"
    api_key: str


    MODEL_PROMPT_INSTRUCTIONS = """\
When passing strings to tools, you do not need to escape the values. They are already formatted as expected.
"""

    def auth(self, **kwargs) -> None:
        auth_token = None
        if 'api_key' in kwargs:
            auth_token = kwargs['api_key']
        else:
            auth_token = self.config.api_key
        if auth_token:
            self.api_key = auth_token
        else:
            raise AuthenticationError("Gemini API key not provided.")

    def initialize_model(self, **kwargs):
        return ChatGoogleGenerativeAI(
            model=self.config.model_name or self.DEFAULT_MODEL,
            api_key=self.api_key
        )

    async def ainvoke(self, input, *, config=None, stop=None, **kwargs):
        # Gemini doesn't accept a temperature keyword on invoke
        kwargs.pop("temperature")
        return await super().ainvoke(input, config=config, stop=stop, **kwargs)

    @lru_cache()
    def contextsize(self, model_name = None):
        if model_name is None:
            model_name = self.model_name
        if "flash" in model_name:
            return 1_048_576
        elif "pro" in model_name:
            return 2_097_152

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
            elif any(('exceeds the maximum number of tokens allowed' in arg for arg in error.args)):
                raise ContextWindowExceededError("Context window maximum tokens exceeded.") from error
        raise ExecutionError(*error.args) from error

    def process_result(self, response_message: "AIMessage"):
        response = super().process_result(response_message)
        return response
