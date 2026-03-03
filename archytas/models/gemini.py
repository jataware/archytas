import json
import re
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI, ChatGoogleGenerativeAIError

from .base import BaseArchytasModel
from ..message_schemas import ToolUseRequest
from ..exceptions import AuthenticationError, ExecutionError, ContextWindowExceededError

if TYPE_CHECKING:
    from ..agent import SystemMessage, AutoContextMessage, AIMessage, ToolMessage, FunctionMessage


class GeminiModel(BaseArchytasModel):
    DEFAULT_MODEL = "gemini-2.5-flash"
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

    def _model_supports_thinking(self, model_name: str | None = None) -> bool:
        """Check if the model supports thinking blocks (gemini-2.5+ and gemini-3+)."""
        if model_name is None:
            model_name = self.config.model_name or self.DEFAULT_MODEL
        # Match gemini-2.X where X >= 5, or gemini-3+
        match = re.search(r'(\d+)\.(\d+)', model_name)
        if not match:
            return False
        major, minor = int(match.group(1)), int(match.group(2))
        return major >= 3 or (major == 2 and minor >= 5)

    def _model_supports_medium_thinking_level(self, model_name: str | None = None) -> bool:
        """
        Check if the model supports a MEDIUM thought level instead of HIGH or LOW
        (gemini-3.1-pro-preview).
        """
        if model_name is None:
            model_name = self.config.model_name or self.DEFAULT_MODEL
        # Match gemini-2.X where X >= 5, or gemini-3+
        match = re.search(r'(\d+)\.(\d+)', model_name)
        if not match:
            return False
        major, minor = int(match.group(1)), int(match.group(2))
        return major >= 3 and minor == 1

    def initialize_model(self, **kwargs):
        model_name = self.config.model_name or self.DEFAULT_MODEL
        model_kwargs = dict(
            model=model_name,
            api_key=self.api_key,
        )
        if self._model_supports_thinking(model_name):
            model_kwargs["include_thoughts"] = True
        if self._model_supports_medium_thinking_level(model_name):
            # model_kwargs["thinking_level"] = "medium"
            # model_kwargs["thinking_budget_token_limit"] = 1000
            pass
        return ChatGoogleGenerativeAI(**model_kwargs)

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
            # gemini 2.0/2.5 have 2M context, others do not
            if "2." in model_name:
                return 2_097_152
            else:
                return 1_048_576

    def _preprocess_messages(self, messages):
        from langchain_core.messages import AIMessage
        from ..agent import SystemMessage, AutoContextMessage
        output = []
        system_messages = []
        thinking_supported = self._model_supports_thinking()
        for message in messages:
            if isinstance(message, (SystemMessage, AutoContextMessage)):
                system_messages.append(message.content)
            elif thinking_supported and isinstance(message, AIMessage) and message.tool_calls:
                # langchain-google-genai's _parse_chat_history drops text content
                # from AIMessages that have tool_calls, but preserves thinking/reasoning
                # blocks. Convert text content to thinking blocks so it survives the
                # round-trip through langchain's Gemini message conversion.
                output.append(self._wrap_text_as_thinking(message))
            else:
                output.append(message)
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output

    @staticmethod
    def _wrap_text_as_thinking(message: "AIMessage") -> "AIMessage":
        """Convert text content in an AIMessage to thinking blocks.

        langchain-google-genai preserves thinking/reasoning blocks but drops
        regular text from AIMessages with tool_calls. This converts text to
        thinking blocks so the model's reasoning survives the round-trip.
        """
        content = message.content
        if isinstance(content, str):
            if not content:
                return message
            new_content = [{"type": "thinking", "thinking": content}]
            return message.model_copy(update={"content": new_content})
        elif isinstance(content, list):
            new_content = []
            changed = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        new_content.append({"type": "thinking", "thinking": text})
                        changed = True
                    # Drop empty text blocks
                else:
                    # Preserve thinking, reasoning, and any other block types as-is
                    new_content.append(block)
            if changed:
                return message.model_copy(update={"content": new_content})
        return message

    def handle_invoke_error(self, error: BaseException):
        if isinstance(error, ChatGoogleGenerativeAIError):
            if any(('400 API key not valid' in arg for arg in error.args)):
                raise AuthenticationError("API key invalid.") from error
            elif any(('exceeds the maximum number of tokens allowed' in arg for arg in error.args)):
                raise ContextWindowExceededError("Context window maximum tokens exceeded.") from error
        raise ExecutionError(*error.args) from error

    def process_result(self, response_message: "AIMessage") -> "AgentResponse":
        from ..agent import AgentResponse
        content = response_message.content
        tool_calls = response_message.tool_calls

        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Extract all content blocks with source labels
            labeled_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                block_type = item.get("type")
                if block_type == "text" and len(item.get("text") or "") > 0:
                    labeled_parts.append(str(item['text']))
                # thinking blocks are much too verbose for a beaker user experience
                elif block_type == "thinking" and item.get("thinking"):
                    # labeled_parts.append(f"[thinking] {item['thinking']}")
                    pass
                elif block_type == "reasoning" and item.get("reasoning"):
                    # labeled_parts.append(f"[reasoning] {item['reasoning']}")
                    pass
            if labeled_parts:
                text = "\n".join(labeled_parts)

        # tool call blocks:
        #
        # tool_thoughts = [
        #     tool_call["args"].pop("thought", f"Calling tool: `{tool_call['name']}`.")
        #     for tool_call in tool_calls
        #     if tool_call['name'] not in ['ask_user', 'run_code']
        # ]
        # if not text:
        #     if tool_calls:
        #         text = "\n".join(tool_thoughts)
        #     else:
        #         raise ValueError("Response from LLM does not include any content or tool calls.")

        return AgentResponse(text=text, tool_calls=tool_calls)
