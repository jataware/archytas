from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar

from langchain_core.messages import SystemMessage

from ..base_family import BaseModelFamily, ModelVersion

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage
    from ...agent import AgentResponse


class GeminiFamily(BaseModelFamily):
    """Model family for Google Gemini models."""

    VERSIONS: ClassVar[dict[str, ModelVersion]] = {
        "latest": ModelVersion(model_name="gemini-2.5-flash"),
        "2.5-flash": ModelVersion(model_name="gemini-2.5-flash", context_size=1_048_576),
        "2.5-pro": ModelVersion(model_name="gemini-2.5-pro", context_size=2_097_152),
        "2.0-flash": ModelVersion(model_name="gemini-2.0-flash", context_size=1_048_576),
    }
    DEFAULT_VERSION: ClassVar[str] = "latest"
    PROMPT_INSTRUCTIONS: ClassVar[str] = (
        "When passing strings to tools, you do not need to escape the values. "
        "They are already formatted as expected.\n"
    )

    def _model_supports_thinking(self, model_name: str) -> bool:
        """Check if the model supports thinking blocks (gemini-2.5+ and gemini-3+)."""
        match = re.search(r"(\d+)\.(\d+)", model_name)
        if not match:
            return False
        major, minor = int(match.group(1)), int(match.group(2))
        return major >= 3 or (major == 2 and minor >= 5)

    def get_create_model_kwargs(self, model_name: str) -> dict:
        """Return extra kwargs to pass to provider.create_chat_model()."""
        kwargs = {}
        if self._model_supports_thinking(model_name):
            kwargs["include_thoughts"] = True
        return kwargs

    @lru_cache()
    def context_size(self, model_name: str) -> int | None:
        if "flash" in model_name:
            return 1_048_576
        elif "pro" in model_name:
            if "2." in model_name:
                return 2_097_152
            else:
                return 1_048_576
        return None

    def supports_temperature(self, model_name: str) -> bool:
        return False

    def supports_thinking(self, model_name: str) -> bool:
        return self._model_supports_thinking(model_name)

    def preprocess_messages(self, messages: list["BaseMessage"]) -> list["BaseMessage"]:
        from langchain_core.messages import AIMessage
        from ...chat_history import AutoContextMessage

        output = []
        system_messages = []
        thinking_supported = False
        # Determine model name from context — check first system message for hints
        # The Model facade sets self._model_name before calling preprocess
        model_name = getattr(self, "_current_model_name", "gemini-2.5-flash")
        thinking_supported = self._model_supports_thinking(model_name)

        for message in messages:
            if isinstance(message, (SystemMessage, AutoContextMessage)):
                system_messages.append(message.content)
            elif thinking_supported and isinstance(message, AIMessage) and message.tool_calls:
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
                else:
                    new_content.append(block)
            if changed:
                return message.model_copy(update={"content": new_content})
        return message

    def process_result(self, response_message: "AIMessage") -> "AgentResponse":
        from ...agent import AgentResponse

        content = response_message.content
        tool_calls = response_message.tool_calls

        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            labeled_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                block_type = item.get("type")
                if block_type == "text" and len(item.get("text") or "") > 0:
                    labeled_parts.append(str(item["text"]))
            if labeled_parts:
                text = "\n".join(labeled_parts)

        return AgentResponse(text=text, tool_calls=tool_calls)
