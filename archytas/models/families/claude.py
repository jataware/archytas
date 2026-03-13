from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar

from langchain_core.messages import SystemMessage

from ..base_family import BaseModelFamily, ModelVersion

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage
    from ...agent import AgentResponse


class ClaudeFamily(BaseModelFamily):
    """Model family for Anthropic Claude models."""

    VERSIONS: ClassVar[dict[str, ModelVersion]] = {
        "latest": ModelVersion(model_name="claude-sonnet-4-6-20250514"),
        "sonnet-4.6": ModelVersion(model_name="claude-sonnet-4-6-20250514", context_size=200_000),
        "opus-4.6": ModelVersion(model_name="claude-opus-4-6", context_size=200_000),
        "haiku-3.5": ModelVersion(model_name="claude-3-5-haiku-latest", context_size=200_000),
        "sonnet-3.5": ModelVersion(model_name="claude-3-5-sonnet-latest", context_size=200_000),
        "claude-3": ModelVersion(model_name="claude-3-sonnet-20240229", context_size=200_000),
    }
    DEFAULT_VERSION: ClassVar[str] = "latest"
    PROMPT_INSTRUCTIONS: ClassVar[str] = ""

    @lru_cache()
    def context_size(self, model_name: str) -> int | None:
        if model_name and ("haiku" in model_name or "sonnet" in model_name or "opus" in model_name):
            return 200_000
        return None

    def preprocess_messages(self, messages: list["BaseMessage"]) -> list["BaseMessage"]:
        from ...chat_history import ContextMessage, AutoContextMessage

        output = []
        system_messages = []
        for message in messages:
            match message:
                case SystemMessage() | ContextMessage() | AutoContextMessage():
                    system_messages.append(message.content)
                case _:
                    output.append(message)
        output.insert(0, SystemMessage(content="\n".join(system_messages)))
        return output
