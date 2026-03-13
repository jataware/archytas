from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from langchain_core.messages import SystemMessage

from ..base_family import BaseModelFamily, ModelVersion

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class LlamaFamily(BaseModelFamily):
    """Model family for Meta Llama models (via Groq, Ollama, Bedrock, OpenRouter, etc.)."""

    VERSIONS: ClassVar[dict[str, ModelVersion]] = {
        "latest": ModelVersion(model_name="llama3-8b-8192"),
        "llama-3.3-70b": ModelVersion(model_name="llama-3.3-70b-versatile", context_size=128_000),
        "llama-3-8b": ModelVersion(model_name="llama3-8b-8192", context_size=8_192),
    }
    DEFAULT_VERSION: ClassVar[str] = "latest"
    PROMPT_INSTRUCTIONS: ClassVar[str] = (
        "When generating JSON, remember to not wrap strings in triple quotes such as "
        "''' or \"\"\". If you want to add newlines to the JSON text, use `\\n` to add newlines.\n"
        "Ensure all generated JSON is valid and would pass a JSON validator.\n"
    )

    def context_size(self, model_name: str) -> int | None:
        # Context size varies by deployment; return None to let providers handle it
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
