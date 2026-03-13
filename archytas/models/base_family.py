from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, BaseMessage
    from ..agent import AgentResponse


@dataclass
class ModelVersion:
    """A specific configuration point within a model family."""

    model_name: str
    context_size: int | None = None
    max_tokens: int | None = None
    extra: dict = field(default_factory=dict)

    def handle_version_error(self, error: Exception) -> None:
        """Version-specific error mapping. First in chain.
        Default: re-raise (fall through to family).
        """
        raise error


class BaseModelFamily(ABC):
    """Owns model-behavior logic shared across a family of models."""

    VERSIONS: ClassVar[dict[str, ModelVersion]]
    DEFAULT_VERSION: ClassVar[str] = "latest"
    DEFAULT_PROVIDER: ClassVar[type | None] = None
    PROMPT_INSTRUCTIONS: ClassVar[str] = ""

    @abstractmethod
    def context_size(self, model_name: str) -> int | None:
        """Return context window size for a specific model name."""
        ...

    def preprocess_messages(self, messages: list["BaseMessage"]) -> list["BaseMessage"]:
        """Transform messages before sending to LLM. Default: no-op."""
        return messages

    def process_result(self, response_message: "AIMessage") -> "AgentResponse":
        """Post-process LLM response. Default: extract text content."""
        from ..agent import AgentResponse
        content = response_message.content
        tool_calls = response_message.tool_calls
        tool_thoughts = [
            tool_call["args"].pop("thought", f"Calling tool '{tool_call['name']}'")
            for tool_call in tool_calls
        ]

        match content:
            case list():
                try:
                    text = "\n".join(
                        item['text'] for item in content if item.get('type', None) == "text"
                    )
                except AttributeError:
                    thinking, summary = content
                    text = summary
            case "":
                if tool_calls:
                    text = "\n".join(tool_thoughts)
                else:
                    raise ValueError(
                        "Response from LLM does not include any content or tool calls. "
                        "This shouldn't happen."
                    )
            case str():
                text = content
            case _:
                raise ValueError(
                    "Response from LLM does not match expected format."
                )

        if text == "":
            text = "Thinking..."
        return AgentResponse(text=text, tool_calls=tool_calls)

    def handle_model_error(self, error: Exception) -> None:
        """Map model-level errors (e.g. context window exceeded).
        Called after version, before provider. Default: re-raise.
        """
        raise error

    def supports_temperature(self, model_name: str) -> bool:
        return True

    def supports_thinking(self, model_name: str) -> bool:
        return False

    def rectify_result(self, response_message: "AIMessage") -> "AIMessage":
        """Post-process raw AIMessage before storing in history. Default: no-op."""
        return response_message

    # TODO: Token counting strategy needs review — currently split across provider
    # and model concerns. Revisit whether this belongs on family, provider, or both.
    async def get_num_tokens_from_messages(
        self,
        chat_model: "Any",
        messages: "list[BaseMessage]",
        tools: "Any | None" = None,
    ) -> int:
        """Estimate token count for messages. Default delegates to LangChain model."""
        try:
            return chat_model.get_num_tokens_from_messages(messages=messages, tools=tools)
        except Exception:
            return 0
