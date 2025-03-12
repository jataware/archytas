import json
from typing import TYPE_CHECKING, TypeAlias, Callable

if TYPE_CHECKING:
    from .agent import Agent
    from langchain_core.messages import ToolMessage, AIMessage, BaseMessage


def default_summarizer(message: "ToolMessage", all_messages: "list[BaseMessage]", agent: "Agent"):
    message.artifact["summarized"] = True
