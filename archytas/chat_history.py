import inspect
import logging
from dataclasses import dataclass
from typing import Callable, Collection, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage, AIMessage, ToolCall
from pydantic import Field

from .exceptions import AuthenticationError, ExecutionError, ModelError
from .models.base import BaseArchytasModel

from .exceptions import AuthenticationError

logger = logging.getLogger(__name__)


class ContextMessage(SystemMessage):
    """Simple wrapper around a system message to facilitate message disambiguation."""


class AutoContextMessage(ContextMessage):
    """An automatically updating context message that remains towards the top of the message list."""

    default_content: str = Field(exclude=True)
    content_updater: Callable[[], str] = Field(exclude=True)

    def __init__(self, default_content: str, content_updater: Callable[[], str], **kwargs):
        super().__init__(content=default_content, default_content=default_content, content_updater=content_updater, **kwargs)

    async def update_content(self):
        if inspect.iscoroutinefunction(self.content_updater):
            result = await self.content_updater()
        else:
            result = self.content_updater()
        self.content = result


class AutoSummarizedToolMessage(ToolMessage):
    """A message that replaces its full tool output with a summary after the ReAct loop is complete."""

    summary: str = ""
    summarized: bool = False

    async def update_content(self):
        if not self.summarized:
            self.content=self.summary
            self.summarized = True


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[ToolCall]


class ChatHistory:
    raw_messages: list[BaseMessage]
    system_message: SystemMessage

    _current_context_id: int
    auto_context_message: Optional[ContextMessage]
    auto_update_context: bool

    def __init__(self, messages: Optional[Collection[BaseMessage]] = None):
        self.raw_messages = []
        self.system_message = None
        if messages:
            self.raw_messages.extend(messages)

        # use to generate unique ids for context messages
        self._current_context_id = 0

        # Initialize the auto_context_message to empty
        self.auto_context_message = None
        self.auto_update_context = False

    async def token_estimate(
        self,
        model: BaseArchytasModel,
        messages: Optional[list[BaseMessage]]=None,
        tools: Optional[dict]=None,
        raw: bool=False,
        auto_update_contexts: bool=False
    ):
        if auto_update_contexts or self.auto_update_context:
            await
        if messages is None:
            if raw:
                messages = await self.all_messages()
            else:
                messages = await self.messages()
        return await model.token_estimate(messages, agent_tools=tools)

    async def messages(self) -> list[BaseMessage]:
        messages = [self.system_message]
        if self.auto_context_message:
            if self.auto_update_context:
                await self.auto_context_message.update_content()
            messages.append(self.auto_context_message)
        messages.extend(self.raw_messages)
        return messages

    async def all_messages(self) -> list[BaseMessage]:
        messages = [self.system_message]
        if self.auto_context_message:
            if self.auto_update_context:
                await self.auto_context_message.update_content()
            messages.append(self.auto_context_message)
        messages.extend(self.raw_messages)
        return messages

    def add_message(self, message: BaseMessage):
        """Appends a message to the message list."""
        self.raw_messages.append(message)
