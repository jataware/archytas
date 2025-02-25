import copy
import inspect
import json
import logging
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Collection, Optional, TypeVar, Generic, cast, Any
from typing_extensions import Self

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage, AIMessage, ToolCall
from pydantic import Field

from .exceptions import AuthenticationError, ExecutionError, ModelError, ContextWindowExceededError
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
        kwargs.update({"default_content": default_content, "content_updater": content_updater})
        super().__init__(content=default_content, **kwargs)

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

_archytas_types: dict[str, type] = {}
T = TypeVar("T", bound=BaseMessage)

class ArchytasMessage(BaseMessage, Generic[T]):
    token_estimate: Optional[int]

    @classmethod
    def wrap(
        cls,
        base_instance: T,
        model: Optional[BaseArchytasModel]=None,
    ) -> "ArchytasMessage[T]":
        self: ArchytasMessage[T] = cast(ArchytasMessage[T], copy.deepcopy(base_instance))
        self.__class__ = cls
        self.token_estimate = None
        if model:
            self.update_token_estimate(model)
        return self

    def update_token_estimate(self, model: BaseArchytasModel):
        contents = self.model_dump_json()
        try:
            token_estimate = model.model.get_num_tokens(contents)
        except:
            raise
        self.token_estimate = token_estimate


def extend_message(base: T|ArchytasMessage[T], model: Optional[BaseArchytasModel]=None) -> ArchytasMessage[T]:
    if isinstance(base, ArchytasMessage):
        return base
    base_class = base.__class__
    name = f"Archytas{base_class.__name__}"
    if name in _archytas_types:
        return _archytas_types[name](base)
    NewType: type[ArchytasMessage[T]] = type(name, (ArchytasMessage, base_class), dict(base_class.__dict__))
    result = NewType.wrap(base, model=model)
    return result


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[ToolCall]


class ChatHistory:
    raw_messages: list[ArchytasMessage]
    system_message: Optional[ArchytasMessage[SystemMessage]]
    model: Optional[BaseArchytasModel]
    context_manager: "BaseContextManager"

    _current_context_id: int
    auto_context_message: Optional[AutoContextMessage]
    auto_update_context: bool

    def __init__(
        self,
        messages: Optional[Collection[BaseMessage]] = None,
        model: Optional[BaseArchytasModel] = None,
        context_manager: "Optional[BaseContextManager]" = None,
    ):
        self.raw_messages = []
        self.system_message = None
        self.model = model
        if context_manager is None:
            context_manager = BaseContextManager()
        self.context_manager = context_manager
        if messages:
            self.raw_messages.extend((extend_message(message) for message in messages))

        # use to generate unique ids for context messages
        self._current_context_id = 0

        # Initialize the auto_context_message to empty
        self.auto_context_message = None
        self.auto_update_context = False

    def set_system_message(
        self,
        system_message: SystemMessage|str
    ):
        if isinstance(system_message, str):
            system_message = SystemMessage(system_message)
        self.system_message = extend_message(system_message)

    async def token_estimate(
        self,
        model: BaseArchytasModel,
        messages: Optional[list[ArchytasMessage]]=None,
        tools: Optional[dict]=None,
        raw: bool=False,
        auto_update_contexts: bool=False
    ):
        if messages is None:
            if raw:
                messages = await self.all_messages()
            else:
                messages = await self.messages()
        return await model.token_estimate(messages, agent_tools=tools)

    async def messages(self) -> list[ArchytasMessage]:
        """
        Messages
        """
        messages: list[ArchytasMessage[Any]] = []
        if self.system_message:
            messages.append(self.system_message)
        if self.auto_context_message:
            if self.auto_update_context:
                await self.auto_context_message.update_content()
            messages.append(extend_message(self.auto_context_message))
        messages.extend(self.raw_messages)
        return messages

    async def all_messages(self) -> list[ArchytasMessage]:
        messages: list[ArchytasMessage[Any]] = []
        if self.system_message:
            messages.append(self.system_message)
        if self.auto_context_message:
            if self.auto_update_context:
                await self.auto_context_message.update_content()
            messages.append(extend_message(self.auto_context_message))
        messages.extend(self.raw_messages)
        return messages

    def add_message(
        self,
        message: BaseMessage|ArchytasMessage[T],
        model: Optional[BaseArchytasModel] = None,
    ):
        """Appends a message to the message list."""
        self.raw_messages.append(extend_message(message, model))

class BaseContextManager(ABC):
    def needs_management(self, context) -> bool:
        return False

class PassiveContextManager(BaseContextManager):
    "Doesn't do anything to manage the context. Keeps all messages and history as passed in."
    pass

class RecentMessages(BaseContextManager):
    pass
