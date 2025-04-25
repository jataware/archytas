import asyncio
import copy
import functools
import hashlib
import inspect
import json
import logging
import uuid
from dataclasses import MISSING
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING,Callable, Collection, Optional, TypeVar, Generic, cast, Any, TypeAlias, Coroutine
from typing_extensions import Self

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage, AIMessage, ToolCall
from pydantic import Field

from .exceptions import AuthenticationError, ExecutionError, ModelError, ContextWindowExceededError
from .models.base import BaseArchytasModel
from .utils import ensure_async

from .exceptions import AuthenticationError
from .summarizers import (
    MessageSummarizerFunction, LoopSummarizerFunction, HistorySummarizerFunction, # BaseHistorySummarizer, # DefaultHistoryManager
    default_history_summarizer, default_loop_summarizer
)


if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


def tool_hash(tools: list[any]) -> str:
    import hashlib
    hash_value = hashlib.sha1()
    for tool in tools:
        hash_value.update(str(tool).encode())
    return hash_value.hexdigest()


class ContextMessage(SystemMessage):
    """Simple wrapper around a system message to facilitate message disambiguation."""

class AutoContextMessage(ContextMessage):
    """An automatically updating context message that remains towards the top of the message list."""

    default_content: str = Field(exclude=True)
    content_updater: Callable[[], str] = Field(exclude=True)

    def __init__(self, default_content: str, content_updater: Callable[[], str], model: Optional[BaseArchytasModel] = None, **kwargs):
        kwargs.update({"default_content": default_content, "content_updater": content_updater})
        super().__init__(content=default_content, **kwargs)
        self._token_count = None
        self._model = model

    @property
    def content_hash(self):
        return hashlib.sha1(self.text().encode()).hexdigest()

    @property
    def token_count(self):
        return self._token_count

    async def update_content(self):
        orig_hash = self.content_hash
        result = await ensure_async(self.content_updater())
        self.content = result
        if self.content_hash != orig_hash and self._model:
            print("Need to update count")
            self._token_count = await self._model.get_num_tokens_from_messages([self])


MessageType = TypeVar("MessageType", bound=BaseMessage)

@dataclass
class AgentResponse:
    text: str
    tool_calls: list[ToolCall]


@dataclass
class MessageRecord(Generic[MessageType]):
    message: MessageType
    uuid: str = dataclass_field(default_factory=lambda: uuid.uuid4().hex)
    token_count: Optional[int] = dataclass_field(default=None)
    metadata: dict[str, Any] = dataclass_field(default_factory=lambda: {})
    react_loop_id: Optional[int] = dataclass_field(default=None)


@dataclass
class SummaryRecord(MessageRecord[SystemMessage]):
    message: SystemMessage
    summarized_messages: set[str] = dataclass_field(default_factory=lambda: set())


RecordType: TypeAlias = MessageRecord | SummaryRecord


@dataclass
class OutputMessage:
    content: str | list[str | dict]


@dataclass
class OutboundModel:
    provider: str
    model_name: str
    context_window: Optional[int]


@dataclass
class OutboundChatHistory:
    records: list[RecordType]
    system_message: Optional[MessageRecord[SystemMessage]]
    tool_token_usage_estimate: Optional[int]
    model: OutboundModel
    token_estimate: Optional[int]
    message_token_count: Optional[int]
    summary_token_count: Optional[int]
    overhead_token_count: Optional[int]
    summarization_threshold: Optional[int]


class ChatHistory:
    base_tokens: int
    current_loop_id: int|None
    raw_records: list[MessageRecord]
    summaries: list[SummaryRecord]
    system_message: Optional[MessageRecord[SystemMessage]]
    model: Optional[BaseArchytasModel]
    loop_summarizer: Optional[callable]
    history_summarizer: Optional[callable]
    summarization_threshold: int
    tool_token_estimate: int
    _tool_hash: str
    _token_estimate: int|None
    history_summarization_task: Optional[asyncio.Task]

    _current_context_id: int
    auto_context_message: Optional[AutoContextMessage]
    auto_update_context: bool

    def __init__(
        self,
        messages: Optional[Collection[BaseMessage]] = None,
        model: Optional[BaseArchytasModel] = None,
        loop_summarizer: Optional[callable] = default_loop_summarizer,
        history_summarizer: Optional[callable] = default_history_summarizer,
    ):
        self.base_tokens = 0
        self.current_loop_id = None
        self.raw_records = []
        self.summaries = []
        self.history_summarization_task = None
        self.system_message = None
        self.model = model
        self.tool_token_estimate = 0
        self._tool_hash = ""
        self._token_estimate = None
        self.loop_summarizer = loop_summarizer
        self.history_summarizer = history_summarizer
        if messages:
            self.raw_records.extend((
                MessageRecord(message=message) for message in messages
            ))

        if self.model is not None:
            self.summarization_threshold = self.model.default_summarization_threshold
        else:
            self.summarization_threshold = -1

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
        self.system_message = MessageRecord(
            message=system_message,
        )

    @property
    def auto_context_message_token_estimate(self) -> int:
        if self.auto_context_message:
            return self.auto_context_message.token_count
        else:
            return 0

    async def update_auto_context(self):
        coroutines: list[Coroutine] = []
        if self.auto_update_context:
            if self.auto_context_message:
                coroutines.append(self.auto_context_message.update_content())
            for message in self.raw_records:
                if isinstance(message.message, AutoContextMessage):
                    coroutines.append(message.message.update_content())
            if coroutines:
                await asyncio.gather(*coroutines)

    @property
    def token_overhead(self):
        return sum((value for value in (
            self.base_tokens,
            self.tool_token_estimate,
            self.auto_context_message_token_estimate,
        ) if isinstance(value, int)))

    def get_tool_caller(self, tool_call_id: str) -> tuple[MessageRecord, ToolCall]:
        calling_record, tool_call = next(
            (
                (record, tc) for record
                in self.raw_records
                for tc in getattr(record.message, "tool_calls", [])
                if tc.get('id', MISSING) == tool_call_id
            ),
            (None, None),
        )
        return (cast(MessageRecord, calling_record), tool_call)

    async def needs_summarization(self, model: Optional[BaseArchytasModel]=None):
        if model is not None:
            threshold = model.summarization_threshold
        elif self.model is not None:
            threshold = self.model.summarization_threshold
            model = self.model
        else:
            return False
        token_estimate = await self.token_estimate(model=model)
        return token_estimate > threshold

    async def summarize_loop(
        self,
        loop_records: Optional[list[MessageRecord]] = None,
        loop_record_id: Optional[int] = None,
        agent: "Agent" = None,
        force_update: bool = False
    ):
        if not loop_records:
            if not loop_record_id:
                loop_record_id: int = self.current_loop_id
            if not loop_record_id:
                logger.warning("Unable to determine intended loop to summarize. Neither loop_records nor loop_record_id provided, and no current .")
                return []
            loop_records: list[MessageRecord] = [
                record for record
                in await self.all_records(auto_update_context=False)
                if record.react_loop_id == loop_record_id
            ]
        if loop_records and self.loop_summarizer:
            logger.debug("Initiating loop summarization")
            await self.loop_summarizer(loop_records, self, agent, model=agent.model, force_update=force_update)
            logger.debug("Loop summarization completed")

    async def summarize_history(
            self,
            agent: "Agent" = None,
            token_threshold: Optional[int] = None,
            force_update: bool = False,
            in_loop: bool = False,
        ):

        # Return running summarization task if one is running to prevent duplicate runs
        if self.history_summarization_task is not None:
            return self.history_summarization_task

        if self.history_summarizer is None:
            return

        if token_threshold is None:
            token_threshold = agent.model.summarization_threshold

        def task_callback(task: asyncio.Task):
            print(f"Summarization task {task} completed.")
            self.history_summarization_task = None

        threshold = token_threshold or self.summarization_threshold
        from .summarizers import get_records_up_to_threshold, get_records_up_to_loop
        if in_loop:
            recordset = await get_records_up_to_loop(
                chat_history=self,
                model=agent.model,
            )
        else:
            recordset = await get_records_up_to_threshold(
                chat_history=self,
                model=agent.model,
                token_threshold=threshold,
            )

        # If not recordset, nothing to summarize, nothing left to do.
        if not recordset:
            return

        task = asyncio.create_task(
            self.history_summarizer(
                chat_history=self,
                agent=agent,
                recordset=recordset,
            )
        )
        task.add_done_callback(task_callback)
        self.history_summarization_task = task
        return task

    async def token_estimate(
        self,
        model: BaseArchytasModel,
        messages: Optional[list[RecordType]|list[MessageType]]=None,
        force_update: bool = False,
        tools: Optional[dict]=None,
        use_cache: bool=True,
    ) -> Optional[int]:
        """
        Does not update/process any AutoContextMessages. All messages should be updated prior to calling this method.
        By default, will calculate the summarized chat history. If you need the full history, pass in an appropriate
        list of messages to the messages argument, e.g. `self.token_estimate(messages=messages.raw_messages)`.
        """

        if messages is None:
            messages = await self.records(auto_update_context=False)

        message_records: list[RecordType] = []
        base_messages: list[BaseMessage] = []
        tool_content: Optional[str] = None

        if self.tool_token_estimate is None or (self._tool_hash and self._tool_hash != tool_hash(tools)):
            if tools:
                lc_tools = model.convert_tools(tuple(tools.items()))
            elif model.lc_tools:
                lc_tools = model.lc_tools
            if lc_tools and (not self.tool_token_estimate or self._tool_hash != tool_hash(lc_tools)):
                self._tool_hash = tool_hash(lc_tools)
                tool_content = {
                    tool.name: {
                        "name": tool.name,
                        "description": tool.description,
                        "arg_schema": tool.args_schema.model_json_schema(),
                    }
                    for tool in lc_tools
                }
                self.tool_token_estimate = await model.get_num_tokens_from_messages([HumanMessage(content=json.dumps(tool_content))])

        for item in messages:
            if isinstance(item, MessageRecord):
                message_records.append(item)
            elif isinstance(item, BaseMessage):
                base_messages.append(item)
            else:
                raise ValueError(f"Unable to handle message of type {item.__class__.__name__}")

        if message_records and base_messages:
                raise ValueError("It is invalid to mix ChatHistory MessageRecords and LangChain BaseMessages.")

        if message_records:
            token_estimate_sum = self.token_overhead
            for record in message_records:
                if force_update or record.token_count is None:
                    content = record.message.content
                    message = HumanMessage(content=content)
                    try:
                        message_token_est = await model.get_num_tokens_from_messages([message])
                    except NotImplementedError:
                        message_token_est = 0
                    record.token_count = message_token_est
                token_estimate_sum += record.token_count
            self._token_estimate = token_estimate_sum
            return token_estimate_sum

        if base_messages:
            try:
                token_estimate = await model.get_num_tokens_from_messages([base_messages])
            except NotImplementedError:
                token_estimate = 0
            token_estimate_sum = self.base_tokens + self.tool_token_estimate + token_estimate
            self._token_estimate = token_estimate_sum
            return token_estimate_sum

        self._token_estimate = None
        return None

    async def records(self, auto_update_context: bool = True) -> list[RecordType]:
        """
        Messages
        """
        records: list[RecordType] = []
        summarized_messages: set[str] = set()

        if self.system_message:
            records.append(self.system_message)

        if auto_update_context:
            await self.update_auto_context()

        if self.auto_context_message:
            records.append(
                MessageRecord(message=self.auto_context_message))

        for summary_record in self.summaries:
            records.append(summary_record)
            summarized_messages.update(summary_record.summarized_messages)

        for message_record in self.raw_records:
            if message_record.uuid not in summarized_messages:
                records.append(message_record)
        return records

    async def messages(self, auto_update_context: bool = True) -> list[MessageType]:
        records = await self.records(auto_update_context=auto_update_context)
        return [cast(MessageType, record.message) for record in records]

    async def all_records(self, auto_update_context: bool = True) -> list[RecordType]:
        messages: list[RecordType] = []
        if self.system_message:
            messages.append(self.system_message)

        if auto_update_context:
            await self.update_auto_context()

        if self.auto_context_message:
            messages.append(
                MessageRecord(message=self.auto_context_message))
        messages.extend(self.raw_records)
        return messages

    async def all_messages(self, auto_update_context: bool = True) -> list[MessageType]:
        records = await self.all_records(auto_update_context=auto_update_context)
        return [cast(MessageType, record.message) for record in records]

    def add_message(
        self,
        message: MessageType,
        token_count: Optional[int] = None,
    ):
        """Appends a message to the message list."""
        record = MessageRecord(message=message, token_count=token_count, react_loop_id=self.current_loop_id)
        self.raw_records.append(record)
        return record
