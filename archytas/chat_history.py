import copy
import inspect
import json
import logging
import uuid
from dataclasses import MISSING
from abc import ABC, abstractmethod
from asyncio.tasks import gather
from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING,Callable, Collection, Optional, TypeVar, Generic, cast, Any, TypeAlias, Coroutine
from typing_extensions import Self

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage, AIMessage, ToolCall
from pydantic import Field

from .exceptions import AuthenticationError, ExecutionError, ModelError, ContextWindowExceededError
from .models.base import BaseArchytasModel

from .exceptions import AuthenticationError
from .summarizers import MessageSummarizerFunction, LoopSummarizerFunction, HistorySummarizerFunction


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

    def __init__(self, default_content: str, content_updater: Callable[[], str], **kwargs):
        kwargs.update({"default_content": default_content, "content_updater": content_updater})
        super().__init__(content=default_content, **kwargs)

    async def update_content(self):
        if inspect.iscoroutinefunction(self.content_updater):
            result = await self.content_updater()
        else:
            result = self.content_updater()
        self.content = result


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


@dataclass
class SummaryRecord(MessageRecord[SystemMessage]):
    message: SystemMessage
    summarized_messages: set[str] = dataclass_field(default_factory=lambda: set())


RecordType: TypeAlias = MessageRecord | SummaryRecord


class ChatHistory:
    base_tokens: int
    raw_records: list[MessageRecord]
    summaries: list[SummaryRecord]
    system_message: Optional[MessageRecord[SystemMessage]]
    model: Optional[BaseArchytasModel]
    context_manager: "BaseContextManager"
    summarization_threshold: int
    tool_token_estimate: int
    _tool_hash: str

    _current_context_id: int
    auto_context_message: Optional[AutoContextMessage]
    auto_update_context: bool

    def __init__(
        self,
        messages: Optional[Collection[BaseMessage]] = None,
        model: Optional[BaseArchytasModel] = None,
        context_manager: "Optional[BaseContextManager]" = None,
    ):
        self.base_tokens = 0
        self.raw_records = []
        self.summaries = []
        self.system_message = None
        self.model = model
        self.tool_token_estimate = 0
        self._tool_hash = ""
        if context_manager is None:
            context_manager = SummarizeOldestMessages()
        self.context_manager = context_manager
        if messages:
            self.raw_records.extend((
                MessageRecord(message=message) for message in messages
            ))

        if self.model is not None:
            self.summarization_threshold = self.model.default_summarization_threshold
        else:
            self.summarization_threshold = -1

        self.summarization_threshold = 3800

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

    def get_tool_caller(self, tool_call_id: str) -> tuple[MessageRecord, ToolCall]:
        calling_record, tool_call = next(
            (
                (record, tc) for record
                in self.raw_records
                for tc in getattr(record.message, "tool_calls", [])
                if tc.get('id', MISSING) == tool_call_id
            ),
            None
        )
        return (cast(MessageRecord, calling_record), tool_call)

    async def summarize_messages(self, agent: "Agent" = None, force_update: bool = False):
        for record in self.raw_records:
            if isinstance(record.message, ToolMessage) and (artifact := getattr(record.message, "artifact", None)):
                message = record.message
                tool_fn = artifact.get("tool", None)
                summarized: bool = artifact.get("summarized", False)
                summarizer: MessageSummarizerFunction = getattr(tool_fn, "summarizer", None)
                if summarizer and (force_update or not summarized):
                    summarizer(message, self, agent)

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
            messages = await self.records()

        message_records: list[RecordType] = []
        base_messages: list[BaseMessage] = []
        tool_content: Optional[str] = None

        # if self.tool_token_estimate is None or (self._tool_hash and self._tool_hash != tool_hash(tools)):
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
            self.tool_token_estimate = model._model.get_num_tokens_from_messages([HumanMessage(content=json.dumps(tool_content))])

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
            sum = self.base_tokens + self.tool_token_estimate
            for record in message_records:
                if force_update or record.token_count is None:
                    content = record.message.content
                    message = HumanMessage(content=content)
                    message_token_est = model._model.get_num_tokens_from_messages([message])
                    record.token_count = message_token_est
                    print(f"``` = {message_token_est}\n{message}\n```")
                sum += record.token_count
            return sum

        if base_messages:
            token_estimate = model._model.get_num_tokens_from_messages([base_messages])
            return self.base_tokens + self.tool_token_estimate + token_estimate

        return None

    async def records(self) -> list[RecordType]:
        """
        Messages
        """
        records: list[RecordType] = []
        summarized_messages: set[str] = set()
        seen_unsummarized_message = False

        if self.system_message:
            records.append(self.system_message)

        for summary_record in self.summaries:
            records.append(summary_record)
            if summary_record.summarized_messages.intersection(summarized_messages):
                # TODO: Determine if this is needed
                raise ValueError("Message is included in multiple summaries")
            else:
                summarized_messages.update(summary_record.summarized_messages)
        for message_record in self.raw_records:
            if message_record.uuid in summarized_messages:
                if seen_unsummarized_message:
                    # TODO: Determine if need. If not, how do we enforce order, since we process all summaries before raw messages?
                    raise ValueError("Summarized message found after unsummarized message. All summaries should be at the beginning of message list")
                else:
                    continue
            else:
                seen_unsummarized_message = True
                records.append(message_record)
        return records

    async def messages(self) -> list[MessageType]:
        records = await self.records()
        return [cast(MessageType, record.message) for record in records]

    async def all_records(self) -> list[RecordType]:
        messages: list[RecordType] = []
        if self.system_message:
            messages.append(self.system_message)
        # if self.auto_context_message:
        if self.auto_update_context:
            coroutines: list[Coroutine] = []
            for message in self.raw_records:
                if isinstance(message.message, AutoContextMessage):
                    coroutines.append(message.message.update_content())
            if coroutines:
                await gather(*coroutines)
        messages.append(
            MessageRecord(message=self.auto_context_message))
        messages.extend(self.raw_records)
        return messages

    async def all_messages(self) -> list[MessageType]:
        records = await self.all_records()
        return [cast(MessageType, record.message) for record in records]

    def add_message(
        self,
        message: MessageType,
        model: Optional[BaseArchytasModel] = None,
        token_count: Optional[int] = None,
    ):
        """Appends a message to the message list."""
        self.raw_records.append(MessageRecord(message=message, token_count=token_count))

    async def summarize(
        self,
        model: BaseArchytasModel,
        token_threshold: Optional[int] = None,
    ):
        if token_threshold is None:
            token_threshold = model.default_summarization_threshold
        if self.context_manager:
            return await self.context_manager.summarize(self, model=model, token_threshold=token_threshold)


class BaseContextManager(ABC):
    @abstractmethod
    async def summarize(self, chat_history: ChatHistory, model: BaseArchytasModel, token_threshold: int):
        """

        """
        ...

    @abstractmethod
    async def get_records_to_summarize(self, chat_history: ChatHistory, model: BaseArchytasModel, token_threshold: int) -> list[RecordType]:
        """
        """
        ...

    async def generate_summaries(self, chat_history: ChatHistory, model: BaseArchytasModel, records_to_summarize: list[RecordType]):
        pass

    async def update_history(self, chat_history: ChatHistory, model: BaseArchytasModel, summaries: list[SummaryRecord]):
        pass


class PassiveContextManager(BaseContextManager):
    "Doesn't do anything to manage the context. Keeps all messages and history as passed in."

    async def summarize(self, chat_history, model, token_threshold):
        logger.debug("Summarization skipped due to Passive Context Management.")
        pass

    async def get_records_to_summarize(self, chat_history, model, token_threshold):
        return []


class OldestMessagesMixin:
    async def get_records_to_summarize(self, chat_history: ChatHistory, model: BaseArchytasModel, token_threshold: int) -> list[RecordType]:
        token_threshold = 3800
        all_records = await chat_history.records()
        filtered_records: list[RecordType] = [
            record for record in all_records
            if record.message is not None
            and (
                isinstance(record, SummaryRecord)
                or not isinstance(record.message, (SystemMessage, AutoContextMessage))
            )
        ]
        token_count = chat_history.base_tokens or 0
        target_idx: int = -1
        for idx, record in enumerate(filtered_records):
            print(f"{idx}: {record.__class__.__name__} {record.message.__class__.__name__} {record.token_count} {token_count}")
            if record.message:
                print(f"{record.message.content[:200]}")
            print("---------")
            if record.token_count is not None:
                token_count += record.token_count
            if token_count > token_threshold:
                target_idx = idx
                break

        print(f"{target_idx=}")
        done = False
        while target_idx >= 0 and not done:
            target_record = filtered_records[target_idx]
            target_message = target_record.message
            if isinstance(target_message, AIMessage) and target_message.tool_calls:
                target_idx -= 1
            else:
                done = True

        if target_idx >= 0:
            print(f"Targeting message {target_idx} out of {len(all_records)}")
            return filtered_records[:target_idx+1]
        else:
            print(f"No target message out of {len(all_records)}")
            return []


class DropOldestMessages(OldestMessagesMixin, BaseContextManager):
    "Drops oldest messages, deleting them completely. Therefore lies when it says it's summarizing."
    async def summarize(self, chat_history, model, token_threshold):
        records_to_drop = await self.get_records_to_summarize(chat_history, model, token_threshold)
        uuids_to_drop: set[str] = set(record.uuid for record in records_to_drop)
        chat_history.raw_records = [record for record in chat_history.raw_records if record.uuid in uuids_to_drop]


class SummarizeOldestMessages(OldestMessagesMixin, BaseContextManager):
    "Keeps recent messages, and summarizes old messages"

    async def summarize(self, chat_history, model, token_threshold):
        # token_threshold = token_threshold * 0.01
        token_threshold = 3800
        records_to_summarize: list[MessageRecord] = []
        summaries: list[SummaryRecord]  = []
        for record in await self.get_records_to_summarize(chat_history, model, token_threshold):
            match record:
                case MessageRecord():
                    records_to_summarize.append(record)
                case SummaryRecord():
                    summaries.append(record)

        if not records_to_summarize:
            return

        uuids = [record.uuid for record in records_to_summarize]

        summary_system_message = f"""\
You are an intelligent agent capable of reviewing conversations with an LLM by analyzing the system message, human
messages, AI responses, and tool usage and generating a detailed but terse summary of the conversation that can then be
used by other agents to continue long conversations that may exceed the context window.
"""
        if chat_history.system_message:
            summary_system_message += f"""
For reference purposes, the original system message for the following conversation is below, you should not summarize
this system message, but instead use the information to inform summarizing the requested messages.
```original system context message
{chat_history.system_message.message.content.strip()}
```
"""

        summary_human_message_parts = [
            """\
Please summarize all of the following messages into a single block of text that will replace the messages in future calls
to the LLM. Please include all details needed to preserve fidelity with the original meaning, while being as short as
reasonably possible so that the context window remains available for future conversation. Try to generate one sentence
per message, but you can combine messages or use multiple sentences as needed due to light or heavy information load,
respectively.

While summarizing, please include each message UUID along with a brief summary of the message(s). Messages can be grouped
for narrative sake, but try to keep each group to be 5 messages or less and be sure to include the UUIDs of each message
in the group.

If higher fidelity recall of the summarized messages are needed in the future, they original message content can be
retrieved using the UUID. However, be sure to focus the summaries on semantic understanding for conversation over
searching and retrieval.

The header for each message will include the message type and the message's UUID. For example a human message with UUID
ff06fc99e66d4d649406a670c9f9eb87, followed by an AI message response may look like this:

-----

```HumanMessage 857e620cf983428ea5a72f0c243414cb content
What is the weather today in Chicago?
```

```AIMessage b64613f35e0951a8e88fac40d3552301 content
Let me look that up by calling a tool.
```
```AIMessage b64613f35e0951a8e88fac40d3552301 tool_call
tool_name: check_weather
args: {{"location": "Chicago, IL"}}
tool_call_id: 13255332
```

```ToolMessage 51cfc17a46edac9b09031d416d3fbd64 content
{{"temperature": "68F", "humidity": "33%", "precip_chance": "3%"}}
```

-----

These messages could be summarized as follows:
```response
Messages: 857e620cf983428ea5a72f0c243414cb, b64613f35e0951a8e88fac40d3552301, 51cfc17a46edac9b09031d416d3fbd64
The user requested current weather conditions in Chicago, Illinois.
The "check_weather" tool was called.
The user was informed that it is a pleasant 68 degree day.
```

The above messages are just examples, do not include them in your summary.

"""
        ]

        if summaries:
            summary_human_message_parts.append("""
Below are previous summaries of this conversation. Please use them, if needed to inform the summaries for messages below,
but do not resummarize them.

### START OF PREVIOUS SUMMARIES ###
""")
            for summary_record in summaries:
                summary_human_message_parts.append(f"""\
```Summary {summary_record.uuid}
{summary_record.message.content}
```
""")
            summary_human_message_parts.append(f"""\
### END OF PREVIOUS SUMMARIES ###
""")

        summary_human_message_parts.append("""\
The messages to summarize start are below:

### START OF MESSAGES ###
""")
        for record in records_to_summarize:
            content_parts: list[str] = []
            msg_content: str = "<not found>"
            match record.message.content:
                case str():
                    msg_content = record.message.content
                case list():
                    msg_content = "\n".join(content.get("text", "") for content in record.message.content if content.get("type", None) == "text")
            content_parts.append(f"""\
```{record.message.__class__.__name__} {record.uuid} content
{msg_content.strip()}
```
""")
            if isinstance(record.message, AIMessage) and record.message.tool_calls:
                for tool_call in record.message.tool_calls:
                    content_parts.append(f"""
```{record.message.__class__.__name__} {record.uuid} tool_call
tool_name: {tool_call.get('name')}
args: {tool_call.get('args')}
tool_call_id: {tool_call.get('id')}
```
""")
            summary_human_message_parts.append("\n".join(content_parts))

        summary_human_message_parts.append("""
### END OF MESSAGES ###
""")
        human_content = "\n".join(summary_human_message_parts)
        messages = [SystemMessage(content=summary_system_message), HumanMessage(content=human_content)]
        response = await model._model.ainvoke(
            input=messages,
        )
        print(response)
        summary_text = response.content
        usage_metadata = getattr(response, "usage_metadata", None)
        summary_tokens = usage_metadata.get("output_tokens")
        print("Actual usage for query: ", usage_metadata)
        print("Estimated tokens", summary_tokens)
        # top_summarized_token_count = max(record for record in sum)

        summary_record = SummaryRecord(
            message=SystemMessage(
                content=f"""\
Below is a summary of {len(uuids)} messages with UUIDs of: {uuids}

```summary
{summary_text}
```
""",
            ),
            summarized_messages=set(uuids),
            token_count=summary_tokens,
        )
        chat_history.summaries.append(summary_record)
