import asyncio
import copy
import functools
import hashlib
import inspect
import json
import logging
import uuid
import warnings
from dataclasses import MISSING
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING,Callable, Collection, Mapping, Optional, TypeVar, Generic, cast, Any, TypeAlias, Coroutine
from typing_extensions import Self

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, FunctionMessage, AIMessage, ToolCall
from langchain_core.messages import message_to_dict, messages_from_dict
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
    """
    DEPRECATED. Superseded by `InstructionSource` + tail-injected instruction
    messages (see plans/auto-context-relocation.md).

    Constructing this class directly still works, but nothing inside archytas
    creates or consumes instances any more — `Agent.set_auto_context()` now
    registers an `InstructionSource` on `ChatHistory.instruction`. Scheduled
    for removal in a future release.
    """

    default_content: str = Field(exclude=True)
    content_updater: Callable[[], str] = Field(exclude=True)

    def __init__(self, default_content: str, content_updater: Callable[[], str], model: Optional[BaseArchytasModel] = None, **kwargs):
        warnings.warn(
            "AutoContextMessage is deprecated. Use Agent.set_auto_context(), "
            "which now registers an InstructionSource delivered as a tail "
            "XML-tagged HumanMessage on every execute() call.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.update({"default_content": default_content, "content_updater": content_updater})
        super().__init__(content=default_content, **kwargs)
        self._token_count = None
        self._model = model

    @property
    def content_hash(self):
        return hashlib.sha1(self.text.encode()).hexdigest()

    @property
    def token_count(self):
        return self._token_count

    async def update_content(self):
        orig_hash = self.content_hash
        result = await ensure_async(self.content_updater())
        self.content = result
        if self.content_hash != orig_hash and self._model:
            self._token_count = await self._model.get_num_tokens_from_messages([self])


INSTRUCTION_TEMPLATE = "<system_context_update>\n{content}\n</system_context_update>"


class InstructionSource:
    """
    Registration for the tail-injected instruction message (plan §4.3).

    The rendered content is delivered as an XML-wrapped HumanMessage at the tail
    of the outgoing message list on every Agent.execute() call. It is never
    persisted into chat history, so per-turn content changes do not disturb the
    cacheable prefix of the conversation.
    """

    default_content: str
    content_updater: Optional[Callable[[], str]]
    auto_update: bool

    def __init__(
        self,
        default_content: str,
        content_updater: Optional[Callable[[], str]] = None,
        auto_update: bool = True,
        model: Optional[BaseArchytasModel] = None,
    ) -> None:
        self.default_content = default_content
        self.content_updater = content_updater
        self.auto_update = auto_update
        self._model = model
        self._current_content: str = default_content
        self._token_count: Optional[int] = None

    @property
    def current_content(self) -> str:
        return self._current_content

    @property
    def token_count(self) -> Optional[int]:
        return self._token_count

    async def update_content(self) -> None:
        """Re-run the updater (if any) and refresh the cached token count on change."""
        if self.content_updater is None:
            return
        orig = self._current_content
        result = await ensure_async(self.content_updater())
        self._current_content = result
        if self._current_content is None:
            self._token_count = 0
        elif self._current_content != orig and self._model is not None:
            try:
                self._token_count = await self._model.get_num_tokens_from_messages(
                    [HumanMessage(content=self.render())]
                )
            except Exception:
                # Token estimation failures (provider-specific) should not break
                # content updates; leave the prior estimate in place.
                pass

    def render(self) -> str:
        """Return the XML-wrapped content ready for delivery as a HumanMessage body."""
        return INSTRUCTION_TEMPLATE.format(content=self._current_content)

    def to_message(self) -> HumanMessage|None:
        if self._current_content is None:
            return None
        return HumanMessage(content=self.render())


class _AutoContextMessageShim:
    """
    DEPRECATED backwards-compat shim returned by
    `ChatHistory.auto_context_message` when an `InstructionSource` has been
    registered via `Agent.set_auto_context()`. Forwards the handful of
    attributes that downstream code historically reached for (`_model`,
    `content`, `update_content()`, `token_count`) onto the underlying
    `InstructionSource`.

    Note: this shim is NOT a subclass of `AutoContextMessage`, so
    `isinstance(shim, AutoContextMessage)` returns False. Any downstream code
    that depended on the isinstance relationship needs to migrate to the
    `InstructionSource` / `chat_history.instruction` API.
    """

    def __init__(self, instruction: "InstructionSource") -> None:
        # Store on the object dict directly so our descriptors below don't
        # intercept this one.
        object.__setattr__(self, "_instruction", instruction)

    @property
    def _model(self):
        return self._instruction._model

    @_model.setter
    def _model(self, value) -> None:
        self._instruction._model = value

    @property
    def content(self) -> str:
        return self._instruction._current_content

    @content.setter
    def content(self, value: str) -> None:
        self._instruction._current_content = value

    @property
    def default_content(self) -> str:
        return self._instruction.default_content

    @property
    def content_updater(self):
        return self._instruction.content_updater

    @property
    def token_count(self) -> Optional[int]:
        return self._instruction._token_count

    async def update_content(self) -> None:
        await self._instruction.update_content()


MessageType = TypeVar("MessageType", bound=BaseMessage)

@dataclass
class AgentResponse:
    text: str
    tool_calls: list[ToolCall]
    metadata: dict = dataclass_field(default_factory=dict)


# ----------------------------------------------------------------------------
# Serialization (serde) schema registry.
#
# Each serialized document/record carries a reified `SerdeSchema` — a named,
# versioned descriptor that can be inspected, compared, and embedded in the
# JSON. The intent is to decouple two independently-evolving concerns:
#
#   * The *envelope* schema (record/document structure: uuid, token_count,
#     metadata, which lists exist, etc.). Versioned by RECORD_*_SCHEMA /
#     CHAT_HISTORY_SCHEMA.
#   * The *message body* format — how a LangChain `BaseMessage` is turned into
#     JSON. Versioned separately by MESSAGE_FORMAT and stamped onto every
#     record as a `message_format` marker. If LangChain's wire format drifts,
#     only MESSAGE_FORMAT needs to move; the envelope versions stay put.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SerdeSchema:
    """A reified, comparable description of a serialization format.

    Instances are hashable/comparable (frozen dataclass), serialize to a small
    inspectable dict via :meth:`to_dict`, and support compatibility checks via
    :meth:`is_compatible`.
    """
    name: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SerdeSchema":
        return cls(name=str(data["schema"]), version=int(data["version"]))

    def is_compatible(self, other: "SerdeSchema") -> bool:
        """Two schemas are compatible when they name the same format at the
        same major version. (Versioning is currently a single integer, so this
        is exact-match; widen here if minor/forward-compatible versions land.)
        """
        return self.name == other.name and self.version == other.version


# Inner message-body format marker (LangChain `message_to_dict`/`messages_from_dict`).
MESSAGE_FORMAT = SerdeSchema(name="archytas.message.langchain", version=1)

# Envelope schemas for each record/document type.
MESSAGE_RECORD_SCHEMA = SerdeSchema(name="archytas.MessageRecord", version=1)
SUMMARY_RECORD_SCHEMA = SerdeSchema(name="archytas.SummaryRecord", version=1)
CHAT_HISTORY_SCHEMA = SerdeSchema(name="archytas.ChatHistory", version=1)


def _serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a JSON-compatible dict."""
    return message_to_dict(message)


def _deserialize_message(data: Mapping[str, Any]) -> BaseMessage:
    """Reconstruct a LangChain message from its serialized dict form."""
    return messages_from_dict([dict(data)])[0]


def _read_schema(data: Mapping[str, Any], key: str = "format") -> Optional[SerdeSchema]:
    """Extract the embedded :class:`SerdeSchema` from a serialized payload, if present."""
    raw = data.get(key)
    if not raw:
        return None
    return SerdeSchema.from_dict(raw)


def _check_schema(data: Mapping[str, Any], expected: SerdeSchema, key: str = "format") -> None:
    """Warn (do not raise) when a payload's declared schema is unknown or drifts.

    Deserialization stays lenient so that older/newer envelopes can still be
    loaded best-effort; the warning makes the drift visible for inspection.
    """
    found = _read_schema(data, key=key)
    if found is None:
        warnings.warn(
            f"Serialized payload is missing a '{key}' schema marker; "
            f"expected {expected.to_dict()}. Attempting to load anyway.",
            stacklevel=3,
        )
    elif not found.is_compatible(expected):
        warnings.warn(
            f"Serialized payload schema {found.to_dict()} is not compatible with "
            f"expected {expected.to_dict()}. Attempting to load anyway.",
            stacklevel=3,
        )


@dataclass
class MessageRecord(Generic[MessageType]):
    message: MessageType
    uuid: str = dataclass_field(default_factory=lambda: uuid.uuid4().hex)
    token_count: Optional[int] = dataclass_field(default=None)
    metadata: dict[str, Any] = dataclass_field(default_factory=lambda: {})
    react_loop_id: Optional[int] = dataclass_field(default=None)

    @classmethod
    def _record_schema(cls) -> SerdeSchema:
        """The envelope schema for this record type. Overridden by subclasses."""
        return MESSAGE_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record to a JSON-compatible dict.

        The result carries the envelope ``format`` schema and an inner
        ``message_format`` marker for the LangChain message body.
        """
        return {
            "format": self._record_schema().to_dict(),
            "message_format": MESSAGE_FORMAT.to_dict(),
            "uuid": self.uuid,
            "token_count": self.token_count,
            "metadata": copy.deepcopy(self.metadata),
            "react_loop_id": self.react_loop_id,
            "message": _serialize_message(self.message),
        }

    @classmethod
    def _common_kwargs(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        """Parse the fields shared by every record type from a serialized dict."""
        _check_schema(data, key="message_format", expected=MESSAGE_FORMAT)
        return {
            "message": _deserialize_message(data["message"]),
            "uuid": data["uuid"],
            "token_count": data.get("token_count"),
            "metadata": dict(data.get("metadata") or {}),
            "react_loop_id": data.get("react_loop_id"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MessageRecord":
        """Reconstruct a record from its serialized dict form.

        When invoked on the base class against a payload that declares a
        subclass schema (e.g. a :class:`SummaryRecord`), dispatches to that
        subclass so a heterogeneous list can be loaded through one entry point.
        """
        schema = _read_schema(data)
        if cls is MessageRecord and schema is not None and schema.name == SUMMARY_RECORD_SCHEMA.name:
            return SummaryRecord.from_dict(data)
        _check_schema(data, expected=cls._record_schema())
        return cls(**cls._common_kwargs(data))


@dataclass
class SummaryRecord(MessageRecord[SystemMessage]):
    message: SystemMessage
    summarized_messages: set[str] = dataclass_field(default_factory=lambda: set())

    @classmethod
    def _record_schema(cls) -> SerdeSchema:
        return SUMMARY_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Serialize, carrying ``summarized_messages`` as a (sorted) list across
        the JSON boundary; :meth:`from_dict` restores it to a set."""
        data = super().to_dict()
        data["summarized_messages"] = sorted(self.summarized_messages)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SummaryRecord":
        _check_schema(data, expected=cls._record_schema())
        kwargs = cls._common_kwargs(data)
        return cls(
            summarized_messages=set(data.get("summarized_messages") or []),
            **kwargs,
        )


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
    system_preamble: Optional[MessageRecord[SystemMessage]]
    user_preamble: Optional[MessageRecord[HumanMessage]]
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
    # `auto_context_message` and `auto_update_context` are exposed as
    # deprecation-shim properties below; they route to `instruction` for
    # backwards compatibility with code that predates the tail-injection
    # relocation.
    instruction: Optional[InstructionSource]
    last_sent_messages: Optional[list[BaseMessage]]

    def __init__(
        self,
        messages: Optional[Collection[BaseMessage]] = None,
        model: Optional[BaseArchytasModel] = None,
        loop_summarizer: Optional[callable] = default_loop_summarizer,
        history_summarizer: Optional[callable] = default_history_summarizer,
    ):
        self.base_tokens = 0
        self.system_preamble = None
        self.user_preamble = None
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

        # Tail-injection instruction registration (plan §4.3). Set via
        # Agent.set_auto_context. Rendered fresh into the outgoing list on every
        # Agent.execute() call and never persisted.
        # The legacy `auto_context_message` / `auto_update_context` fields are
        # now deprecation-shim properties on the class (see below); they
        # delegate to `instruction` for backwards compatibility.
        self.instruction = None

        # Observability hook (plan §4.5). Populated by Agent.execute() with a
        # snapshot of the final outgoing message list — including tail
        # injections — just before the model is invoked. Intended for tests,
        # verbose logging, and cache-hit measurement.
        self.last_sent_messages = None

    def set_system_message(
        self,
        system_message: SystemMessage|str
    ):
        if isinstance(system_message, str):
            system_message = SystemMessage(system_message)
        self.system_message = MessageRecord(
            message=system_message,
        )

    # ------------------------------------------------------------------
    # Deprecation shims for the legacy auto-context API. These forward to
    # the new `instruction` (InstructionSource) mechanism. Scheduled for
    # removal in a future release.
    # ------------------------------------------------------------------

    @property
    def auto_context_message(self) -> Optional["_AutoContextMessageShim"]:
        """DEPRECATED. Returns a shim over `self.instruction`. Use
        `chat_history.instruction` directly in new code."""
        warnings.warn(
            "ChatHistory.auto_context_message is deprecated; "
            "use chat_history.instruction (an InstructionSource) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.instruction is None:
            return None
        return _AutoContextMessageShim(self.instruction)

    @auto_context_message.setter
    def auto_context_message(self, value) -> None:
        """DEPRECATED. Accepts either None or a legacy AutoContextMessage; routes
        to the new instruction mechanism. Use `Agent.set_auto_context()` instead."""
        warnings.warn(
            "Direct assignment to ChatHistory.auto_context_message is deprecated; "
            "use Agent.set_auto_context() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if value is None:
            self.instruction = None
            return
        default_content = (
            getattr(value, "default_content", None)
            or (getattr(value, "content", "") if isinstance(getattr(value, "content", ""), str) else "")
        )
        self.instruction = InstructionSource(
            default_content=default_content,
            content_updater=getattr(value, "content_updater", None),
            auto_update=True,
            model=getattr(value, "_model", None),
        )

    @property
    def auto_update_context(self) -> bool:
        """DEPRECATED. Returns `self.instruction.auto_update` if an instruction
        is registered, else False."""
        warnings.warn(
            "ChatHistory.auto_update_context is deprecated; "
            "check/set chat_history.instruction.auto_update instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.instruction.auto_update if self.instruction is not None else False

    @auto_update_context.setter
    def auto_update_context(self, value: bool) -> None:
        """DEPRECATED. Sets `self.instruction.auto_update` if an instruction is
        registered; no-op otherwise."""
        warnings.warn(
            "ChatHistory.auto_update_context is deprecated; "
            "pass auto_update=... to Agent.set_auto_context() or set "
            "chat_history.instruction.auto_update directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.instruction is not None:
            self.instruction.auto_update = bool(value)

    @property
    def auto_context_message_token_estimate(self) -> int:
        """DEPRECATED. Always returns 0 — instruction token accounting now lives
        on `instruction_token_estimate`."""
        warnings.warn(
            "ChatHistory.auto_context_message_token_estimate is deprecated; "
            "use chat_history.instruction_token_estimate instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return 0

    async def update_auto_context(self):
        """DEPRECATED. Instruction updates are handled automatically by
        `Agent.execute()` via `build_tail_injections()`. If you need manual
        control, call `chat_history.instruction.update_content()` directly.
        """
        warnings.warn(
            "ChatHistory.update_auto_context() is deprecated; "
            "instruction updates happen automatically on Agent.execute(). "
            "Use chat_history.instruction.update_content() for manual control.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.instruction is not None and self.instruction.auto_update:
            await self.instruction.update_content()

    @property
    def instruction_token_estimate(self) -> int:
        if self.instruction and isinstance(self.instruction.token_count, int):
            return self.instruction.token_count
        return 0

    @property
    def token_overhead(self):
        return sum((value for value in (
            self.base_tokens,
            self.tool_token_estimate,
            self.instruction_token_estimate,
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
                model=agent.model,
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
        Build the persisted record list that Agent.execute() will send (prior to
        any ephemeral tail injections). The `auto_update_context` parameter is
        retained for signature compatibility; as of the auto-context-relocation
        work (Phase 1) there is nothing in the persisted list that needs
        per-call updating, so the flag is effectively a no-op.
        """
        records: list[RecordType] = []
        summarized_messages: set[str] = set()

        if self.system_message:
            records.append(self.system_message)

        for summary_record in self.summaries:
            records.append(summary_record)
            summarized_messages.update(summary_record.summarized_messages)

        if self.system_preamble:
            records.append(self.system_preamble)

        if self.user_preamble:
            records.append(self.user_preamble)

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

        if self.system_preamble:
            messages.append(self.system_preamble)

        if self.user_preamble:
            messages.append(self.user_preamble)

        messages.extend(self.raw_records)
        return messages

    async def all_messages(self, auto_update_context: bool = True) -> list[MessageType]:
        records = await self.all_records(auto_update_context=auto_update_context)
        return [cast(MessageType, record.message) for record in records]

    def set_system_preamble_text(self, value: str|MessageRecord|None=None):
        """
        Sets/updates the system_preamble
        The passed in value can be a raw string, an existing MessageRecord or None.

        Set text to an empty string or None to remove the preamble message.
        """
        match value:
            case str() if bool(value.strip()):
                self.system_preamble = MessageRecord(message=SystemMessage(content=value), metadata={"preamble": True})
            case MessageRecord():
                if isinstance(value.message, SystemMessage) and value.metadata.get("preamble", None) == True:
                    self.system_preamble = value
                else:
                    self.system_preamble = MessageRecord(message=SystemMessage(content=value.message.content), metadata={"preamble": True})
            case _:
                self.system_preamble = None

    def set_user_preamble_text(self, value: str|MessageRecord|None=None):
        """
        Sets/updates the user_preamble
        The passed in value can be a raw string, an existing MessageRecord or None.

        Set text to an empty string or None to remove the preamble message.
        """
        match value:
            case str() if bool(value.strip()): 
                self.user_preamble = MessageRecord(message=HumanMessage(content=value), metadata={"preamble": True})
            case MessageRecord():
                if isinstance(value.message, HumanMessage) and value.metadata.get("preamble", None) == True:
                    self.user_preamble = value
                else:
                    self.user_preamble = MessageRecord(message=HumanMessage(content=value.message.content), metadata={"preamble": True})
            case _: 
                self.user_preamble = None

    def add_message(
        self,
        message: MessageType,
        token_count: Optional[int] = None,
    ):
        """Appends a message to the message list."""
        record = MessageRecord(message=message, token_count=token_count, react_loop_id=self.current_loop_id)
        self.raw_records.append(record)
        return record

    # ------------------------------------------------------------------
    # Serialization (serde)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize this chat history to a JSON-compatible aggregate document.

        The document carries the :data:`CHAT_HISTORY_SCHEMA` envelope marker,
        the system message, the system/user preambles, the raw message records,
        the summary records, and a block of useful (non-reconstructive) metadata
        describing the model and token budget.

        The system message and preambles serialize to ``None`` when unset.

        The model itself and runtime-only state (summarizers, in-flight
        summarization task, last sent messages) are intentionally not
        serialized; provide a live ``model``/summarizers to :meth:`from_dict`.
        """
        model_meta: Optional[dict[str, Any]] = None
        if self.model is not None:
            try:
                context_window = self.model.contextsize()
            except Exception:
                context_window = None
            model_meta = {
                "class": type(self.model).__name__,
                "model_name": getattr(self.model, "model_name", None),
                "context_window": context_window,
            }

        return {
            "format": CHAT_HISTORY_SCHEMA.to_dict(),
            "metadata": {
                "model": model_meta,
                "current_loop_id": self.current_loop_id,
                "summarization_threshold": self.summarization_threshold,
                "tool_token_estimate": self.tool_token_estimate,
                "token_estimate": self._token_estimate,
            },
            "system_message": self.system_message.to_dict() if self.system_message else None,
            "system_preamble": self.system_preamble.to_dict() if self.system_preamble else None,
            "user_preamble": self.user_preamble.to_dict() if self.user_preamble else None,
            "raw_records": [record.to_dict() for record in self.raw_records],
            "summaries": [summary.to_dict() for summary in self.summaries],
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        model: Optional[BaseArchytasModel] = None,
        loop_summarizer: Optional[callable] = default_loop_summarizer,
        history_summarizer: Optional[callable] = default_history_summarizer,
    ) -> "ChatHistory":
        """Reconstruct a :class:`ChatHistory` from :meth:`to_dict` output.

        ``uuid``s on every record are preserved unchanged, and
        :class:`SummaryRecord`s are restored with their ``summarized_messages``
        sets intact. The system message and preambles are restored when present
        and left unset when ``None`` or absent. A live ``model`` and summarizers
        may be supplied since they are not part of the serialized document.
        """
        _check_schema(data, expected=CHAT_HISTORY_SCHEMA)

        history = cls(
            model=model,
            loop_summarizer=loop_summarizer,
            history_summarizer=history_summarizer,
        )

        # System message and preambles are optional: null or absent => leave unset.
        if data.get("system_message"):
            history.system_message = MessageRecord.from_dict(data["system_message"])
        if data.get("system_preamble"):
            history.system_preamble = MessageRecord.from_dict(data["system_preamble"])
        if data.get("user_preamble"):
            history.user_preamble = MessageRecord.from_dict(data["user_preamble"])

        history.raw_records = [
            MessageRecord.from_dict(record) for record in data.get("raw_records", [])
        ]
        history.summaries = [
            SummaryRecord.from_dict(summary) for summary in data.get("summaries", [])
        ]

        metadata = data.get("metadata") or {}
        if metadata.get("current_loop_id") is not None:
            history.current_loop_id = metadata["current_loop_id"]
        if metadata.get("summarization_threshold") is not None:
            history.summarization_threshold = metadata["summarization_threshold"]
        if metadata.get("tool_token_estimate") is not None:
            history.tool_token_estimate = metadata["tool_token_estimate"]
        if metadata.get("token_estimate") is not None:
            history._token_estimate = metadata["token_estimate"]

        return history
