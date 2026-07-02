"""
Summarization pipeline for context management.

Two independent gates shrink the outgoing chat history, and keeping them
separate is what makes the flow tractable:

1. Selection (threshold): ``get_records_up_to_threshold`` decides *which*
   records are eligible for summarization, based on ``token_threshold``. It
   never truncates content — it only chooses where the summarize/keep boundary
   sits.
2. Fitting (context window): ``fit_records_to_context`` runs on the records the
   selection step chose, just before the summarization request, and
   middle-truncates copies of any records that, together, overflow the model's
   context window. Records that fit are passed through untouched, and the
   original records are never mutated.

The distinction matters: a record larger than the summarization threshold but
smaller than the context window is summarized in full and never truncated.
Truncation happens only when the *selected* records don't fit the context
window (see ``fit_records_to_context`` for the exact budget).
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, Callable, Awaitable

from jinja2 import Environment, Template, FileSystemLoader

from .models.base import BaseArchytasModel

prompt_path = Path(__file__).parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(prompt_path))

if TYPE_CHECKING:
    from .agent import Agent
    from .chat_history import ChatHistory, MessageRecord, SummaryRecord, RecordType
    from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger("beaker")


MessageSummarizerFunction: TypeAlias = "Callable[[ToolMessage, ChatHistory, Agent, Optional[Model]], Awaitable[None]]"
LoopSummarizerFunction: TypeAlias = "Callable[[list[MessageRecord], ChatHistory, Agent, bool Optional[Model]], Awaitable[None]]"
HistorySummarizerFunction: TypeAlias = "Callable[[ChatHistory, Agent], Awaitable[None]]"

MESSAGE_SUMMARIZATION_THRESHOLD: int = 1000
MESSAGE_SUMMARIZATION_SNIPPET_SIZE: int = 1000

# Headroom reserved out of the model's context window when sizing a
# summarization request: covers the system/user template scaffolding plus the
# generated summary output.
SUMMARIZATION_TOKEN_RESERVE: int = 8192
# Below this per-record token target, middle-truncation would leave too little
# content to be meaningful; fall back to a placeholder instead.
MIN_TRUNCATION_TOKENS: int = 256

TRUNCATION_NOTICE_TEMPLATE = (
    "\n\n[... {removed} characters removed from the middle of this message "
    "because it was too large to summarize in full ...]\n\n"
)
TRUNCATION_PLACEHOLDER = (
    "[Message content omitted from summarization because it was too large to "
    "fit in the context window.]"
)


async def get_summarizable_records(chat_history: "ChatHistory") -> "list[RecordType]":
    from .chat_history import RecordType, SummaryRecord, SystemMessage, AIMessage
    all_records = await chat_history.records(auto_update_context=False)
    # SystemMessage covers ContextMessage and the legacy AutoContextMessage
    # via inheritance, so a single isinstance check suffices.
    summarizable_records: list[RecordType] = [
        record for record in all_records
        if record.message is not None
        and (
            isinstance(record, SummaryRecord)
            or not isinstance(record.message, SystemMessage)
        )
    ]
    return summarizable_records


def _has_pending_tool_calls(message: "BaseMessage") -> bool:
    """True if `message` is an AIMessage that issued tool calls (whose
    ToolMessage responses therefore appear in later records)."""
    from .chat_history import AIMessage
    return isinstance(message, AIMessage) and bool(message.tool_calls)


def retreat_past_pending_tool_calls(records: "list[RecordType]", idx: int) -> int:
    """Move `idx` backward so that ``records[:idx+1]`` doesn't end on an
    AIMessage whose ToolMessage responses would be left out. Returns the new
    boundary index (may be -1 if no complete boundary exists at or before
    `idx`)."""
    while idx >= 0 and _has_pending_tool_calls(records[idx].message):
        idx -= 1
    return idx


def extend_to_include_tool_responses(records: "list[RecordType]", idx: int) -> int:
    """Move `idx` forward so that ``records[:idx+1]`` includes the ToolMessage
    responses following an AIMessage's tool calls, avoiding a split between an
    AIMessage and its responses. Returns the new boundary index."""
    from .chat_history import ToolMessage
    while idx < len(records) - 1 and (
        _has_pending_tool_calls(records[idx].message)
        or isinstance(records[idx + 1].message, ToolMessage)
    ):
        idx += 1
    return idx


async def get_records_up_to_threshold(chat_history: "ChatHistory", model: BaseArchytasModel, token_threshold: int) -> "list[RecordType]":
    """Select the prefix of summarizable records whose cumulative token count
    stays under ``token_threshold``.

    Note: the returned prefix may extend *past* the threshold. When the very
    first summarizable record is already larger than the whole threshold, the
    normal walk would select nothing and summarization could never make
    progress, so the fallback branch selects the minimal oversized prefix
    instead. This function never truncates content — fitting the selected
    records into the context window (including any truncation) is handled
    separately by ``fit_records_to_context``.
    """
    token_count = chat_history.base_tokens or 0
    target_idx: int = -1
    threshold_exceeded = False
    summarizable_records = await get_summarizable_records(chat_history)
    for idx, record in enumerate(summarizable_records):
        token_count += record.token_count or 0
        if token_count >= token_threshold:
            threshold_exceeded = True
            break
        target_idx = idx
    else:
        # The whole history fits under the threshold; nothing to summarize.
        target_idx = -1

    # Don't end the selection on an AIMessage whose tool responses would be split off.
    target_idx = retreat_past_pending_tool_calls(summarizable_records, target_idx)

    if target_idx >= 0:
        return summarizable_records[:target_idx+1]

    # Fallback (issue #85): a single record at the head of the history is
    # larger than the whole summarization threshold, so the walk above selected
    # nothing. Without this, summarization triggers on every query but never
    # makes progress. Select the minimal prefix instead — the head record plus
    # however many records are needed to avoid splitting an AIMessage from its
    # ToolMessage responses — and let the summarizer shrink any content that
    # doesn't fit in the context window (see fit_records_to_context).
    if threshold_exceeded and summarizable_records:
        end_idx = extend_to_include_tool_responses(summarizable_records, 0)
        selected = summarizable_records[:end_idx + 1]
        logger.warning(
            "A single message (%s tokens) exceeds the summarization threshold "
            "(%s tokens); summarizing %d record(s) beyond the threshold, "
            "truncating content as needed to fit the context window.",
            selected[0].token_count, token_threshold, len(selected),
        )
        return selected

    return []


def middle_truncate_text(text: str, keep_chars: int) -> str:
    """Cut characters out of the middle of `text` so roughly `keep_chars`
    characters remain, keeping the head and tail and inserting a notice where
    content was removed."""
    if keep_chars <= 0:
        return TRUNCATION_PLACEHOLDER
    if len(text) <= keep_chars:
        return text
    head = keep_chars // 2
    tail = keep_chars - head
    notice = TRUNCATION_NOTICE_TEMPLATE.format(removed=len(text) - keep_chars)
    return text[:head] + notice + text[-tail:]


async def shrink_record_for_summarization(
    record: "MessageRecord",
    model: BaseArchytasModel,
    target_tokens: int,
) -> "MessageRecord":
    """
    Return a copy of `record` whose message content has been middle-truncated
    to fit within `target_tokens` (issue #85 remediation b, with placeholder
    replacement as the fallback). The copy keeps the original record's uuid so
    the resulting summary excludes the original — which is never mutated — from
    future outgoing history.
    """
    from langchain_core.messages import HumanMessage
    from .chat_history import MessageRecord

    message = record.message.model_copy(deep=True)
    content = message.content
    new_content: str
    token_count: int

    if isinstance(content, str) and target_tokens >= MIN_TRUNCATION_TOKENS:
        token_count = record.token_count or 0
        new_content = content
        # Estimate a keep-size from the observed chars-per-token ratio, then
        # re-estimate and tighten a few times if the cut wasn't deep enough.
        for _ in range(4):
            chars_per_token = len(new_content) / max(token_count, 1)
            keep_chars = int(target_tokens * chars_per_token * 0.9)
            new_content = middle_truncate_text(content, keep_chars)
            token_count = await model.get_num_tokens_from_messages(
                [HumanMessage(content=new_content)]
            )
            if token_count <= target_tokens:
                break
        else:
            new_content = TRUNCATION_PLACEHOLDER
            token_count = await model.get_num_tokens_from_messages(
                [HumanMessage(content=new_content)]
            )
    else:
        # Non-string content (multimodal/structured blocks) has no
        # well-defined middle to cut; replace it outright.
        new_content = TRUNCATION_PLACEHOLDER
        token_count = await model.get_num_tokens_from_messages(
            [HumanMessage(content=new_content)]
        )

    message.content = new_content
    return MessageRecord(
        message=message,
        uuid=record.uuid,
        token_count=token_count,
        metadata=dict(record.metadata),
        react_loop_id=record.react_loop_id,
    )


async def fit_records_to_context(
    records: "list[MessageRecord]",
    model: BaseArchytasModel,
) -> "list[MessageRecord]":
    """
    Ensure the records destined for a summarization request fit within the
    model's context window (minus SUMMARIZATION_TOKEN_RESERVE headroom).
    Oversized records are replaced by middle-truncated copies (originals are
    left untouched); records that fit are passed through unchanged.
    """
    context_window = model.contextsize()
    if not context_window or not records:
        return records

    # Reserve headroom for the summarization prompt scaffolding + generated
    # summary, but never let that reserve claim more than half the window: on a
    # small context window SUMMARIZATION_TOKEN_RESERVE could otherwise dominate
    # (or drive the budget negative), leaving no room to summarize at all.
    budget = max(context_window - SUMMARIZATION_TOKEN_RESERVE, context_window // 2)
    fitted = list(records)
    total = sum(record.token_count or 0 for record in fitted)

    attempts = 0
    while total > budget and attempts < len(fitted) * 2:
        attempts += 1
        idx, largest = max(
            enumerate(fitted), key=lambda pair: pair[1].token_count or 0
        )
        largest_tokens = largest.token_count or 0
        if largest_tokens <= 0:
            break
        overage = total - budget
        target_tokens = max(largest_tokens - overage, 0)
        logger.warning(
            "Record %s (%s tokens) is too large to summarize in full; "
            "truncating its copy to ~%s tokens for the summarization request.",
            largest.uuid, largest_tokens, target_tokens,
        )
        shrunk = await shrink_record_for_summarization(largest, model, target_tokens)
        if (shrunk.token_count or 0) >= largest_tokens:
            # No progress is possible on this record; give up rather than loop.
            break
        fitted[idx] = shrunk
        total = sum(record.token_count or 0 for record in fitted)

    return fitted


async def get_records_up_to_loop(chat_history: "ChatHistory", model: BaseArchytasModel) -> "list[RecordType]":
    all_records = await get_summarizable_records(chat_history)
    loop_start_index = next((idx for idx, record in enumerate(all_records) if record.react_loop_id == chat_history.current_loop_id), None)
    if loop_start_index is not None:
        summarizable_records = all_records[:loop_start_index]
    else:
        summarizable_records = []
    return summarizable_records


async def default_loop_summarizer(
    loop_records: "list[MessageRecord]",
    chat_history: "ChatHistory",
    agent: "Agent",
    model: "BaseArchytasModel" = None,
    token_threshold: int = 4000,
    force_update: bool = False,
):
    from langchain_core.messages import ToolMessage
    if model is None:
        model = agent.model
    coroutines = []
    for record in loop_records:
        if isinstance(record.message, ToolMessage) and (artifact := getattr(record.message, "artifact", None)):
            message = record.message
            tool_name = artifact.get("tool_name", None)
            tool_fn = agent.tools[tool_name]
            summarized: bool = artifact.get("summarized", False)
            summarizer: MessageSummarizerFunction = getattr(tool_fn, "summarizer", None)
            if summarizer and (force_update or not summarized):
                coroutines.append(summarizer(message, chat_history, agent, model=model))
    await asyncio.gather(*coroutines)


async def default_history_summarizer(
    chat_history: "ChatHistory",
    agent: "Agent",
    recordset: "list[MessageRecord[BaseMessage]|SummaryRecord]",
    model: "BaseArchytasModel" = None,
    force_update: bool = False,
):
    from .chat_history import MessageRecord, SummaryRecord, AIMessage, SystemMessage, HumanMessage, BaseMessage
    logger.debug(f"Summarizing history {chat_history=}, {agent=}, {force_update=}")

    if not recordset:
        return

    if model is None:
        model = agent.model

    records_to_summarize: list[MessageRecord[BaseMessage]] = []
    summaries: list[SummaryRecord] = []
    for record in recordset:
        match record:
            case SummaryRecord():
                summaries.append(record)
            case MessageRecord():
                records_to_summarize.append(record)

    # Shrink any content too large to fit the summarization request into the
    # context window (issue #85). Only the copies sent to the summarizer are
    # truncated; the original records are untouched and are excluded from
    # future outgoing history once the summary lands.
    records_to_summarize = await fit_records_to_context(records_to_summarize, model)

    uuids = [record.uuid for record in records_to_summarize]

    jinja_globals = {
        "isinstance": isinstance,
        "MessageRecord": MessageRecord,
        "AIMessage": AIMessage,
        "SummaryRecord": SummaryRecord,
        "BaseMessage": BaseMessage,
    }
    system_template = jinja_env.get_template(
        name="summarization/default_history_summarization_system.jinja",
        globals=jinja_globals,
    )
    user_template = jinja_env.get_template(
        name="summarization/default_history_summarization_user.jinja",
        globals=jinja_globals
    )
    system_prompt = system_template.render({"chat_history": chat_history})
    user_prompt = user_template.render({
        "records_to_summarize": records_to_summarize,
        "summaries": summaries,
    })

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = await model._model.ainvoke(
        input=messages,
    )
    summary_text = response.content
    usage_metadata = getattr(response, "usage_metadata", None)
    summary_tokens = usage_metadata.get("output_tokens")

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


async def default_tool_summarizer(
    message: "ToolMessage",
    chat_history: "ChatHistory",
    agent: "Agent",
    model: "BaseArchytasModel" = None
):
    if model is None:
        model = agent.model

    message_length = len(message.content)
    if message_length < MESSAGE_SUMMARIZATION_THRESHOLD:
        # Message is already short
        return

    _, tool_call = chat_history.get_tool_caller(message.tool_call_id)

    message.content = f"""\
Summary of run:
ID: {tool_call.get("id")}
Name: {tool_call.get("name")}
Arguments: {json.dumps(tool_call.get("args"))}
Status: {message.status}
The first {MESSAGE_SUMMARIZATION_SNIPPET_SIZE} characters of the tools output are:
```
{message.content[:MESSAGE_SUMMARIZATION_SNIPPET_SIZE]}
```
"""
    message.artifact["summarized"] = True



async def llm_message_summarizer(message: "ToolMessage", chat_history: "ChatHistory", agent: "Agent"):
    from .chat_history import MessageRecord, AIMessage
    calling_record: MessageRecord[AIMessage]
    calling_record, tool_call = chat_history.get_tool_caller(message.tool_call_id)

    prompt: str = """\
You are a ReAct agent who has been conversing with the user.
The full history is growing large, and tool outputs can be large, so you will be helping by summarizing the output of a tool call.
Please format your output in a way that is concise, yet provides enough context in case you need to refer back to the tool call
in the future.
If you ever need the full output of the tool, there will be a tool available that will allow you to temporarily retrieve the full
output for the period in which it is needed.
"""
    query_parts: list[str] = []
    query_parts.append(
        f"""\
Below is the output of calling this tool:
    ToolCallId: {tool_call.get('id')}
    Name: {tool_call.get("name")}
    Arguments: {json.dumps(tool_call.get("args"))}
    Status: {message.status}.

Please summarize the output for the purpose of future reference in the conversational history.
""")
    if calling_record.message.text:
        query_parts.append(
            f"""\
To assist in the summarization, the your thought process when calling the tool was:
```
{calling_record.message.text}
```
""")
    query_parts.append(
        f"""\
Here is the output generated by the tool:
```
{message.text}
```
""")
    query = "\n".join(query_parts)
    logger.warning(f"`````summarization prompt`````\n{prompt}\n````` end `````")


    message.additional_kwargs["orig_content"] = message.content

    summarized_text = await agent.oneshot(prompt=prompt, query=query)
    logger.warning(summarized_text)
    #TODO replace text

    message.artifact["summarized"] = True
