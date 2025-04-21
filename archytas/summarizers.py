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
    from .chat_history import ChatHistory, MessageRecord, SummaryRecord, RecordType, AutoContextMessage
    from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger("beaker")


MessageSummarizerFunction: TypeAlias = "Callable[[ToolMessage, ChatHistory, Agent, Optional[Model]], Awaitable[None]]"
LoopSummarizerFunction: TypeAlias = "Callable[[list[MessageRecord], ChatHistory, Agent, bool Optional[Model]], Awaitable[None]]"
HistorySummarizerFunction: TypeAlias = "Callable[[ChatHistory, Agent], Awaitable[None]]"

MESSAGE_SUMMARIZATION_THRESHOLD: int = 1000
MESSAGE_SUMMARIZATION_SNIPPET_SIZE: int = 1000


async def get_summarizable_records(chat_history: "ChatHistory") -> "list[RecordType]":
    from .chat_history import RecordType, SummaryRecord, SystemMessage, AutoContextMessage, AIMessage
    all_records = await chat_history.records(auto_update_context=False)
    summarizable_records: list[RecordType] = [
        record for record in all_records
        if record.message is not None
        and (
            isinstance(record, SummaryRecord)
            or not isinstance(record.message, (SystemMessage, AutoContextMessage))
        )
    ]
    return summarizable_records


async def get_records_up_to_threshold(chat_history: "ChatHistory", model: BaseArchytasModel, token_threshold: int) -> "list[RecordType]":
    from .chat_history import AIMessage
    token_count = chat_history.base_tokens or 0
    target_idx: int = -1
    summarizable_records = await get_summarizable_records(chat_history)
    for idx, record in enumerate(summarizable_records):
        token_count += record.token_count
        if token_count >= token_threshold:
            break
        target_idx = idx
    else:
        target_idx = -1

    # Don't split AIMessages and their tool call
    done = False
    while target_idx >= 0 and not done:
        target_record = summarizable_records[target_idx]
        target_message = target_record.message
        if isinstance(target_message, AIMessage) and target_message.tool_calls:
            target_idx -= 1
        else:
            done = True

    if target_idx >= 0:
        return summarizable_records[:target_idx+1]
    else:
        return []


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
    force_update: bool = False,
):
    from .chat_history import MessageRecord, SummaryRecord, AIMessage, SystemMessage, HumanMessage, BaseMessage
    logger.debug(f"Summarizing history {chat_history=}, {agent=}, {force_update=}")

    if not recordset:
        return

    records_to_summarize: list[MessageRecord[BaseMessage]] = []
    summaries: list[SummaryRecord] = []
    for record in recordset:
        match record:
            case SummaryRecord():
                summaries.append(record)
            case MessageRecord():
                records_to_summarize.append(record)

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
    response = await agent.model._model.ainvoke(
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


async def default_tool_summarizer(message: "ToolMessage", chat_history: "ChatHistory", agent: "Agent"):
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
    if calling_record.message.text():
        query_parts.append(
            f"""\
To assist in the summarization, the your thought process when calling the tool was:
```
{calling_record.message.text()}
```
""")
    query_parts.append(
        f"""\
Here is the output generated by the tool:
```
{message.text()}
```
""")
    query = "\n".join(query_parts)
    logger.warning(f"`````summarization prompt`````\n{prompt}\n````` end `````")


    message.additional_kwargs["orig_content"] = message.content

    summarized_text = await agent.oneshot(prompt=prompt, query=query)
    logger.warning(summarized_text)
    #TODO replace text

    message.artifact["summarized"] = True
