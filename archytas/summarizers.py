import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeAlias, Callable, Awaitable
from .models.base import BaseArchytasModel

if TYPE_CHECKING:
    from .agent import Agent
    from .chat_history import ChatHistory, MessageRecord, SummaryRecord, RecordType, AutoContextMessage
    from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger("beaker")


MessageSummarizerFunction: TypeAlias = "Callable[[ToolMessage, ChatHistory, Agent], Awaitable[None]]"
LoopSummarizerFunction: TypeAlias = "Callable[[list[MessageRecord], ChatHistory, Agent, bool], Awaitable[None]]"
HistorySummarizerFunction: TypeAlias = "Callable[[ChatHistory, Agent], Awaitable[None]]"

MESSAGE_SUMMARIZATION_THRESHOLD: int = 1000
MESSAGE_SUMMARIZATION_SNIPPET_SIZE: int = 1000


class BaseHistorySummarizer(ABC):
    @abstractmethod
    async def get_records_to_summarize(self, chat_history: "ChatHistory", agent: "Agent", token_threshold: int) -> "list[RecordType]":
        """
        """
        ...


    async def summarize_loop(
        self,
        loop_records: "list[MessageRecord]",
        chat_history: "ChatHistory",
        agent: "Agent",
        force_update: bool = False
    ):
        from langchain_core.messages import ToolMessage
        coroutines = []
        for record in loop_records:
            if isinstance(record.message, ToolMessage) and (artifact := getattr(record.message, "artifact", None)):
                message = record.message
                tool_fn = artifact.get("tool", None)
                summarized: bool = artifact.get("summarized", False)
                summarizer: MessageSummarizerFunction = getattr(tool_fn, "summarizer", None)
                if summarizer and (force_update or not summarized):
                    coroutines.append(summarizer(message, chat_history, agent))
        await asyncio.gather(*coroutines)


    async def summarize_history(
        self,
        chat_history: "ChatHistory",
        agent: "Agent",
        token_threshold: int = -1,
        force_update: bool = False
    ):
        from .chat_history import MessageRecord, SummaryRecord, AIMessage, SystemMessage, HumanMessage
        print(f"Summarizing history {chat_history=}, {agent=}, {token_threshold=}, {force_update=}")

        # token_threshold = 2200
        records_to_summarize: list[MessageRecord[BaseMessage]] = []
        summaries: list[SummaryRecord] = []
        for record in await self.get_records_to_summarize(chat_history, agent.model, token_threshold):
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
            msg_content: str = record.message.text() or "<not found>"
#             match record.message.content:
#                 case str():
#                     msg_content = record.message.content
#                 case list():
#                     msg_content = "\n".join(content.get("text", "") for content in record.message.content if content.get("type", None) == "text")
#             content_parts.append(f"""\
# ```{record.message.__class__.__name__} {record.uuid} content
# {msg_content.strip()}
# ```
# """)
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
        response = await agent.model._model.ainvoke(
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


class DefaultHistoryManager(BaseHistorySummarizer):
    async def get_records_to_summarize(self, chat_history: "ChatHistory", model: BaseArchytasModel, token_threshold: int) -> "list[RecordType]":
        from .chat_history import RecordType, SummaryRecord, SystemMessage, AutoContextMessage, AIMessage
        token_threshold = 2200
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


class PassiveContextManager(BaseHistorySummarizer):
    async def get_records_to_summarize(self, chat_history, model, token_threshold):
        return []


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

    message.artifact["summarized"] = True
