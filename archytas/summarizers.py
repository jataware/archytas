import json
from typing import TYPE_CHECKING, TypeAlias, Callable

if TYPE_CHECKING:
    from .agent import Agent
    from .chat_history import BaseMessage, ToolMessage, ToolCall, ChatHistory


MessageSummarizerFunction: TypeAlias = "Callable[[ToolMessage, ChatHistory, Agent], None]"
LoopSummarizerFunction: TypeAlias = "Callable[[list[BaseMessage], ChatHistory, Agent], None]"
HistorySummarizerFunction: TypeAlias = "Callable[[], None]"

MESSAGE_SUMMARIZATION_THRESHOLD: int = 100
MESSAGE_SUMMARIZATION_SNIPPET_SIZE: int = 100


def default_summarizer(message: "ToolMessage", chat_history: "ChatHistory", agent: "Agent"):
    message_length = len(message.content)
    if message_length < MESSAGE_SUMMARIZATION_THRESHOLD:
        # Message is already short
        return
    # calling_record, tool_call = next(((record, tc) for record in chat_history.raw_records for tc in getattr(record.message, "tool_calls", [])), None)
    calling_record, tool_call = chat_history.get_tool_caller(message.tool_call_id)

    message.content = f"""\
Summary of run: Ran tool {tool_call.get("name")} with arguments: {json.dumps(tool_call.get("args"))}, which completed with status "{message.status}".
The first {MESSAGE_SUMMARIZATION_SNIPPET_SIZE} characters of the generated output are:
```
{message.content[:MESSAGE_SUMMARIZATION_SNIPPET_SIZE]}
```
"""
    message.artifact["summarized"] = True



def summarization_tool(summaries: dict):
    pass

def llm_message_summarizer(message: "ToolMessage", chat_history: "ChatHistory", agent: "Agent"):
    from .chat_history import MessageRecord
    calling_record: MessageRecord
    calling_record, tool_call = chat_history.get_tool_caller(message.tool_call_id)

    prompt_parts: list[str] = []
    prompt_parts.append(
        f"""\
Below is the output of calling a tool named {tool_call.get("name")} with arguments: {json.dumps(tool_call.get("args"))}, which completed with status "{message.status}".
Please summarize the output for the purpose of future reference in the conversational history.

To assist in the summarization, the agent's thought process for calling the tool was:
```
{calling_record.message.text}
```

Below is the output of the tool and what you should summarize.
```
{message.content}
```
"""
    )


    message.artifact["summarized"] = True
