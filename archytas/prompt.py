import json
from dataclasses import dataclass
from typing import Callable, Optional
from .message_schemas import ToolUseRequest

tool_use_schema = json.dumps(ToolUseRequest.model_json_schema(), indent=2)


# ---------------------------------------------------------------------------
# Sectioned prompt assembly
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSection:
    """A single named section in an assembled system prompt.

    Attributes:
        body: The section text. Empty/whitespace-only bodies are dropped at
            assembly time.
        name: Display name passed to the header formatter. ``None`` renders
            the section without a header.
        role: Stable, machine-readable identifier for programmatic matching
            in subclass overrides (filter, reorder, replace). Decoupled from
            ``name`` so renaming the display name does not silently change
            matching behavior.
    """

    body: str
    name: Optional[str] = None
    role: Optional[str] = None


HeaderFormatter = Callable[[Optional[str]], Optional[str]]


def DEFAULT_HEADER_FORMATTER(name: Optional[str]) -> Optional[str]:
    """Default section header formatter.

    Returns a markdown ``##``-level header for non-empty names; returns
    ``None`` for ``None`` or empty names so the section renders without a
    header.
    """
    if not name:
        return None
    return f"## {name}"


def assemble_prompt(
    sections: list[PromptSection],
    header_formatter: HeaderFormatter = DEFAULT_HEADER_FORMATTER,
) -> str:
    """Join an ordered list of ``PromptSection``s into a single prompt string.

    Sections with empty/whitespace-only bodies are dropped. Each remaining
    section's name is passed to ``header_formatter``; if the formatter
    returns a non-empty string the header is prepended to the body,
    separated by a blank line. Sections are joined by blank lines.
    """
    parts: list[str] = []
    for section in sections:
        body = section.body.strip() if section.body else ""
        if not body:
            continue
        header = header_formatter(section.name) if header_formatter else None
        if header:
            parts.append(f"{header}\n\n{body}")
        else:
            parts.append(body)
    return "\n\n".join(parts)


# Default Framework section text for the base Agent. ReAct-specific text
# lives in DEFAULT_REACT_FRAMEWORK_PROMPT below.
DEFAULT_BASE_FRAMEWORK_PROMPT = "You are a helpful assistant."


# Default Framework section text for ReActAgent. Subclasses may override
# the ``framework_prompt`` class attribute to replace this entirely.
DEFAULT_REACT_FRAMEWORK_PROMPT = """\
You are a ReAct (Reason & Action) agent. You act as an interface between a user and the system.

Your job is to help the user complete their tasks by calling the appropriate tools, evaluating results, and communicating effectively.

Key principles:
1. Focus on what the user has requested. Don't go off the rails.
2. Use tools as needed to fulfill the request - calling multiple tools is fine.
3. Be efficient but thorough in addressing the specific request.
4. If the user is explicit in their request, never expand the scope beyond those bounds.
5. Proper communication is key. Explain the what and why as you are working.
"""

def tool_intro() -> str:
    return "# Tools\nYou have access to the following tools which can help you in your job:"

def system_tools() -> str:
    return f"""
planning:
    The planning tool is used as a scratchpad for you to record notes or plans related to completing the user's task.
    This will not result in any changes in the system or provide any new information, however the information provided will
    be included during future loops. This can be used to provide a plan so you remain focused, or anything else you need
    to remember for a future point.
    Note: This action is not without cost, and so should only be used when deemed helpful. If the task can be completely
    resolved in one or two iterations, you likely should not use this tool.
    _input_: A string representing any information you want to store that may be helpful for you to complete the task.

final_answer:
    The final_answer tool is used to indicate that you have completed the task. You should use this tool to communicate the final answer to the user.
    _input_: A string representing a human-readable reply that conveys the final answer to the user's request, task or question.

fail_task
    The fail_task tool is used to indicate that you have failed to complete the task. You should use this tool to communicate the reason for the failure to the user. Do not call this tool unless you have given a good effort to complete the task.
    _input_: A plain text explanation of the reason for the failure, along with the raw root cause reason of the error for debugging.
""".strip()


# TODO: there should be some way to give an example relevant to the environment/tools...
#      or use a system tool for the example
def formatting(tool_names: list[str], *, ask_user: bool) -> str:
    tool_names += ["final_answer", "fail_task"]
    quoted_tool_list =", ".join(f"`{tool_name}`" for tool_name in tool_names)
    return f"""
# Formatting
Every response you generate should EXACTLY follow this JSON schema:
```jsonschema
{tool_use_schema}
```
The tools available are: {quoted_tool_list}

In the response text, you should answer using markdown encoding of the json and as such precede the JSON payload with "```json" and follow it with "```" like so:
```json
{{
  "thought": "This is what I'm thinking",
  "tool": "final_answer",
  "tool_input": "This is my final answer to the user",
}}
```

When specifying a tool, be sure to use the entire name, including any preceding or trailing identifiers.

You can communicate with the user via the {'`final_answer` tool, or the `ask_user` tool' if ask_user else '`final_answer` tool'}.
`tool_input` must be a valid JSON value (i.e. null, string, number, boolean, array, or object).
The input type will depend on which tool you select, so make sure to follow the instructions for each tool.

For example, if the user asked you what the square-root of 2, you would use the calculator like so:
```json
{{
    "thought": "I need to use the calculator to find the square-root of 2.",
    "tool": "calculator",
    "tool_input": "2^0.5"
}}
```
""".strip()


def notes(*, ask_user: bool) -> str:
    return f"""
# Notes
- assume any time based knowledge you have is out of date, and should be looked up, if possible. Things like the current date, current world leaders, celebrities ages, etc.
- You are not very good at arithmetic, so you should generally use tools to do arithmetic for you.
- The user may or may not see your thoughts. If you want to communicate something to the user, it should be via the {'ask_user or final_answer tools' if ask_user else 'final_answer tool'}.
""".strip()


def build_all_tool_names(tools: list[Callable]) -> list[str]:
    """
    Build a list of tool names from a list of tools

    Args:
        tools (list): A list of tools to use. Each tool should have the @tool decorator. applied.

    Returns:
        list: A list of tool names
    """
    # Deferred import to avoid a circular import (tool_utils -> agent ->
    # prompt) when this module is imported during agent initialization.
    from archytas.tool_utils import get_tool_names
    return get_tool_names(tools)
