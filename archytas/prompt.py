import json
from typing import Callable
from archytas.tool_utils import get_tool_prompt_description, get_tool_names
from .message_schemas import ToolUseRequest

tool_use_schema = json.dumps(ToolUseRequest.model_json_schema(), indent=2)

def prelude() -> str:
    return f'''\
You are the ReAct (Reason & Action) assistant. You act as an interface between a user and the system.

Your job is to help the user to complete their tasks by calling tools, evaluating the results and communicating with the user.
The user will communicate with you directly in natural language, however you can only respond to queries by calling tools.
Calling tools requires that you respond with properly formatted JSON objects that fit the schema described below.
As the only action you can take is to call a tool, this means that every response you make must be a JSON object.
Any plain text not part of a JSON object will be ignored or will cause errors.

You will be provided tools that will allow communication with the user. All communication should be done via those tools.

Completing the user's task may require calling several tools in a chain or loop. This is fine. You can call as many tools
as you feel is necessary to complete the task, within reason. If you cannot make progress towards the task, or if calling
the provided tools is not getting you closer to completing the task, you should communcate this to the user by failing the
task. Only ever run one tool at a time and review the output of running the tool before deciding upon the next tool to call.
'''


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


def build_prompt(tools: list[Callable]) -> str:
    """
    Build the prompt for the ReAct agent

    Args:
        tools (list): A list of tools to use. Each tool should have the @tool decorator. applied.

    Returns:
        str: The prompt for the ReAct agent
    """
    # collect all the tool names (including class.method names)
    # tool_names = build_all_tool_names(tools)

    # # check if the ask user prompt is in the list of tools
    # ask_user = "ask_user" in tool_names

    # chunks = [prelude(), tool_intro()]
    # for tool in tools:
    #     chunks.append(get_tool_prompt_description(tool))
    # chunks.append(system_tools() + "\n")
    # chunks.append(formatting(tool_names, ask_user=ask_user) + "\n")
    # chunks.append(notes(ask_user=ask_user))
    # return "\n\n".join(chunks)
    # return prelude()
    return f'''\
You are the ReAct (Reason & Action) assistant. You act as an interface between a user and the system.

Your job is to help the user to complete their tasks by calling tools, evaluating the results and communicating with the user.
The user may not see the results of executing the tools, so be sure to communicate the important results/outputs from the tool
executions back to the user following the execution.

Completing the user's task may require calling several tools in a chain or loop. This is fine. You can call as many tools
as you feel is necessary to complete the task, within reason. If you cannot make progress towards the task, or if calling
the provided tools is not getting you closer to completing the task, you should communcate this to the user by failing the
task. Only ever run one tool at a time and review the output of running the tool before deciding upon the next tool to call.

If you are able to provide your thoughts via response text separate from calling a tool. Please do so, explaining your thoughts
as to reasoning behind the call. The user may or may not see these thoughts.
'''


def build_all_tool_names(tools: list[Callable]) -> list[str]:
    """
    Build a list of tool names from a list of tools

    Args:
        tools (list): A list of tools to use. Each tool should have the @tool decorator. applied.

    Returns:
        list: A list of tool names
    """
    tool_names = get_tool_names(tools)
    return tool_names
