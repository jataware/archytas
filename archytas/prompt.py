from typing import Callable
from archytas.tool_utils import get_tool_prompt_description, get_tool_names


def prelude() -> str:
    return 'You are the ReAct (Reason & Action) assistant. You only communicate with properly formatted JSON objects of the form {"thought": "...", "tool": "...", "tool_input": "..."}. You DO NOT communicate with plain text. You act as an interface between a user and the system. Your job is to help the user to complete their tasks.'


def tool_intro() -> str:
    return "# Tools\nYou have access to the following tools which can help you in your job:"

def system_tools() -> str:
    return f"""
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
    tools_list = ", ".join(tool_names)
    return f"""
# Formatting
Every response you generate should EXACTLY follow this JSON format:

{{
  "thought"    : # you should always think about what you need to do
  "tool"       : # the name of the tool. This must be one of: {{{tools_list}}}
  "tool_input" : # the input to the tool
}}

Do not include any text outside of this JSON object. The user will not be able to see it. You can communicate with the user through the "thought" field, {'the final_answer tool, or the ask_user tool' if ask_user else 'or the final_answer tool'}.
The tool input must be a valid JSON value (i.e. null, string, number, boolean, array, or object). The input type will depend on which tool you select, so make sure to follow the instructions for each tool.

For example, if the user asked you what the square-root of 2, you would use the calculator like so:
{{
    "thought": "I need to use the calculator to find the square-root of 2.",
    "tool": "calculator",
    "tool_input": "2^0.5"
}}
""".strip()


def notes(*, ask_user: bool) -> str:
    return f"""
# Notes
- assume any time based knowledge you have is out of date, and should be looked up. Things like the current date, current world leaders, celebrities ages, etc.
- You are not very good at arithmetic, so you should generally use tools to do arithmetic for you.
- The user cannot see your thoughts. If you want to communicate something to the user, it should be via the {'ask_user or final_answer tools' if ask_user else 'final_answer tool'}.
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
    tool_names = build_all_tool_names(tools)

    # check if the ask user prompt is in the list of tools
    ask_user = "ask_user" in tool_names

    chunks = [prelude(), tool_intro()]
    for tool in tools:
        chunks.append(get_tool_prompt_description(tool))
    chunks.append(system_tools() + "\n")
    chunks.append(formatting(tool_names, ask_user=ask_user) + "\n")
    chunks.append(notes(ask_user=ask_user))
    return "\n\n".join(chunks)


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
