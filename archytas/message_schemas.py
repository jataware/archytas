import pydantic
from typing import Any

class ToolUseRequest(pydantic.BaseModel):
    thought: str = pydantic.Field(description="Thought that led to calling the tool")
    tool: str = pydantic.Field(description="Tool that will be invoked to satisfy thought")
    tool_input: Any = pydantic.Field(description="Argument(s) to pass in to the tool, if any")
    helpful_thought: bool = pydantic.Field(description="Determines whether the thought should be shown to the user")


class ToolUseResponse(pydantic.BaseModel):
    thought: str = pydantic.Field(description="Thought that led to calling the tool")
    tool: str = pydantic.Field(description="Tool that was invoked to satisfy thought")
    tool_input: Any = pydantic.Field(description="Argument(s) that were passed to the tool, if any")
    tool_output: Any = pydantic.Field(description="Output generated by running the tool mentioned")
    helpful_ouput: bool = pydantic.Field(description="Determines whether the output should be shown to the user")


class ReActError(pydantic.BaseModel):
    ename: str = pydantic.Field(description="Name of the error")
    eval: str = pydantic.Field(description="value of error")
    traceback: str | list[str] = pydantic.Field(description="Traceback related to error")
