from functools import cache
from typing import Any, Annotated

from pydantic import BaseModel as PydanticModel, Field, create_model
from pydantic.fields import FieldInfo
from langchain_core.tools import StructuredTool


class FinalAnswerSchema(PydanticModel):
    response: str = Field(..., description=(
        "Final response that should be displayed to the user, answering the user's question, summarizing the "
        "results of the task, and/or display any useful information. If any important information is returned by using "
        "a tool, be sure to include that information here as the user does not have access to the raw output of the "
        "tool executions."
    ))


class FailedTaskSchema(PydanticModel):
    reason: str = Field(...,
        description="A plain text explanation of the reason for the failure."
    )
    error: str | None = Field(...,
        description=(
            "A plain text rendering of the underlying error, along with the stacktrace, if available, for debugging "
            "purposes."
        )
    )


final_answer = StructuredTool(
    name="final_answer",
    description="""\
This tool should ALWAYS be called last during a successful ReAct loop to provide a final response to the user. This
ensures that the user is properly informed as the user does not have access to all outputs from tools.
The response should either answer a question, summarize the results of a task, and/or provide any useful information
that the user will find helpful or desirable.
""",
    args_schema=FinalAnswerSchema,
)

fail_task = StructuredTool(
    name="fail_task",
    description=(
        "The fail_task tool is used to indicate that you have failed to complete the task. You should use this "
        "tool to communicate the reason for the failure to the user. Do not call this tool unless you have given "
        "a good effort to complete the task.\n"
        "In particular, you should call this tool if the same request keeps repeating itself and/or you do not "
        "seem to be able to make progress."
    ),
    args_schema=FailedTaskSchema,
)


@cache
def convert_tools(archytas_tools: tuple[tuple[str, Any], ...]) -> list[StructuredTool]:
    tools = [final_answer, fail_task]
    for name, tool in archytas_tools:
        arg_dict = {}
        for arg_name, arg_type, arg_desc, _ in tool._args_list:
            arg_dict[arg_name] = Annotated[arg_type.sub_type, FieldInfo(description=arg_desc)]
        tool_model = create_model(name, **arg_dict)
        lc_tool = StructuredTool(
            name=name,
            description=tool.__doc__,
            args_schema=tool_model,
            func=tool,
        )
        tools.append(lc_tool)
    return tools
