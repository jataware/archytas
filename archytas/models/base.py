import os
from abc import ABC, abstractmethod
from pydantic import BaseModel as PydanticModel, ConfigDict, create_model, Field
from pydantic.fields import FieldInfo
from typing import TYPE_CHECKING, Annotated, Any
from functools import lru_cache


if TYPE_CHECKING:
    from langchain.tools import StructuredTool
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from ..agent import AgentResponse


class EnvironmentAuth:
    env_settings: dict[str, str]

    def __init__(self, **env_settings: dict[str, str]) -> None:
        for key, value in env_settings.items():
            if not (isinstance(key, str) and isinstance(value, str)):
                raise ValueError("EnvironmentAuth variables names and values must be strings.")
        self.env_settings = env_settings

    def apply(self):
        os.environ.update(self.env_settings)

def set_env_auth(**env_settings: dict[str, str]) -> None:
    for key, value in env_settings.items():
        if not (isinstance(key, str) and isinstance(value, str)):
            raise ValueError("EnvironmentAuth variables names and values must be strings.")
    os.environ.update(env_settings)


class ModelConfig(PydanticModel, extra='allow'):
    model_name: str
    api_key: str

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class FinalAnswerSchema(PydanticModel):
    response: str = Field(..., description=(
        "Final response that should be displayed to the user, answering the user's question, summarizing the "
        "results of the task, and/or display any useful information. If any important information is returned by using "
        "a tool, be sure to include that information here as the user does not have access to the raw output of the "
        "tool executions."
    ))



class BaseArchytasModel(ABC):

    MODEL_PROMPT_INSTRUCTIONS: str = ""

    model: "BaseChatModel"
    config: ModelConfig

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        self.config = config
        self.auth(**kwargs)
        self.model = self.initialize_model(**kwargs)

    def auth(self, **kwargs) -> None:
        pass

    @property
    def additional_prompt_info(self) -> str | None:
        return None

    @abstractmethod
    def initialize_model(self, **kwargs):
        ...

    def invoke(self, input, *, config=None, stop=None, agent_tools=None, **kwargs):
        return self.model.invoke(
            self._preprocess_messages(input),
            config,
            stop=stop,
            agent_tools=agent_tools,
            **kwargs
        )

    @staticmethod
    @lru_cache()
    def convert_tools(archytas_tools: tuple[tuple[str, Any], ...])-> "list[StructuredTool]":
        from langchain.tools import StructuredTool
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
        tools = [final_answer]
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


    async def ainvoke(self, input, *, config=None, stop=None, agent_tools: dict=None, **kwargs):
        agent_tools = tuple(sorted(agent_tools.items(), key=lambda tool: tool[0]))
        try:
            messages = self._preprocess_messages(input)
            if agent_tools is not None:
                tools = self.convert_tools(agent_tools)
                model = self.model.bind_tools(tools)
            else:
                model = self.model

            return await model.ainvoke(
                messages,
                config,
                stop=stop,
                **kwargs
            )
        except Exception as error:
            print(error)
            return self.handle_invoke_error(error)

    def _preprocess_messages(self, messages: "list[BaseMessage]"):
        return messages

    def _rectify_result(self, response_message: "AIMessage"):
        return response_message

    def process_result(self, response_message: "AIMessage") -> "AgentResponse":
        from ..agent import AgentResponse
        content = response_message.content
        match content:
            case list():
                text = "\n".join(item['text'] for item in content if item.get('type', None) == "text")
            case str():
                text = content
            case _:
                raise ValueError("Response from LLM does not match expected format. Expected ")
        tool_calls = response_message.tool_calls
        return AgentResponse(text=text, tool_calls=tool_calls)

    def handle_invoke_error(self, error: BaseException):
        raise error
