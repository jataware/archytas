import copy
import json
import os
from abc import ABC, abstractmethod
from pydantic import BaseModel as PydanticModel, ConfigDict, create_model, Field
from pydantic.fields import FieldInfo
from typing import TYPE_CHECKING, Annotated, Any, Optional, ClassVar, Sequence
from functools import cache

from langchain.tools import StructuredTool

if TYPE_CHECKING:
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
        for env_name, env_value in self.env_settings.items():
            os.environ.setdefault(env_name, env_value)


def set_env_auth(**env_settings: dict[str, str]) -> None:
    for key, value in env_settings.items():
        if not (isinstance(key, str) and isinstance(value, str)):
            raise ValueError("EnvironmentAuth variables names and values must be strings.")
    for env_name, env_value in env_settings.items():
        os.environ.setdefault(env_name, env_value)


class ModelConfig(PydanticModel, extra='allow'):
    model_name: str
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    api_key: str | None = None
    summarization_ratio: float | None = None
    summarization_threshold: int | None = None
    summarization_threshold_pct: int | None = None

    # extra fields --
    # max_tokens: int | None = None
    # region: str | None = None

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
    error: Optional[str] = Field(...,
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


class BaseArchytasModel(ABC):
    DEFAULT_MODEL: ClassVar[Optional[str]] = None
    DEFAULT_SUMMARIZATION_RATIO: float = 0.5
    MODEL_PROMPT_INSTRUCTIONS: str = ""

    _model: "BaseChatModel"
    config: ModelConfig
    lc_tools: "list[StructuredTool] | None"

    def __init__(self, config: ModelConfig | dict, **kwargs) -> None:
        if isinstance(config, dict):
            self.config = ModelConfig(**config)
        else:
            self.config = config
        self.auth(**kwargs)
        self._model = self.initialize_model(**kwargs)
        self.lc_tools = None

    def auth(self, **kwargs) -> None:
        pass

    @property
    def additional_prompt_info(self) -> str | None:
        return None

    @property
    def default_summarization_threshold(self) -> int | None:
        context_max = self.contextsize(self.model_name)
        if context_max is None:
            return None
        return int(context_max * self.DEFAULT_SUMMARIZATION_RATIO)

    @property
    def summarization_threshold(self) -> int | None:
        context_size = self.contextsize(self.model_name)
        if summarization_threshold := getattr(self.config, 'summarization_threshold', None):
            if context_size is None:
                return summarization_threshold
            else:
                return min(int(summarization_threshold), context_size)
        elif summarization_ratio := getattr(self.config, 'summarization_ratio', None):
            pass
        elif summarization_threshold_pct := getattr(self.config, 'summarization_threshold_pct', None):
            summarization_ratio = float(summarization_threshold_pct) / 100
            self.config.summarization_ratio = summarization_ratio
        else:
            summarization_ratio = self.DEFAULT_SUMMARIZATION_RATIO
        if context_size is None:
            return None
        return int(context_size * summarization_ratio)

    @property
    def model(self) -> "BaseChatModel":
        if self.lc_tools is not None:
            return self._model.bind_tools(self.lc_tools)
        else:
            return self._model

    @property
    def model_name(self) -> str | None:

        lc_model_name = getattr(self._model, "model", None)
        if isinstance(lc_model_name, str):
            return lc_model_name

        config_model_name = getattr(self.config, "model_name", None)
        if isinstance(config_model_name, str):
            return config_model_name

        class_default_modelname = getattr(self, "DEFAULT_MODEL", None)
        return class_default_modelname


    @cache
    def contextsize(self, model_name: Optional[str]=None) -> int | None:
        if model_name is None:
            model_name = self.model_name
        return None

    @abstractmethod
    def initialize_model(self, **kwargs):
        ...

    def invoke(self, input, *, config=None, stop=None, agent_tools=None, **kwargs):
        result = self.model.invoke(
            self._preprocess_messages(input),
            config,
            stop=stop,
            agent_tools=agent_tools,
            **kwargs
        )
        return result

    @staticmethod
    @cache
    def convert_tools(archytas_tools: tuple[tuple[str, Any], ...])-> "list[StructuredTool]":
        tools = [final_answer, fail_task]
        for name, tool in archytas_tools:
            arg_dict = {}
            for arg_name, arg_type, arg_desc, _ in tool._args_list:
                arg_dict[arg_name] = Annotated[arg_type.sub_type, FieldInfo(description=arg_desc)]
            if "thought" not in arg_dict:
                arg_dict["thought"] = Annotated[str, FieldInfo(description="Reasoning around why this tool is being called.")]
            tool_model = create_model(name, **arg_dict)
            lc_tool = StructuredTool(
                name=name,
                description=tool.__doc__,
                args_schema=tool_model,
                func=tool,
            )
            tools.append(lc_tool)
        return tools

    def set_tools(self, agent_tools: dict):
        agent_tools = tuple(
            sorted(
                [(name, func) for name, func in agent_tools.items() if not getattr(func, '_disabled', False)],
                key=lambda tool: tool[0]
            )
        )
        tools = self.convert_tools(agent_tools)
        self.lc_tools = tools
        return tools

    async def get_num_tokens_from_messages(
        self,
        messages: "list[BaseMessage]",
        tools: Optional[Sequence] = None,
    ) -> int:
        try:
            return self._model.get_num_tokens_from_messages(messages=messages, tools=tools)
        except Exception as err:
            print(err)
            pass
        return 0

    async def token_estimate(
        self,
        messages: "Optional[list[BaseMessage]]" = None,
        tools: Optional[dict] = None
    ):
        if tools:
            tools = tuple(
                sorted(
                    [(name, func) for name, func in tools.items() if not getattr(func, '_disabled', False)],
                    key=lambda tool: tool[0]
                )
            )
            tools = self.convert_tools(tools)
        messages = self._preprocess_messages(messages)
        messages: list[BaseMessage] = copy.deepcopy(messages)
        for message in messages:
            if isinstance(message.content, list):
                content = []
                for content_object in message.content:
                    match content_object:
                        case {"type": "text"}:
                            content.append(content_object)
                        case dict():
                            content.append(
                                {
                                    "type": "text",
                                    "text": json.dumps(content_object)
                                }
                            )
                        case _:
                            content.append(content_object)
                message.content = content

        return await self.get_num_tokens_from_messages(messages=messages, tools=tools)

    async def ainvoke(self, input, *, config=None, stop=None, agent_tools: dict=None, **kwargs):
        if self.lc_tools is None and agent_tools is not None:
            self.set_tools(agent_tools)

        try:
            messages = self._preprocess_messages(input)
            result = await self.model.ainvoke(
                messages,
                config,
                stop=stop,
                **kwargs
            )
            return result
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
        tool_calls = response_message.tool_calls
        tool_thoughts = [tool_call["args"].pop("thought", f"Calling tool '{tool_call['name']}'") for tool_call in tool_calls]

        match content:
            case list():
                text = "\n".join(item['text'] for item in content if item.get('type', None) == "text")
            case "":
                if tool_calls:
                    text = "\n".join(tool_thoughts)
                else:
                    raise ValueError("Response from LLM does not include any content or tool calls. This shouldn't happen.")
            case str():
                text = content
            case _:
                # TODO: Finish this
                raise ValueError("Response from LLM does not match expected format. Expected ")
        if text == "":
            text = "Thinking..."
        return AgentResponse(text=text, tool_calls=tool_calls)

    def handle_invoke_error(self, error: BaseException):
        raise error
