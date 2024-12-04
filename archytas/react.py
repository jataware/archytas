import asyncio
import json
import sys
import logging
import typing
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, ToolCall

from .agent import Agent, BaseMessage, SystemMessage
from .prompt import build_prompt, build_all_tool_names
from .tools import ask_user
from .tool_utils import make_tool_dict, sanitize_toolname
from .models.base import BaseArchytasModel
from .utils import extract_json


logger = logging.Logger("archytas")


class Undefined:
    ...


class FailedTaskError(Exception):
    def __init__(self, message: str, last_error: typing.Optional[BaseException] = None, tool_call_id: typing.Optional[str] = None, *extra_args: object) -> None:
        self.message = message
        self.last_error = last_error
        self.tool_call_id = tool_call_id
        super().__init__(*extra_args)

    def __str__(self) -> str:
        result = f"Error completing task: {self.message}"
        if self.last_error:
            result += f"\n\nThe last error is as follows:\n{str(self.last_error)}"
        return result


class LoopController:
    PROCEED = 0
    STOP_SUCCESS = 1
    STOP_FATAL = 2

    state: int

    def __init__(self) -> None:
        self.reset()

    def set_state(self, new_value):
        self.state = new_value

    def reset(self):
        self.state = 0


class AutoSummarizedToolMessage(SystemMessage):
    """A message that replaces its full tool output with a summary after the ReAct loop is complete."""

    summary_content: str
    summarized: bool

    def __init__(
        self,
        tool_content: str,
        summary_content: str,
    ):
        self.summarized = False
        self.summary_content = summary_content
        super().__init__(tool_content)

    async def update_content(self):
        self.update(content=self.summary_content)
        self.summarized = True


class ReActAgent(Agent):
    def __init__(
        self,
        *,
        model: BaseArchytasModel | None = None,
        api_key: str | None = None,
        tools: list = None,
        allow_ask_user: bool = True,
        max_errors: int | None = 3,
        max_react_steps: int | None = None,
        thought_handler: typing.Callable | None = Undefined,
        messages: typing.Optional[list[BaseMessage]] | None = None,
        **kwargs,
    ):
        """
        Create a ReAct agent

        Args:
            model (BaseArchytasModel): The model to use. Defaults to OpenAIModel(model_name="gpt-4o").
            api_key (str, optional): The LLM provider API key to use. Defaults to None. If None, the provider will use the default environment variable (e.g. OPENAI_API_KEY).
            tools (list): A list of tools to use. Defaults to None. If None, only the system tools (final_answer, fail_task) will be used.
            allow_ask_user (bool): Whether to include the ask_user tool, which allows the model to ask the user for clarification. Defaults to True.
            max_errors (int, optional): The maximum number of errors to allow during a task. Defaults to 3.
            max_react_steps (int, optional): The maximum number of steps to allow during a task. Defaults to infinity.
            thought_handler (function, optional): Hook to control logging/output of the thoughts made in the middle of a react loop. Set to None to disable, or leave default of Undefined to
                    print to terminal. Otherwise expects a callable function with the signature of `func(thought: str, tool_name: str, tool_input: str) -> None`.
            messages (list[BaseMessage], optional): A list of messages to start the agent with. Defaults to None.
        """
        # create a dictionary for looking up tools by name
        tools = tools or []
        if allow_ask_user:
            tools.append(ask_user)
        tools.append(self)
        self._raw_tools = tools
        self.tools = make_tool_dict(tools)
        self.current_query = None

        if thought_handler is Undefined:
            self.thought_handler = self.thought_callback
        else:
            self.thought_handler = thought_handler

        # check that the tools dict keys match the list of generated tool names
        names, keys = sorted(build_all_tool_names(tools)), sorted([*self.tools.keys()])
        assert (
            names == keys
        ), f"Internal Error: tools dict keys does not match list of generated tool names. {names} != {keys}"

        # create the prompt with the tools, and initialize the agent
        self.prompt = build_prompt(tools)
        super().__init__(model=model, prompt=self.prompt, api_key=api_key, messages=messages, **kwargs)

        # react settings
        self.max_errors = max_errors or float("inf")
        self.max_react_steps = max_react_steps or float("inf")

        # number of errors and steps during current task
        self.errors = 0
        self.steps = 0

        # keep track of the last tool used (for error messages)
        self.last_tool_name = ""

    def update_prompt(self):
        self.prompt = build_prompt(self._raw_tools)
        if self.model.MODEL_PROMPT_INSTRUCTIONS:
            self.prompt += "\n\n" + self.model.MODEL_PROMPT_INSTRUCTIONS
        self.system_message = SystemMessage(content=self.prompt)

    def disable(self, *tool_names):
        if len(tool_names) == 0:
            return
        for tool_name in tool_names:
            if tool_name in self.tools:
                setattr(self.tools[tool_name], '_disabled', True)
            elif "." not in tool_name:
                matches = [name for name in self.tools.keys() if name.endswith(f".{tool_name}")]
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous name: Multiple tools called '{tool_name}'")
                elif len(matches) == 1:
                    subtool_name = matches[0]
                    method = self.tools[subtool_name]

                    setattr(method.__func__, '_disabled', True)

        self.update_prompt()


    def thought_callback(self, thought: str, tool_name: str, tool_input: str) -> None:
        if self.verbose:
            # TODO: better coloring
            self.print(
                f"thought: {thought}\ntool: {tool_name}\ntool_input: {tool_input}\n"
            )

    def react(self, query: str, react_context:dict=None) -> str:
        """
        Synchronous wrapper around the asynchronous react_async method.
        """
        return asyncio.run(self.react_async(query, react_context))

    async def react_async(self, query: str, react_context:dict=None) -> str:
        """
        Asynchronous react loop function.
        Continually calls tools until a satisfactory answer is reached.
        """

        # reset error and steps counter
        self.errors = 0
        self.steps = 0

        action = None
        reaction = None

        # Set the current query for use in tools, auto context, etc
        self.current_query = query
        action = await self.handle_message(HumanMessage(content=query))

        controller = LoopController()

        # ReAct loop
        while True:
            # Start processing last loops reaction as a new action
            if reaction is not None:
                action = reaction

            message = self.messages[-1]
            tool_id = uuid.uuid4().hex

            # Convert agent output to json
            if isinstance(action, str):
                try:
                    # First, assume it is a ```json ... ``` blocked string. If no blocks found (ValueError), try in case
                    # the action string is a valid JSON string.
                    try:
                        action = extract_json(action)
                    except ValueError as extract_err:
                        if "Unable to find json block" in extract_err.args:
                            action = json.loads(action)
                        else:
                            raise
                except json.JSONDecodeError as decode_err:
                    reaction = await self.error("Error parsing JSON", decode_err, tool_id=tool_id)
                    continue
                except Exception as e:
                    raise

            # verify that action has the correct keys
            try:
                thought, tool_name, tool_input, helpful_thought = self.extract_action(action)
                self.debug(
                    event_type="react_thought",
                    content={
                        "id": tool_id,
                        "thought": thought,
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "helpful_thought": helpful_thought,
                    }
                )
                self.last_tool_name = tool_name  # keep track of the last tool used
            except AssertionError as e:
                reaction = await self.error("Error", e, tool_id=tool_id)
                continue

            if self.thought_handler:
                self.thought_handler(thought, tool_name, tool_input)

            # exit ReAct loop if agent says final_answer or fail_task
            if tool_name == "final_answer":
                await self.summarize_messages()
                self.debug(
                    event_type="react_final_answer",
                    content={
                        "final_answer": tool_input,
                    }
                )
                self.current_query = None
                # Store final answer as response in message,
                if not isinstance(tool_input, str):
                    tool_input = str(tool_input)
                self.messages.append(AIMessage(content=tool_input))
                return tool_input
            else:
                message.tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=sanitize_toolname(tool_name),
                        args={"arg_string": tool_input},
                    )
                )
            if tool_name == "fail_task":
                self.current_query = None
                raise FailedTaskError(
                    f"Unable to complete the requested task. Giving up.\n Tool input: {tool_input}",
                    tool_call_id=tool_id
                )

            # run tool
            try:
                tool_fn = self.tools[tool_name]
            except KeyError:
                reaction = await self.error(f'Unknown tool "{tool_name}"\nAvailable tools: {", ".join(self.tools.keys())}', None, tool_id=tool_id)
                continue

            try:
                tool_context = {
                    "agent": self,
                    "tool_name": tool_name,
                    "raw_tool": tool_fn,
                    "loop_controller": controller,
                    "react_context": react_context,
                }
                tool_self_ref = getattr(tool_fn, "__self__", None)
                self.debug(
                    event_type="react_tool",
                    content={
                        "tool": tool_name,
                        "input": tool_input,
                    }
                )
                tool_output = await tool_fn.run(tool_input, tool_context=tool_context, self_ref=tool_self_ref)
                self.debug(
                    event_type="react_tool_output",
                    content={
                        "tool": tool_name,
                        "input": tool_input,
                        "output": tool_output,
                    }
                )
            except Exception as e:
                reaction = await self.error(f'error running tool "{tool_name}"', e, tool_id=tool_id)
                continue

            if controller.state != LoopController.PROCEED:
                self.debug(
                    event_type="react_controller_state",
                    content={
                        "state": controller.state,
                    }
                )
            # Check loop controller to see if we need to stop or error
            if controller.state == LoopController.STOP_SUCCESS:
                await self.summarize_messages()
                self.current_query = None
                return tool_output
            if controller.state == LoopController.STOP_FATAL:
                await self.summarize_messages()
                self.current_query = None
                raise FailedTaskError(
                    tool_output,
                    tool_call_id=tool_id
                )

            # have the agent observe the result, and get the next action
            if self.verbose:
                self.display_observation(tool_output)
            if getattr(tool_fn, "autosummarize", False):
                reaction = await self.handle_message(AutoSummarizedToolMessage(
                    tool_content=tool_output,
                    summary_content=f"Summary of action: Executed command '{tool_name}' with input '{tool_input}'",
                ))
            else:
                reaction = await self.handle_message(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_id,
                    )
                )

    @staticmethod
    def extract_action(action: dict) -> tuple[str, str, str, bool]:
        """Verify that action has the correct keys. Otherwise, raise an error"""
        assert isinstance(
            action, dict
        ), f"Action must be a json dictionary, got {type(action)}"
        assert "thought" in action, "Action json is missing key 'thought'"
        assert "tool" in action, "Action json is missing key 'tool'"
        assert "tool_input" in action, "Action json is missing key 'tool_input'"

        thought = action["thought"]
        tool = action["tool"]
        tool_input = action["tool_input"]
        helpful_thought = action.get("helpful_thought", True)


        return thought, tool, tool_input, helpful_thought

    def execute(self, additional_messages: list[BaseMessage] = []) -> str:
        """
        Execute the model and return the output (see `Agent.execute()`).
        Keeps track of the number of execute calls, and raises an error if there are too many.
        """
        try:
            self.steps += 1
            if self.steps > self.max_react_steps:
                last_tool_id = None
                raise FailedTaskError(
                    f"Too many steps ({self.steps} > max_react_steps) during task.\nLast action should have been either final_answer or fail_task. Instead got: {self.last_tool_name}",
                    tool_call_id=last_tool_id,
                )
            result = super().execute(additional_messages)
        except FailedTaskError as err:
            # Ensure that the last message is a tool message, adding a new one if needed
            if not isinstance(self.messages[-1], ToolMessage):
                content = f"""\
Task failed:
{err.message}
"""
                if err.last_error:
                    content += f"""
During processing, the following error was raised:
{err.last_error}
"""
                error_message = ToolMessage(
                    content=content,
                    tool_call_id=err.tool_call_id,
                )
                self.messages.append(error_message)
            raise
        return result

    def error(self, mesg: str, err: BaseException, tool_id: str | None = None) -> str:
        """error handling. If too many errors, break the ReAct loop. Otherwise tell the agent, and continue"""
        error_message = ToolMessage(
            content=f"{mesg}: {err}",
            tool_call_id=tool_id,
        )

        self.errors += 1
        if self.errors >= self.max_errors:
            self.messages.append(error_message)
            raise FailedTaskError(
                f"Too many errors during task. Last error: {mesg}",
                last_error=err,
                tool_call_id=tool_id
            )
        if self.verbose:
            if self.rich_print:
                self.print(f"[red]error: {mesg}[/red]", file=sys.stderr)
            else:
                self.print(f"error: {mesg}", file=sys.stderr)

        # tell the agent about the error, and get its response (call parent .error method)
        return super().error(error_message)

    def display_observation(self, observation):
        """
        Display the observation. Can be overridden by subclasses to display the observation in different ways.
        """
        self.print(f"observation: {observation}\n")

    async def summarize_messages(self):
        """Summarizes and self-summarizing tool messages."""
        for message in self.messages:
            if isinstance(message, AutoSummarizedToolMessage) and not message.summarized:
                await message.update_content()
