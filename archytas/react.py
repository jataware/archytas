import asyncio
import inspect
import json
import sys
import logging
import traceback
import typing
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, ToolCall

from .agent import Agent, BaseMessage, SystemMessage, AgentResponse
from .prompt import build_prompt, build_all_tool_names
from .tools import ask_user
from .tool_utils import make_tool_dict, sanitize_toolname
from .models.base import BaseArchytasModel
from .utils import extract_json


logger = logging.Logger("archytas")


class Undefined:
    ...


class FailedTaskError(Exception):
    def __init__(
            self,
            message: str,
            last_error: typing.Optional[BaseException] = None,
            tool_call_id: typing.Optional[str] = None,
            extra: typing.Optional[str] = None,
            *extra_args: object
        ) -> None:
        self.message = message
        self.last_error = last_error
        self.tool_call_id = tool_call_id
        self.extra = extra
        super().__init__(*extra_args)

    def __str__(self) -> str:
        err_str = f"""
Error message:
  {self.message}
"""
        if self.last_error:
            err_str += f"""
During processing, the following error was raised:
  {self.last_error}
"""
        if self.extra:
            err_str += f"""
Extra details:
  {self.extra}
"""
        return err_str


def catch_failure(fn):
    def handle_error(messages: list[BaseMessage], error: Exception):
        last_ai_message = None
        seen_tool_message_ids = set()
        for message in messages[::-1]:
            match message:
                case AIMessage():
                    last_ai_message = message
                    break
                case ToolMessage():
                    seen_tool_message_ids.add(message.id)
        if not last_ai_message:
            return
        if last_ai_message.tool_calls:
            missing_tool_ids = set(tool_call["id"] for tool_call in last_ai_message.tool_calls if tool_call["id"] not in seen_tool_message_ids)
            for missing_tool_id in missing_tool_ids:
                error_message = ToolMessage(
                    content=str(error),
                    tool_call_id=missing_tool_id,
                )
                messages.append(error_message)

    async def inner_async(self, *args, **kwargs):
        try:
            return await fn(self, *args, **kwargs)
        except FailedTaskError as failed_task:
            handle_error(self.messages, failed_task)
            raise
        except Exception as error:
        # TODO: Is it good to handle other errors here too? Probably, but flow can be tricky.
        # TODO: Would need to find dangling tools and generate tool messages for any dangling tool calls
            handle_error(self.messages, error)
            raise

    def inner(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except FailedTaskError as failed_task:
            handle_error(self.messages, failed_task)
            raise

    if inspect.iscoroutine(fn) or inspect.iscoroutinefunction(fn) or inspect.isawaitable(fn):
        return inner_async
    else:
        return inner


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


class AutoSummarizedToolMessage(ToolMessage):
    """A message that replaces its full tool output with a summary after the ReAct loop is complete."""

    summary: str = ""
    summarized: bool = False

    async def update_content(self):
        if not self.summarized:
            self.content=self.summary
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
                tool = self.tools[tool_name]
            elif "." in tool_name:
                matches = [name for name in self.tools.keys() if name.endswith(f".{tool_name}")]
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous name: Multiple tools called '{tool_name}'")
                tool = self.tools[matches[0]]
            if inspect.ismethod(tool):
                setattr(tool.__func__, '_disabled', True)
            else:
                setattr(tool, '_disabled', True)

        self.update_prompt()
        if self.model:
            self.model.set_tools(self.tools)

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

    @catch_failure
    async def react_async(self, query: str, react_context:dict=None) -> str:
        """
        Asynchronous react loop function.
        Continually calls tools until a satisfactory answer is reached.
        """

        # reset error and steps counter
        self.errors = 0
        self.steps = 0

        action: AgentResponse | None = None
        reaction: AgentResponse | None = None

        # Set the current query for use in tools, auto context, etc
        self.current_query = query
        action = await self.handle_message(HumanMessage(content=query))

        controller = LoopController()

        # ReAct loop
        while True:
            # Start processing last loops reaction as a new action
            if reaction is not None:
                action = reaction

            if not action.tool_calls:
                if action.text:
                    try:
                        # Check to ensure the content isn't an tool call in the wrong place.
                        action_json = json.loads(action.text)
                        if isinstance(action_json, dict) and "id" in action_json and "name" in action_json and "args" in action_json:
                            action.tool_calls.append(action_json)
                    except json.JSONDecodeError:
                        # Presume content to be a final answer
                        action.tool_calls.append({
                            "id": uuid.uuid4().hex,
                            "name": "final_answer",
                            "args": {
                                "response": action.text
                            }
                        })

            if action.tool_calls:
                for tool_call in action.tool_calls:
                    tool_id = tool_call["id"]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    # exit ReAct loop if agent says final_answer or fail_task
                    if tool_name == "final_answer":
                        await self.summarize_messages()
                        self.debug(
                            event_type="react_final_answer",
                            content={
                                "final_answer": tool_args,
                            }
                        )
                        self.current_query = None
                        # Store final answer as response in message,
                        response = tool_args.get("response", None)
                        if not response:
                            # TODO: Handle this case
                            raise ValueError("The LLM provided an empty message for a final_answer. This is not valid.")
                        self.messages.append(ToolMessage(content=str(response), tool_call_id=tool_id))
                        return response
                    if tool_name == "fail_task":
                        self.current_query = None
                        reason = tool_args.get("reason", "No reason identified.")
                        error = tool_args.get("error", None)
                        raise FailedTaskError(
                            message=reason,
                            extra=error,
                            tool_call_id=tool_id
                        )

                    if self.thought_handler:
                        self.thought_handler(action.text, tool_name, tool_args)


                    # run tool
                    try:
                        tool_fn = self.tools[tool_name]
                    except KeyError:
                        self.messages.append(ToolMessage(
                            content=f'Unknown tool "{tool_name}"\nAvailable tools: {", ".join(self.tools.keys())}',
                            tool_call_id=tool_id,
                        ))
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
                                "input": tool_args,
                            }
                        )
                        tool_output = await tool_fn.run(args=tool_args, tool_context=tool_context, self_ref=tool_self_ref)
                        self.debug(
                            event_type="react_tool_output",
                            content={
                                "tool": tool_name,
                                "input": tool_args,
                                "output": tool_output,
                            }
                        )

                        # Auto-summarize if required
                        if getattr(tool_fn, "autosummarize", False):
                            summary = f"Summary of action: Executed command '{tool_name}' with arguments '{tool_args}'"
                            tool_message = AutoSummarizedToolMessage(
                                content=tool_output,
                                summary=summary,
                                tool_call_id=tool_id,
                            )
                        else:
                            tool_message = ToolMessage(
                                content=tool_output,
                                tool_call_id=tool_id,
                            )

                        # Always add tool response before handling state changes to ensure message history is correct.
                        self.messages.append(tool_message)

                        # Have the agent observe the result, and get the next action
                        if self.verbose:
                            self.display_observation(tool_output)

                        # Log cases when tools override controller state
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
                    except Exception as e:
                        self.messages.append(ToolMessage(
                            content=f'error running tool "{tool_name}"\n\n:{e}\n{traceback.format_exception(e)}',
                            tool_call_id=tool_id
                        ))

            # Execute to fetch next step in the ReAct loop
            reaction = await self.execute()

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
        self.steps += 1
        if self.steps > self.max_react_steps:
            last_tool_id = None
            raise FailedTaskError(
                f"Too many steps ({self.steps} > max_react_steps) during task.\nLast action should have been either final_answer or fail_task. Instead got: {self.last_tool_name}",
                tool_call_id=last_tool_id,
            )
        return super().execute(additional_messages, tools=self.tools)

    def error(self, mesg: str, err: BaseException, tool_id: str | None = None) -> str:
        """error handling. If too many errors, break the ReAct loop. Otherwise tell the agent, and continue"""
        self.errors += 1
        if self.errors >= self.max_errors:
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
