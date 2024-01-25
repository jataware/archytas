from archytas.agent import Agent, Message, Role
from archytas.prompt import build_prompt, build_all_tool_names
from archytas.tools import ask_user
from archytas.tool_utils import make_tool_dict
import asyncio
import json
import pdb
import sys
import logging
import typing

logger = logging.Logger("archytas")


class Undefined:
    ...


class FailedTaskError(Exception):
    ...


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


class AutoSummarizedToolMessage(Message):
    """A message that replaces its full tool output with a summary after the ReAct loop is complete."""

    summary_content: str
    summarized: bool

    def __init__(
        self,
        role: Role,
        tool_content: str,
        summary_content: str,
    ):
        self.summarized = False
        self.summary_content = summary_content
        super().__init__(role, tool_content)

    async def update_content(self):
        self.update(content=self.summary_content)
        self.summarized = True


class ReActAgent(Agent):
    def __init__(
        self,
        *,
        model: str = "gpt-4-1106-preview",
        api_key: str | None = None,
        tools: list = None,
        allow_ask_user: bool = True,
        max_errors: int | None = 3,
        max_react_steps: int | None = None,
        thought_handler: typing.Callable | None = Undefined,
        **kwargs,
    ):
        """
        Create a ReAct agent

        Args:
            model (str): The model to use. Defaults to 'gpt-4'. Recommended not to change this. gpt-3.5-turbo doesn't follow the prompt format.
            api_key (str, optional): The OpenAI API key to use. If not set, defaults to reading the API key from the OPENAI_API_KEY environment variable.
            tools (list): A list of tools to use. Defaults to None. If None, only the system tools (final_answer, fail_task) will be used.
            allow_ask_user (bool): Whether to include the ask_user tool, which allows the model to ask the user for clarification. Defaults to True.
            max_errors (int, optional): The maximum number of errors to allow during a task. Defaults to 3.
            max_react_steps (int, optional): The maximum number of steps to allow during a task. Defaults to infinity.
            thought_handler (function, optional): Hook to control logging/output of the thoughts made in the middle of a react loop. Set to None to disable, or leave default of Undefined to
                    print to terminal. Otherwise expects a callable function with the signature of `func(thought: str, tool_name: str, tool_input: str) -> None`.
        """

        # create a dictionary for looking up tools by name
        tools = tools or []
        if allow_ask_user:
            tools.append(ask_user)
        tools.append(self)
        self.tools = make_tool_dict(tools)

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
        super().__init__(model=model, prompt=self.prompt, api_key=api_key, **kwargs)

        # react settings
        self.max_errors = max_errors or float("inf")
        self.max_react_steps = max_react_steps or float("inf")

        # number of errors and steps during current task
        self.errors = 0
        self.steps = 0

        # keep track of the last tool used (for error messages)
        self.last_tool_name = ""

    def thought_callback(self, thought: str, tool_name: str, tool_input: str) -> None:
        if self.verbose:
            # TODO: better coloring
            self.print(
                f"thought: {thought}\ntool: {tool_name}\ntool_input: {tool_input}\n"
            )

    def react(self, query: str) -> str:
        """
        Synchronous wrapper around the asynchronous react_async method.
        """
        return asyncio.run(self.react_async(query))

    async def react_async(self, query: str) -> str:
        """
        Asynchronous react loop function.
        Continually calls tools until a satisfactory answer is reached.
        """
        # reset error and steps counter
        self.errors = 0
        self.steps = 0

        # run the initial user query
        action_str = await self.query(query)

        controller = LoopController()

        # ReAct loop
        while True:
            self.debug(
                event_type="react_action",
                content={
                    "action": action_str,
                }
            )
            # Convert agent output to json
            try:
                if action_str.startswith('```json'):
                    action = json.loads(action_str[7:-3]) # Drop "```json" (7) from start and "```" (3) from end to only get the contents
                # Other parsing schemes can be added here if they are added in the future
                else:
                    action = json.loads(action_str)

            except json.JSONDecodeError:
                action_str = await self.error(
                    f'failed to parse action. Action must be a single valid json dictionary {{"thought": ..., "tool": ..., "tool_input": ...}}. There may not be any text or comments outside of the json object. Your input was: {action_str}'
                )
                continue

            # verify that action has the correct keys
            try:
                thought, tool_name, tool_input = self.extract_action(action)
                self.debug(
                    event_type="react_thought",
                    content={
                        "thought": thought,
                        "tool_name": tool_name,
                        "tool_input": tool_input
                    }
                )
                self.last_tool_name = tool_name  # keep track of the last tool used
            except AssertionError as e:
                action_str = await self.error(str(e))
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
                return tool_input
            if tool_name == "fail_task":
                raise FailedTaskError(tool_input)

            # run tool
            try:
                tool_fn = self.tools[tool_name]
            except KeyError:
                action_str = await self.error(f'Unknown tool "{tool_name}"\nAvailable tools: {", ".join(self.tools.keys())}')
                continue

            try:
                tool_context = {
                    "agent": self,
                    "tool_name": tool_name,
                    "raw_tool": tool_fn,
                    "loop_controller": controller,
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
                action_str = await self.error(f'error running tool "{tool_name}": {e}')

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
                return tool_output
            if controller.state == LoopController.STOP_FATAL:
                await self.summarize_messages()
                raise FailedTaskError(tool_output)

            # have the agent observe the result, and get the next action
            if self.verbose:
                self.display_observation(tool_output)
            if getattr(tool_fn, "autosummarize", False):
                action_str = await self.handle_message(AutoSummarizedToolMessage(
                    role=Role.system,
                    tool_content=tool_output,
                    summary_content=f"Summary of action: Executed command '{tool_name}' with input '{tool_input}'",
                ))
            else:
                action_str = await self.observe(tool_output)

    @staticmethod
    def extract_action(action: dict) -> tuple[str, str, str]:
        """Verify that action has the correct keys. Otherwise, raise an error"""
        assert isinstance(
            action, dict
        ), f"Action must be a json dictionary, got {type(action)}"
        assert "thought" in action, "Action json is missing key 'thought'"
        assert "tool" in action, "Action json is missing key 'tool'"
        assert "tool_input" in action, "Action json is missing key 'tool_input'"
        assert (
            len(action) == 3
        ), f"Action must have exactly 3 keys (thought, tool, tool_input), got ({', '.join(action.keys())})"

        thought = action["thought"]
        tool = action["tool"]
        tool_input = action["tool_input"]

        return thought, tool, tool_input

    def execute(self) -> str:
        """
        Execute the model and return the output (see `Agent.execute()`).
        Keeps track of the number of execute calls, and raises an error if there are too many.
        """
        self.steps += 1
        if self.steps > self.max_react_steps:
            raise FailedTaskError(
                f"Too many steps ({self.steps} > max_react_steps) during task.\nLast action should have been either final_answer or fail_task. Instead got: {self.last_tool_name}"
            )
        return super().execute()

    def error(self, mesg) -> str:
        """error handling. If too many errors, break the ReAct loop. Otherwise tell the agent, and continue"""

        self.errors += 1
        if self.errors >= self.max_errors:
            raise FailedTaskError(f"Too many errors during task. Last error: {mesg}")
        if self.verbose:
            if self.rich_print:
                self.print(f"[red]error: {mesg}[/red]", file=sys.stderr)
            else:
                self.print(f"error: {mesg}", file=sys.stderr)

        # tell the agent about the error, and get its response (call parent .error method)
        return super().error(mesg)

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
