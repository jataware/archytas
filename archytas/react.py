from archytas.agent import Agent
from archytas.prompt import build_prompt, build_all_tool_names
from archytas.tools import ask_user
from archytas.tool_utils import make_tool_dict
import json
from rich import print
import pdb
import sys
import logging

logger = logging.Logger('archytas')

class FailedTaskError(Exception): ...


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


class ReActAgent(Agent):
    def __init__(self, *, model:str='gpt-4', api_key:str|None=None, tools:list=None, allow_ask_user:bool=True, max_errors:int|None=3, max_react_steps:int|None=None, verbose:bool=False, **kwargs):
        """
        Create a ReAct agent

        Args:
            model (str): The model to use. Defaults to 'gpt-4'. Recommended not to change this. gpt-3.5-turbo doesn't follow the prompt format.
            api_key (str, optional): The OpenAI API key to use. If not set, defaults to reading the API key from the OPENAI_API_KEY environment variable.
            tools (list): A list of tools to use. Defaults to None. If None, only the system tools (final_answer, fail_task) will be used.
            allow_ask_user (bool): Whether to include the ask_user tool, which allows the model to ask the user for clarification. Defaults to True.
            max_errors (int, optional): The maximum number of errors to allow during a task. Defaults to 3.
            max_react_steps (int, optional): The maximum number of steps to allow during a task. Defaults to infinity.
            verbose (bool, optional): Whether to print the agent's thoughts and observations. Defaults to False.
        """

        # create a dictionary for looking up tools by name
        tools = tools or []
        if allow_ask_user:
            tools.append(ask_user)
        self.tools = make_tool_dict(tools)

        #check that the tools dict keys match the list of generated tool names
        names, keys = sorted(build_all_tool_names(tools)), sorted([*self.tools.keys()])
        assert names == keys, f'Internal Error: tools dict keys does not match list of generated tool names. {names} != {keys}'

        # create the prompt with the tools, and initialize the agent
        self.prompt = build_prompt(tools)
        super().__init__(model=model, prompt=self.prompt, api_key=api_key, **kwargs)

        # react settings
        self.max_errors = max_errors or float('inf')
        self.max_react_steps = max_react_steps or float('inf')
        self.verbose = verbose

        # number of errors and steps during current task
        self.errors = 0
        self.steps = 0

        # keep track of the last tool used (for error messages)
        self.last_tool_name = ''


    def react(self, query:str) -> str:
        # reset error and steps counter
        self.errors = 0
        self.steps = 0

        # run the initial user query
        action_str = self.query(query)

        controller = LoopController()

        # ReAct loop
        while True:

            logger.debug(f"""action: {action_str}""")
            # Convert agent output to json
            try:
                action = json.loads(action_str)
                
            except json.JSONDecodeError:
                action_str = self.error(f'failed to parse action. Action must be a single valid json dictionary {{"thought": ..., "tool": ..., "tool_input": ...}}. There may not be any text or comments outside of the json object. Your input was: {action_str}')
                continue

            # verify that action has the correct keys
            try:
                thought, tool_name, tool_input = self.extract_action(action)
                logger.debug(f"\nThought: {thought}\nTool name: {tool_name}\nTool input: {tool_input}")
                self.last_tool_name = tool_name # keep track of the last tool used
            except AssertionError as e:
                action_str = self.error(str(e))
                continue

            # print action
            if self.verbose:
                #TODO: better coloring
                print(f"thought: {thought}\ntool: {tool_name}\ntool_input: {tool_input}\n")

            # exit ReAct loop if agent says final_answer or fail_task
            if tool_name == 'final_answer':
                return tool_input
            if tool_name == 'fail_task':
                raise FailedTaskError(tool_input)

            # run tool
            try:
                tool_fn = self.tools[tool_name]
            except KeyError:
                action_str = self.error(f"unknown tool \"{tool_name}\"")
                continue

            try:
                tool_context = {
                    "agent": self,
                    "tool_name": tool_name,
                    "raw_tool": tool_fn,
                    "loop_controller": controller,
                }
                tool_output = tool_fn.run(tool_input, tool_context=tool_context)
            except Exception as e:
                action_str = self.error(f"error running tool \"{tool_name}\": {e}")

                continue

            # Check loop controller to see if we need to stop or error
            if controller.state == LoopController.STOP_SUCCESS:
                return tool_output
            if controller.state == LoopController.STOP_FATAL:
                raise FailedTaskError(tool_output)

            # have the agent observe the result, and get the next action
            if self.verbose:
                print(f"observation: {tool_output}\n")
            action_str = self.observe(tool_output)


    @staticmethod
    def extract_action(action:dict) -> tuple[str, str, str]:
        """Verify that action has the correct keys. Otherwise, raise an error"""
        assert isinstance(action, dict), f"Action must be a json dictionary, got {type(action)}"
        assert 'thought' in action, "Action json is missing key 'thought'"
        assert 'tool' in action, "Action json is missing key 'tool'"
        assert 'tool_input' in action, "Action json is missing key 'tool_input'"
        assert len(action) == 3, f"Action must have exactly 3 keys (thought, tool, tool_input), got ({', '.join(action.keys())})"

        thought = action['thought']
        tool = action['tool']
        tool_input = action['tool_input']

        return thought, tool, tool_input


    def execute(self) -> str:
        """
        Execute the model and return the output (see `Agent.execute()`).
        Keeps track of the number of execute calls, and raises an error if there are too many.
        """
        self.steps += 1
        if self.steps > self.max_react_steps:
            raise FailedTaskError(f"Too many steps ({self.steps} > max_react_steps) during task.\nLast action should have been either final_answer or fail_task. Instead got: {self.last_tool_name}")
        return super().execute()


    def error(self, mesg) -> str:
        """error handling. If too many errors, break the ReAct loop. Otherwise tell the agent, and continue"""

        self.errors += 1
        if self.errors >= self.max_errors:
            raise FailedTaskError(f"Too many errors during task. Last error: {mesg}")
        if self.verbose:
            print(f"[red]error: {mesg}[/red]", file=sys.stderr)

        #tell the agent about the error, and get its response (call parent .error method)
        return super().error(mesg)
