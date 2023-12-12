from typing import Any
from archytas.tool_utils import tool, is_tool
from archytas.python import Python


@tool()
def ask_user(query: str) -> str:
    """
    Ask the user a question and get their response.

    You should ask the user a question if you do not have enough information to complete the task, and there is no suitable tool to help you.

    Args:
        query (str): The question to ask the user

    Returns:
        str: The user's response
    """
    return input(f"{query} $ ")


from datetime import datetime
import pytz


@tool(name="datetime")
def datetime_tool(format: str = "%Y-%m-%d %H:%M:%S %Z", timezone: str = "UTC") -> str:
    """
    Get the current date and time.

    Args:
        format (str, optional): The format to return the date and time in. Defaults to '%Y-%m-%d %H:%M:%S %Z'.
        timezone (str, optional): The timezone to return the date and time in. Defaults to 'UTC'.

    Returns:
        str: The current date and time in the specified format
    """
    # TODO: See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for more information.
    # TODO: list of valid timezones: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime(format)


@tool()
def timestamp() -> float:
    """
    Returns the current unix timestamp in seconds

    Returns:
        float: The current unix timestamp in seconds

    Examples:
        >>> timestamp()
        1681445698.726113
    """
    return datetime.now().timestamp()


class PythonTool:
    """
    Tool for running python code. If the user asks you to write code, you can run it here.
    """

    def __init__(self, locals: dict[str, Any] | None = None, prelude: str = ""):
        """
        Create a PythonTool instance

        Args:
            locals (dict[str,Any], optional): A dictionary of variables/classes/functions/modules/etc. to add to the python environment. Defaults to {}. @tools will be correctly extracted from their wrappers so they can be used in the environment.
            prelude (str, optional): Code to run before any other code. Defaults to ''. This could be used to import modules, or define functions/classes/etc. that will be used in the environment.
        """
        # create the python env instance
        self.env = Python()

        if locals is None:
            locals = {}

        prompt_chunks = []

        # collect any @tools from the locals, unwrap them
        # collect the description for each tool or function
        # TODO: make class tools collect the docstring for all tool methods
        env_update = {}
        for name, obj in locals.items():
            if is_tool(obj):
                env_update[name] = obj
                prompt_chunks.append(f"{name} = {obj.__doc__}")
            else:
                env_update[name] = obj
                prompt_chunks.append(f"{name} = {obj} ({type(obj)})")

        # update the env with the locals
        self.env.update_locals(env_update)

        # run the prelude code
        if prelude:
            self.env.run_script(prelude)
            prompt_chunks.append(
                f"The following prelude code was run in the environment:\n```{prelude}```"
            )

        # based on the locals+prelude, update the docstring for the run method to include descriptions about what is available in the environment
        # TODO: this is illegal...
        # self.run.__doc__ += f'\n\nThe following variables are available in the environment:\n\n' + '\n'.join(prompt_chunks)

        # if locals and prelude were empty, the prompt should say this is a fresh instance and anything needs to be imported

    @tool()
    def run(self, code: str) -> str:
        """
        Runs python code in a python environment.

        The environment is persistent between runs, so any variables created will be available in subsequent runs.
        The only visible effects of this tool are from output to stdout/stderr. If you want to view a result, you MUST print it.

        Args:
            code (str): The code to run

        Returns:
            str: The stdout output of the code
        """
        out, err = self.env.run_script(code)
        if err:
            raise Exception(err)

        return out


# @tool()
# def _man(objname:str) -> str:
#     ...
import pydoc


def pyman(obj: Any) -> str:
    """
    Get the documentation for an object in a python environment
    """
    return pydoc.render_doc(obj, renderer=pydoc.plaintext)
