from typing import Any
from archytas.tool_utils import tool, toolset
from archytas.python import Python
from archytas.utils import InstanceMethod


@tool()
def ask_user(query:str) -> str:
    """
    Ask the user a question and get their response. 
    
    You should ask the user a question if you do not have enough information to complete the task, and there is no suitable tool to help you.
    
    Args:
        query (str): The question to ask the user

    Returns:
        str: The user's response
    """
    return input(f'{query} $ ')



from datetime import datetime
import pytz
@tool(name='datetime')
def datetime_tool(format:str='%Y-%m-%d %H:%M:%S %Z', timezone:str='UTC') -> str:
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


#TODO: there's really only a single method that is a tool in this class. look into single method being a tool 
#      @InstanceMethod would be used to make sure the method is bound to an instance of the class
@toolset()
class PythonTool:
    """
    This is not a @toolset, but rather a single tool method that maintains a state between calls
    """
    def __init__(self, locals:dict[str,Any]|None=None, prelude_code:str|None=None):
        #TODO
        # create the python env instance
        # collect any @tools from the locals, and get their docstring?
        # add the locals to the env
        # run the prelude code in the env
        self.env = Python()

    @tool()
    # @InstanceMethod
    def run(self, code:str) -> str:
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


"""
python repl tool
- have a oneshot and multishot version
- should be able to import the functions to be included, and create the tool instance with the list of functions/modules/etc.
- maybe a list of recommended modules the llm can import


usage:
import numpy as np
py_tools:list[tools] = [fib_n, Jackpot, ModelSimulation, etc...]
imports = '''
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd
'''
py_tool = PythonREPL(imports, py_tools)


class Python:
    def __init__(self, libs/tools:list[])
        for tool in tools:
            if is_tool(tool): # pull out the internal function/class
            else: # it's a regular python library/function/etc.

"""