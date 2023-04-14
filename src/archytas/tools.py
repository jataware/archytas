import inspect
from docstring_parser import parse as parse_docstring
from rich import traceback; traceback.install(show_locals=True)
from textwrap import indent
from typing import Callable

import pdb


#TODO: separate system tools from library tools from user tools
#      ideally system tools will have no library dependencies



def tool(*, name:str|None=None):
    """
    Converts a function into a tool for ReAct agents to use.
    """
    def decorator(func):
        args_list, ret, desc = get_tool_signature(func)
        def wrapper(args:dict|list|str|int|bool|None):
            
            if isinstance(args, dict):
                result = func(**args)
            elif isinstance(args, list):
                result = func(*args)
            elif isinstance(args, (str, int, bool)):
                result = func(args)
            elif args is None:
                result = func()
            else:
                raise TypeError(f"args must be a dict, list, str, int, bool, or None. Got {type(args)}")

            #TODO: structure input args so they can be used to call the function
            # args = validate_tool_call(func, args, args_list)
            
            # result = func(**args)

            #convert the result to a string if it is not already a string
            if not isinstance(result, str):
                result = str(result)

            return result
        #attach usage description to the wrapper function
        wrapper._name = name if name else func.__name__
        wrapper._is_tool = True
        wrapper._args_list = args_list
        wrapper._ret = ret
        wrapper._desc = desc

        return wrapper
    
    return decorator


def get_tool_prompt_description(func):
    #check that this function has the @tool decorator attached
    if not hasattr(func, '_is_tool'):
        raise ValueError(f"Function {func.__name__} does not have the @tool decorator attached")
    
    #get the list of arguments
    args_list = func._args_list
    ret_name, ret_type, ret_description = func._ret
    short_desc, long_desc, examples = func._desc

    
    chunks = []
    tab = "    "

    ############### NAME/DESCRIPTION ###############
    chunks.append(f"{func._name}:\n")
    if short_desc:
        chunks.append(f"{tab}{short_desc}\n\n")
    if long_desc:
        chunks.append(f"{indent(long_desc, tab)}\n\n")
    
    #################### INPUT ####################
    chunks.append('    _input_: ')
    
    if len(args_list) == 0:
        chunks.append("None")
    
    elif len(args_list) == 1:
        arg_name, arg_type, arg_desc, arg_default = args_list[0]
        chunks.append(f"({arg_type.__name__}")
        if arg_default:
            chunks.append(f", optional")
        chunks.append(f") {arg_desc}")
    
    else:
        chunks.append("a json object with the following fields:\n    {")
        for arg_name, arg_type, arg_desc, arg_default in args_list:
            chunks.append(f'\n        "{arg_name}": # ({arg_type.__name__}')
            if arg_default:
                chunks.append(f", optional")
            chunks.append(f") {arg_desc}")
        chunks.append(f"\n{tab}}}")

    #################### OUTPUT ####################
    chunks.append("\n    _output_: ")
    chunks.append(f"({ret_type.__name__}) {ret_description}")


    ############### EXAMPLES ###############
    #TODO: examples need to be parsed...


    return ''.join(chunks)




def get_tool_signature(func) -> tuple[list[tuple[str, type, str|None, str|None]], tuple[str|None, type|None, str|None], tuple[str, str|None, list[str]]]:
    """
    Check that the docstring and function signature match for a tool function, and return all function information.

    Args:
        func (function): The function to check and extract information from

    Returns:
        args_list: A list of tuples (name, type, description, default) for each argument
        ret: A tuple (name, type, description) for the return value
        desc: A tuple (short_description, long_description, examples) from the docstring for the function
    """
    # get the function signature from the function and the docstring
    docstring = parse_docstring(func.__doc__)
    signature = inspect.signature(func)

    # Extract argument information from the docstring
    docstring_args = {arg.arg_name: (arg.type_name, arg.description, arg.default) for arg in docstring.params}

    # Extract argument information from the signature
    signature_args = {k: v.annotation for k, v in signature.parameters.items()}

    # Check if the docstring argument names match the signature argument names
    if set(docstring_args.keys()) != set(signature_args.keys()):
        raise ValueError(f"Docstring argument names do not match function signature argument names for function '{func.__name__}'")

    # Check if the docstring argument types match the signature argument types
    for arg_name, arg_type in signature_args.items():
        docstring_arg_type, _, _ = docstring_args[arg_name]
        if arg_type.__name__ != docstring_arg_type:
            raise ValueError(f"Docstring type '{docstring_arg_type}' does not match function signature type '{arg_type.__name__}' for argument '{arg_name}' for function '{func.__name__}'")

    # Generate a list of tuples (name, type, description, default) for each argument
    #TODO: use the signature to determine if an argument is optional or not
    args_list = [(arg_name, signature_args[arg_name], arg_desc, arg_default) for arg_name, (arg_type, arg_desc, arg_default) in docstring_args.items()]

    # get the return type and description
    signature_ret_type = signature.return_annotation

    docstring.returns
    docstring_ret_type = docstring.returns.type_name
    if signature_ret_type.__name__ != docstring_ret_type:
        raise ValueError(f"Docstring return type '{docstring_ret_type}' does not match function signature return type '{signature_ret_type.__name__}' for function '{func.__name__}'")
    
    # get the return type and description
    ret = (docstring.returns.return_name, signature_ret_type, docstring.returns.description)

    # get the docstring description and examples
    examples = [example.description for example in docstring.examples]
    desc = (docstring.short_description, docstring.long_description, examples)
    
    return args_list, ret, desc


# def validate_tool_call(func, args:dict|str, args_list) -> dict:
#     if isinstance(args, str):
#         #coerce the string to the correct type
#         pdb.set_trace()
#         name, type, desc, default = args_list[0]
#         pdb.set_trace()
#         args = {name: type(args)}

#     #validate that the arguments are correct for this tool
#     for arg_name, arg_type, arg_desc in args_list:
#         if arg_name not in args:
#             raise ValueError(f"Missing argument '{arg_name}' for tool '{func.__name__}'")

#         if not isinstance(args[arg_name], arg_type):
#             raise ValueError(f"Argument '{arg_name}' for tool '{func.__name__}' must be of type {arg_type.__name__}, got {type(args[arg_name]).__name__}")
    
#     return args


def make_tool_dict(tools:list[Callable]) -> dict[str, Callable]:
    """
    Create a dictionary of tools from a list of tool functions.

    Tries to use the '_name' attribute of the @tool function, otherwise uses the function name.

    Args:
        tools (list[Callable]): A list of tool functions

    Returns:
        dict[str, Callable]: A dictionary of tools
    """
    tool_dict = {}
    for tool in tools:
        name = getattr(tool, '_name', tool.__name__)
        if name in tool_dict:
            raise ValueError(f"Tool name '{name}' is already in use")
        tool_dict[name] = tool
    return tool_dict




from easyrepl import readl
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
    return readl(prompt=f'{query} ')



from datetime import datetime as dt
import pytz
@tool()
def datetime(format:str='%Y-%m-%d %H:%M:%S %Z', timezone:str='UTC') -> str:
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
    return dt.now(tz).strftime(format)


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
    return dt.now().timestamp()




@tool()
def fib_n(n:int) -> int:
    """
    generate the nth fibonacci number

    Args:
        n (int): The index of the fibonacci number to get

    Returns:
        int: The nth fibonacci number

    Examples:
        >>> fib_n(10)
        55
        >>> fib_n(20)
        6765
    """
    n0 = 0
    n1 = 1
    for _ in range(n):
        n0, n1 = n1, n0 + n1

    return n0



@tool()
def example_tool(arg1:int, arg2:str='', arg3:dict=None) -> int:
    """
    Simple 1 sentence description of the tool

    More detailed description of the tool. This can be multiple lines.
    Explain more what the tool does, and what it is used for.

    Args:
        arg1 (int): Description of the first argument.
        arg2 (str): Description of the second argument. Defaults to ''.
        arg3 (dict): Description of the third argument. Defaults to {}.

    Returns:
        int: Description of the return value

    Examples:
        >>> example_tool(1, 'hello', {'a': 1, 'b': 2})
        3
        >>> example_tool(2, 'world', {'a': 1, 'b': 2})
        4
    """
    return 42

@tool()
def calculator(expression:str) -> float:
    """
    A simple calculator tool. Can perform basic arithmetic

    Expressions must contain exactly:
    - one left operand. Can be a float or integer
    - one operation. Can be one of + - * / ^ %
    - one right operand. Can be a float or integer

    multiple chained operations are not currently supported.

    Expressions may not contain parentheses, or multiple operations. 
    If you want to do a complex calculation, you must do it in multiple steps.

    Args:
        expression (str): A string containing a mathematical expression.

    Returns:
        float: The result of the calculation
    
    Examples:
        >>> calculator('22/7')
        3.142857142857143
        >>> calculator('3.24^2')
        10.4976
        >>> calculator('3.24+2.5')
        5.74
    """

    #ensure that only one operation is present
    ops = [c for c in expression if c in '+-*/^%']
    if len(ops) > 1:
        raise ValueError(f"Invalid expression, too many operators. Expected exactly one of '+ - * / ^ %', found {', '.join(ops)}")
    if len(ops) == 0:
        raise ValueError(f"Invalid expression, no operation found. Expected one of '+ - * / ^ %'")
    
    op = ops[0]

    _a, _b = expression.split(op)
    a = float(_a)
    b = float(_b)

    if op == '+':
        return a+b
    elif op == '-':
        return a-b
    elif op == '*':
        return a*b
    elif op == '/':
        return a/b
    elif op == '^':
        return a**b
    elif op == '%':
        return a%b
    



def test():
    for t in [ask_user, datetime, timestamp, fib_n, example_tool, calculator]:
        print(get_tool_prompt_description(t))
        print()



if __name__ == '__main__':
    test()