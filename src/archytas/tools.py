import inspect
from docstring_parser import parse as parse_docstring
from rich import traceback; traceback.install(show_locals=True)
import pdb


#TODO: separate system tools from library tools from user tools
#      ideally system tools will have no library dependencies



def tool(**todo):
    """
    Converts a function into a tool for ReAct agents to use.
    """
    def decorator(func):
        args_list = validate_tool_signature(func)
        def wrapper(args:dict|str):
            args = validate_tool_call(func, args, args_list)
            
            result = func(**args)

            # assert isinstance(result, str), f"Tool {func.__name__} must return a string"
            #TODO: for now, just coerce the result to a string
            if not isinstance(result, str):
                result = str(result)

            return result
        # pdb.set_trace()
        return wrapper
    
        #attach usage description to the wrapper function

    return decorator


def validate_tool_signature(func):
    # Validate that the signature matches the docstring
    docstring = parse_docstring(func.__doc__)
    signature = inspect.signature(func)

    # Extract argument information from the docstring
    docstring_args = {arg.arg_name: (arg.type_name, arg.description, arg.default) for arg in docstring.params}

    # Extract argument information from the signature
    signature_args = {k: v.annotation for k, v in signature.parameters.items()}

    # Check if the docstring argument names match the signature argument names
    if set(docstring_args.keys()) != set(signature_args.keys()):
        raise ValueError("Docstring argument names do not match function signature argument names")

    # Check if the docstring argument types match the signature argument types
    for arg_name, arg_type in signature_args.items():
        docstring_arg_type, _, _ = docstring_args[arg_name]
        if arg_type.__name__ != docstring_arg_type:
            raise ValueError(f"Docstring type '{docstring_arg_type}' does not match function signature type '{arg_type.__name__}' for argument '{arg_name}'")

    # Generate a list of tuples (name, type, description, default) for each argument
    args_list = [(arg_name, signature_args[arg_name], arg_desc, arg_default) for arg_name, (arg_type, arg_desc, arg_default) in docstring_args.items()]

    return args_list


def validate_tool_call(func, args:dict|str, args_list) -> dict:
    if isinstance(args, str):
        #coerce the string to the correct type
        name, type, _ = args_list[0]
        pdb.set_trace()
        args = {name: type(args)}

    #validate that the arguments are correct for this tool
    for arg_name, arg_type, arg_desc in args_list:
        if arg_name not in args:
            raise ValueError(f"Missing argument '{arg_name}' for tool '{func.__name__}'")

        if not isinstance(args[arg_name], arg_type):
            raise ValueError(f"Argument '{arg_name}' for tool '{func.__name__}' must be of type {arg_type.__name__}, got {type(args[arg_name]).__name__}")
    

    return args

def get_tool_description(): ...


from easyrepl import readl
def ask_user(query:str) -> str:
    """Ask the user a question. Returns the user's response"""
    return readl(prompt=f'{query} ')

from datetime import datetime
import pytz
#TODO: arguments need to be parsed. input should be a json with {format:str, timezone:str}
def datetime_tool(kwargs) -> str:
    def datetime_tool(format:str='%Y-%m-%d %H:%M:%S %Z', timezone:str='UTC') -> str:
        """Returns the current date and time in the specified format. 
        Defaults to YYYY-MM-DD HH:MM:SS format. 
        TODO: See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for more information.
        TODO: list of valid timezones: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        default timezone is UTC
        """
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime(format)

    return datetime_tool(**kwargs)


def timestamp(_) -> str:
    """Returns the current unix timestamp in seconds"""
    return str(datetime.now().timestamp())


@tool()
def test_calculator(expression:str='2+2') -> str:
    """
    just a function signature for testing the @tool decorator

    Args:
        expression (str, optional): A string containing a mathematical expression. Defaults to '2+2'.

    Returns:
        str: The result of the calculation
    """
    ...


def calculator(expression:str) -> str:
    """
    A simple calculator tool. Can perform basic arithmetic

    Args:
        expression (str): A string containing a mathematical expression.

    Returns:
        str: The result of the calculation
    
    Expressions must contain exactly:
    - one left operand. Can be a float or integer
    - one operation. Can be one of + - * / ^ %
    - one right operand. Can be a float or integer

    Expressions may not contain any spaces.

    Expressions may not contain parentheses, or multiple operations. 
    If you want to do a complex calculation, you must do it in multiple steps.

    examples:
    input: 22/7
    output: 3.142857142857143

    input: 3.24^2
    output: 10.4976
    
    input: 3.24+2.5
    output: 5.74
    """

    for op in '+-*/^%':
        if op in expression:
            break
    else:
        return "Invalid expression. No operation found"
    
    if op not in '+-*/^%':
        raise ValueError(f"Invalid operation. Expected one of '+-*/^%', got '{op}'")
    
    _a, _b = expression.split(op)
    a = float(_a)
    b = float(_b)

    if op == '+':
        return str(a+b)
    elif op == '-':
        return str(a-b)
    elif op == '*':
        return str(a*b)
    elif op == '/':
        return str(a/b)
    elif op == '^':
        return str(a**b)
    elif op == '%':
        return str(a%b)
    



def test():
    #call the calculator tool with a valid expression
    test_calculator('22/7')

    #call the calculator tool with invalid arguments
    try:
        test_calculator({"a":22,"b":7,"op":'div'})
    except ValueError as e:
        print(e)

    pdb.set_trace()


if __name__ == '__main__':
    test()