import inspect
from docstring_parser import parse as parse_docstring
from rich import traceback; traceback.install(show_locals=True)
from textwrap import indent
from typing import Callable, Any
import functools

import pdb


#TODO: separate system tools from library tools from user tools
#      ideally system tools will have no library dependencies

#TODO: parse extra long manual page from doc string. man_page can be seen by calling man <tool_name>
# man_page:str|None=None
# wrapper._man_page = man_page


def tool(*, name:str|None=None):
    """
    Decorator to convert a function into a tool for ReAct agents to use.

    Usage:
    ```
        @tool()
        def my_tool(arg:type) -> type:
            '''
            Short description of the tool

            Long description of the tool

            Args:
                arg (type): Description of the argument

            Returns:
                type: Description of the return value

            Examples:
                optional description of the example
                >>> my_tool(arg)
                result
            '''
    ```
    """
    def decorator(obj):
        # determine if function, method, or class, and return the appropriate wrapper
        if inspect.isfunction(obj):
            if is_class_method(obj):
                return make_method_tool_wrapper(obj, name)
            else:
                return make_func_tool_wrapper(obj, name)        
        if inspect.isclass(obj):
            return make_class_tool_wrapper(obj, name)
        
        raise TypeError(f"tool decorator can only be applied to functions or classes. Got {obj} of type {type(obj)}")
    
    return decorator


def make_func_tool_wrapper(func:Callable, name:str|None=None):
    def wrapper(args:dict|list|str|int|float|bool|None):
        """Output from LLM will be dumped into a json object. Depending on object type, call func accordingly."""
        
        if isinstance(args, dict):
            result = func(**args)
        elif isinstance(args, list):
            result = func(*args)
        elif isinstance(args, (str, int, float, bool)):
            result = func(args)
        elif args is None:
            result = func()
        else:
            raise TypeError(f"args must be a valid json object type (dict, list, str, int, float, bool, or None). Got {type(args)}")

        #convert the result to a string if it is not already a string
        if not isinstance(result, str):
            result = str(result)

        return result
    
    #attach usage description to the wrapper function
    args_list, ret, desc = get_tool_signature(func)

    wrapper._name = name if name else func.__name__
    wrapper._is_function_tool = True
    wrapper._args_list = args_list
    wrapper._ret = ret
    wrapper._desc = desc

    return wrapper

def make_method_tool_wrapper(func:Callable, name:str|None=None):
    def wrapper(self, args:dict|list|str|int|float|bool|None):
        """Output from LLM will be dumped into a json object. Depending on object type, call func accordingly."""
        
        if isinstance(args, dict):
            result = func(self, **args)
        elif isinstance(args, list):
            result = func(self, *args)
        elif isinstance(args, (str, int, float, bool)):
            result = func(self, args)
        elif args is None:
            result = func(self)
        else:
            raise TypeError(f"args must be a valid json object type (dict, list, str, int, float, bool, or None). Got {type(args)}")

        #convert the result to a string if it is not already a string
        if not isinstance(result, str):
            result = str(result)

        return result
    
    #attach usage description to the wrapper function
    args_list, ret, desc = get_tool_signature(func)

    wrapper._name = name if name else func.__name__
    wrapper._is_method_tool = True
    wrapper._args_list = args_list
    wrapper._ret = ret
    wrapper._desc = desc

    return wrapper



def make_class_tool_wrapper(cls:type, name:str|None=None):
    #basically just add metadata, and return the class
    #metadata should include a list of the class's tool methods

    #get the list of @tool methods in the class
    methods = inspect.getmembers(cls, predicate=inspect.isfunction)
    methods = [method for name, method in methods if is_tool(method)]    
    
    # get the class docstring description
    docstring = inspect.getdoc(cls)

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        # create an instance of the class
        c = cls(*args, **kwargs)

        # attach the metadata to the instance
        c._name = name if name else cls.__name__
        c._is_class_tool_instance = True
        c._docstring = docstring
        c._tool_methods = methods
    
        return c
    
    # attach the metadata to the class
    wrapper._name = name if name else cls.__name__
    wrapper._is_class_tool = True
    wrapper._docstring = docstring
    wrapper._tool_methods = methods
    
    return wrapper



def is_class_method(func:Callable) -> bool:
    """checks if a function is part of a class, or a standalone function"""
    assert inspect.isfunction(func), f"is_class_method can only be used on functions. Got {func}"
    return func.__qualname__ != func.__name__

def is_tool(obj:Callable|type) -> bool:
    """checks if an object is a tool function, tool method, or tool class (may not be an instance of a class tool)"""
    return hasattr(obj, '_is_function_tool') or hasattr(obj, '_is_method_tool') or hasattr(obj, '_is_class_tool') or hasattr(obj, '_is_class_tool_instance')
        

def get_tool_name(obj:Callable|type) -> str:
    """Get the name of the tool, either from the @tool _name field, or the __name__ attribute"""
    assert is_tool(obj), f"get_tool_name can only be used on decorated @tools. Got {obj}"
    return getattr(obj, '_name')

def get_tool_names(obj:Callable|type) -> list[str]:
    """
    Get the tool name, or all method names if tool is a class tool
    """
    assert is_tool(obj), f"get_tool_name can only be used on decorated @tools. Got {obj}"
    
    if hasattr(obj, '_is_class_tool') or hasattr(obj, '_is_class_tool_instance'):
        cls_name = get_tool_name(obj)

        #construct names as cls_name.method_name
        names = [f"{cls_name}.{get_tool_name(method)}" for method in obj._tool_methods]

        return names
    
    #otherwise, just return the name of the tool
    return [get_tool_name(obj)]

"""
use cases for class tools

@tool()
class MyStaticClass: ...

@tool()
class MyInstanceClass: ...

t = MyInstanceClass(arg0, arg1, ...)

ReActAgent(tools=[MyStaticClass, t])

i.e. the tool could be the original class or an instance. 
If it is the original class, then the agent will make a new instance, else use the instance provided
"""




def get_tool_prompt_description(obj:Callable|type|Any):
    if hasattr(obj, '_is_class_tool'):
        return get_tool_class_prompt_description(obj)
    if hasattr(obj, '_is_function_tool') or hasattr(obj, '_is_method_tool'):
        return get_tool_func_prompt_description(obj)
    if hasattr(obj, '_is_class_tool_instance'):
        return get_tool_class_prompt_description(obj)

    raise TypeError(f"get_tool_prompt_description can only be used on @tools. Got {obj}")

def get_tool_func_prompt_description(func:Callable):
    assert is_tool(func), f"Function {func.__name__} does not have the @tool decorator attached"
    assert inspect.isfunction(func), f"get_tool_func_prompt_description can only be used on functions. Got {func}"

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
    if ret_type is None:
        chunks.append("None")
    else:
        chunks.append(f"({ret_type.__name__}) {ret_description}")


    ############### EXAMPLES ###############
    #TODO: examples need to be parsed...


    return ''.join(chunks)


def get_tool_class_prompt_description(cls:type):
    """
    Get the prompt description for a class tool, including all of its tool methods
    
    Args:
        cls (type|Any): The class tool to get the description for, or an instance of a class tool
    
    Returns:
        str: The prompt description for the class tool
    """
    assert hasattr(cls, '_is_class_tool') or hasattr(cls, '_is_class_tool_instance'), f"class or instance {cls} does not have the @tool decorator attached"

    #class description is as follows:
    #   class_name:
    #       full unmodified class docstring
    #       methods:
    #           <get_tool_func_prompt_description for each method>

    chunks = []
    tab = "    "

    ############### NAME/DESCRIPTION ###############
    chunks.append(f"{cls._name} (class):\n")
    if cls._docstring:
        chunks.append(f"{indent(cls._docstring, tab)}\n\n")

    #################### METHODS ####################
    chunks.append(f"{tab}methods:\n")
    for method in cls._tool_methods:
        method_str = get_tool_func_prompt_description(method)
        chunks.append(f"{indent(method_str, tab*2)}\n\n")

    #strip trailing whitespace
    return ''.join(chunks).rstrip()




def get_tool_signature(func:Callable) -> tuple[list[tuple[str, type, str|None, str|None]], tuple[str|None, type|None, str|None], tuple[str, str|None, list[str]]]:
    """
    Check that the docstring and function signature match for a tool function, and return all function information.

    Args:
        func (function): The function to check and extract information from
        ignore_self (bool): If True, ignore the first argument of the function if it is named 'self'

    Returns:
        args_list: A list of tuples (name, type, description, default) for each argument
        ret: A tuple (name, type, description) for the return value
        desc: A tuple (short_description, long_description, examples) from the docstring for the function
    """
    assert inspect.isfunction(func), f"get_tool_signature can only be used on functions. Got {func}"

    # get the function signature from the function and the docstring
    docstring = parse_docstring(func.__doc__)
    signature = inspect.signature(func)

    # determine if self needs to be ignored
    ignore_self: bool = is_class_method(func)

    # Extract argument information from the docstring
    docstring_args = {arg.arg_name: (arg.type_name, arg.description, arg.default) for arg in docstring.params}

    # Extract argument information from the signature (ignore self from class methods)
    signature_args = {k: v.annotation for i, (k, v) in enumerate(signature.parameters.items()) if not (i == 0 and ignore_self and k == 'self')}

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

    # # if docstring.returns:
    # if signature_ret_type == inspect.Signature.empty and docstring.returns is None:
    #     ...
    try:
        docstring_ret_type = docstring.returns.type_name
    except AttributeError:
        docstring_ret_type = '_empty'
    if signature_ret_type.__name__ != docstring_ret_type:
        raise ValueError(f"Docstring return type '{docstring_ret_type}' does not match function signature return type '{signature_ret_type.__name__}' for function '{func.__name__}'")
    
    # get the return type and description
    if docstring.returns is None:
        ret = (None, None, None)
    else:
        ret = (docstring.returns.return_name, signature_ret_type, docstring.returns.description)

    # get the docstring description and examples
    examples = [example.description for example in docstring.examples]
    desc = (docstring.short_description, docstring.long_description, examples)
    
    return args_list, ret, desc



def make_tool_dict(tools:list[Callable|type|Any]) -> dict[str, Callable]:
    """
    Create a dictionary of tools from a list of tool functions.

    Tries to use the '_name' attribute of the @tool function, otherwise uses the function name.

    Args:
        tools (list[Callable|type|Any]): A list of tool functions, or tool classes, or instances of tool classes.

    Returns:
        dict[str, Callable]: A dictionary of tools. Class methods of class tools are included as separate functions.
    """
    #TODO: extract methods from any class tools
    tool_dict = {}
    for tool in tools:
        assert is_tool(tool), f"make_tool_dict can only be used on tools. Got {tool}"
        name = getattr(tool, '_name')
        
        
        if hasattr(tool, '_is_function_tool'):
            if name in tool_dict:
                raise ValueError(f"Tool name '{name}' is already in use")
            tool_dict[name] = tool
            continue
        
        if hasattr(tool, '_is_method_tool'):
            #TODO: not sure if methods should be allowed since they need usually need a class instance...
            raise NotImplementedError("Free-floating tool methods are not yet supported")
            pdb.set_trace()
        

        # handle if instance of class or class itself
        if hasattr(tool, '_is_class_tool'):
            instance = tool()
        else:
            assert hasattr(tool, '_is_class_tool_instance'), f"Tool {tool} is not a function, method, or class tool"
            instance = tool

        for _, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not hasattr(method, '_name'):
                continue
            method_name = f'{name}.{method._name}'
            if method_name in tool_dict:
                raise ValueError(f"Tool name '{method_name}' is already in use")
            tool_dict[method_name] = method


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


#TODO: still not great since have to include cls in the signature
class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


import math
# @tool()
class Math:
    def __init__(self):
        pass

    # @tool
    @staticmethod
    def sin(x:float) -> float:
        """
        Calculate the sine of x

        Args:
            x (float): The angle in radians

        Returns:
            float: The sine of x
        """
        return math.sin(x)

    # @tool
    @staticmethod
    def cos(x:float) -> float:
        """
        Calculate the cosine of x

        Args:
            x (float): The angle in radians

        Returns:
            float: The cosine of x
        """
        return math.cos(x)
    
    # @tool
    @staticmethod
    def tan(x:float) -> float:
        """
        Calculate the tangent of x

        Args:
            x (float): The angle in radians

        Returns:
            float: The tangent of x
        """
        return math.tan(x)
    
    # @tool
    @classproperty #TODO: still not great since have to include cls in the signature
    def pi(*_) -> float:
        """
        Get the value of pi

        Returns:
            float: The value of pi
        """
        return math.pi



# @tool()
class StatefulToolExample:
    def __init__(self, i:int, s:str):
        self.i = i
        self.s = s

    # @tool()
    def inc(self) -> int:
        """
        increment the internal counter

        Returns:
            int: The new value of the internal counter
        """
        self.i += 1
        return self.i

    # @tool()
    def set_i(self, i:int):
        """
        set the internal counter

        Args:
            i (int): The new value of the internal counter
        """
        self.i = i

    # @tool()
    def set_s(self, s:str):
        """
        set the internal string

        Args:
            s (str): The new value of the internal string
        """
        self.s = s

    # @tool()
    def get_i(self) -> int:
        """
        get the internal counter
        
        Returns:
            int: The value of the internal counter
        """
        return self.i
    
    # @tool()
    def get_s(self) -> str:
        """
        get the internal string

        Returns:
            str: The value of the internal string
        """

        return self.s


from random import random
@tool()
class Jackpot:
    """
    A simple slot machine game
    
    Start with 100 chips, and make bets to try and win more chips.
    """
    def __init__(self, chips:float=100, win_table:list[tuple[float,float]]=[(0.01, 20), (0.02, 10), (0.05, 4.5), (0.2, 1.25)]):
        self._initial_chips = chips
        self.chips = chips
        self.win_table = win_table

    @tool()
    def spin(self, bet:float) -> float:
        """
        Spin the slot machine

        Args:
            bet (float): The amount to bet. Must be less than or equal to the current amount of chips in your wallet

        Returns:
            float: The amount won or lost
        """
        if bet > self.chips:
            raise ValueError(f"Bet must be less than or equal to the current winnings. Bet: {bet}, Winnings: {self.winnings}")
        
        spin = random()
        total_prob = 0
        multiplier = -1
        for prob, win_multiplier in self.win_table:
            total_prob += prob
            if spin <= total_prob:
                multiplier = win_multiplier
                break
                
        self.chips += bet*multiplier            
        return bet*multiplier
            
    @tool()
    def get_chips(self) -> float:
        """
        Get the current amount of chips in your wallet

        Returns:
            float: The current amount of chips in your wallet
        """
        return self.chips
    
    @tool()
    def reset(self):
        """
        Reset the game back to the initial number of chips
        """
        self.chips = self._initial_chips


#TODO: stateful class example
#      user would call like so:
"""
@class_tool()
class some_tool: ...


t = some_tool(arg1, arg2, ...)
tools = [t, fib_n, calculator, ...]
"""


def test():
    for t in [ask_user, datetime_tool, timestamp, fib_n, example_tool, calculator]:
        print(get_tool_prompt_description(t))
        print()



if __name__ == '__main__':
    test()